"""Settings finder for the detection: mark pores and microstructure by hand, get the best parameters.

    python tools/find_settings.py [image]        (default: first image of input.path in config.yaml)

Runs sample (ROI, scale) and the preprocessing chain of config.yaml on the image, nothing else.
Then you mark in a window (matplotlib, toolbar for zoom / pan):

    1        mode "pore":           circle a pore loosely, with a little matrix around it
    2        mode "microstructure": circle something that is NOT a pore (dark network, scratch, …)
    u        undo the last mark
    t        switch the picture: original <-> preprocessed gray image (what the detector sees)
    Enter    try all parameter combinations and print the best one
    e        show / hide the detection with the best parameters

To draw, zoom / pan in the toolbar must be off (click the active tool again).

Every combination runs the real detector of config.yaml on a small area around each mark:
    pore           correct if something is detected inside AND it does not reach over the circle
                   (reaching over = grown into the network)
    microstructure correct if nothing is detected inside
The best combination has the most correct marks; among equal ones the one whose pores fit best
(IoU) to the expected pore in each circle: the area darker than half-way between the pore core
and the matrix on the circle line (the edge of a blurred pore lies at half contrast). Without it
the size would be free: the circles only say "found / leaked", not how large. Coarse grid first,
then a fine grid around the best.

Simple tool, not part of the program: it uses core internals and is not tested like the pipeline.
"""

import argparse
import itertools
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pore_detection.core import imaging, io, registry  # noqa: E402
from pore_detection.core.config import Config, load_config  # noqa: E402
from pore_detection.core.pipeline import _bounding_box, resolve_path  # noqa: E402
from pore_detection.core.types import ImageContext  # noqa: E402

PAD_PX = 40                 # area around a mark that the detector sees (a pore must reach over the circle)
LEAK_SHARE = 0.25           # leaked: detected outside the circle > this share of the area inside (a loose
                            # circle is not exact – a few px over it, or into a neighbour pore, is no leak)
# searched parameters per detector: name -> (coarse values, fine step). Fine: +-1 coarse step around the best
GRIDS = {
    "hysteresis": {
        "seed_threshold": (np.arange(0.20, 0.701, 0.05), 0.0125),
        "grow_threshold": (np.arange(0.30, 0.901, 0.05), 0.0125),
        "spot_threshold": (np.array([0.0, 0.08, 0.14, 0.2, 0.26, 0.32, 0.4]), 0.02),
    },
    "threshold": {
        "threshold": (np.arange(0.10, 0.901, 0.05), 0.005),
    },
}
VALID = {"hysteresis": lambda p: p["grow_threshold"] >= p["seed_threshold"]}
KINDS = {"1": ("pore", "#ff1a1a"), "2": ("microstructure", "#1c7ed6")}
CONTROLS = __doc__.split("\n\n")[3]     # the key list above


# --- image: exactly as in the pipeline up to the end of preprocessing ---

def prepare(config: Config, path: Path):
    """ImageContext (cropped to the sample) and the preprocessed gray image."""
    info, original = io.load_image(path, path.parent.name)
    roi = registry.create("sample.roi", config.sample.roi).detect_roi(info, original)
    scale = registry.create("sample.scale", config.sample.scale).detect_scale(info, original)
    if scale.scale_mask is not None:
        roi = roi & ~scale.scale_mask
    top, bottom, left, right = _bounding_box(roi)
    ctx = ImageContext(replace(info, offset_px=(top, left)), original[top:bottom, left:right],
                       roi[top:bottom, left:right], scale.pixel_size_um, None)
    gray = imaging.to_gray(ctx.original, config.input.gray_weights)
    for entry in config.preprocessing:
        gray = registry.create("preprocessing", entry).process(ctx, gray)
    return ctx, gray


class Mark:
    """One circled area: kind, polygon (picture px), its area with PAD_PX around it."""

    def __init__(self, kind: str, points: np.ndarray, ctx: ImageContext, gray: np.ndarray):
        self.kind, self.points = kind, points
        height, width = gray.shape
        x0, y0 = np.floor(points.min(axis=0)).astype(int) - PAD_PX
        x1, y1 = np.ceil(points.max(axis=0)).astype(int) + PAD_PX + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        self.gray = gray[y0:y1, x0:x1]
        self.inside = np.zeros(self.gray.shape, np.uint8)
        cv2.fillPoly(self.inside, [np.round(points - (x0, y0)).astype(np.int32)], 1)
        self.inside = self.inside.view(bool)
        top, left = ctx.info.offset_px
        self.ctx = ImageContext(replace(ctx.info, offset_px=(top + y0, left + x0)), ctx.original[y0:y1, x0:x1],
                                ctx.roi_mask[y0:y1, x0:x1], ctx.pixel_size_um, None)
        self.expected = self._half_contrast() if kind == "pore" else None

    def _half_contrast(self) -> np.ndarray:
        """Expected pore inside the circle: darker than half-way between the pore core (darkest
        pixels) and the matrix on the circle line – the edge of a blurred pore lies at half contrast.
        Only areas that reach the core (not dark network that merely lies in the circle)."""
        sample = self.inside & self.ctx.roi_mask
        line = sample & ~cv2.erode(self.inside.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
        if not line.any() or not sample.any():
            return sample
        core = np.percentile(self.gray[sample], 2)
        matrix = np.median(self.gray[line])
        dark = sample & (self.gray < (core + matrix) / 2)
        _, labels = cv2.connectedComponents(dark.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        cores = np.unique(labels[dark & (self.gray <= core + 0.25 * (matrix - core))])
        return np.isin(labels, cores[cores > 0])

    def check(self, detector) -> tuple[bool, float]:
        """(correct, agreement of the detected pore with the expected one, IoU 0..1 – 0 for
        microstructure and wrong marks)."""
        found = detector.detect(self.ctx, self.gray) & self.ctx.roi_mask
        inside = np.count_nonzero(found & self.inside)
        if self.kind == "microstructure":
            return inside == 0, 0
        if inside == 0:
            return False, 0                                 # pore missed
        _, labels = cv2.connectedComponents(found.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        touched = np.unique(labels[found & self.inside])
        outside = np.count_nonzero(np.isin(labels, touched) & ~self.inside)
        ok = outside <= LEAK_SHARE * inside                 # far over the circle = leaked
        detected = found & self.inside
        union = np.count_nonzero(detected | self.expected)
        return ok, (np.count_nonzero(detected & self.expected) / union if ok and union else 0.0)


# --- search ---

def detector_with(config: Config, values: dict):
    return registry.create("detection", Config({**config.detection.to_dict(), **values}))


def score(config: Config, marks: list[Mark], values: dict) -> tuple[list[bool], float]:
    """Correct per mark and the summed size agreement (IoU) of the correct pore marks."""
    detector = detector_with(config, values)
    checks = [mark.check(detector) for mark in marks]
    return [ok for ok, _ in checks], sum(area for _, area in checks)


def search(config: Config, marks: list[Mark], grid: dict, pool) -> tuple[dict, list[bool]]:
    """Most correct marks; among equal ones the best size agreement with the half-contrast pores."""
    names = list(grid)
    valid = VALID.get(config.detection.name, lambda p: True)
    combos = [dict(zip(names, map(float, values))) for values in itertools.product(*grid.values())]
    combos = [c for c in combos if valid(c)]
    results = list(pool.map(lambda c: score(config, marks, c), combos))
    best = max(range(len(combos)), key=lambda i: (sum(results[i][0]), results[i][1]))
    return combos[best], results[best][0]


def find(config: Config, marks: list[Mark]) -> tuple[dict, list[bool]]:
    name = config.detection.name
    if name not in GRIDS:
        raise SystemExit(f"no search grid for detector '{name}' – add one to GRIDS in {Path(__file__).name}")
    grids = GRIDS[name]
    with ThreadPoolExecutor() as pool:
        coarse, _ = search(config, marks, {n: values for n, (values, _) in grids.items()}, pool)
        fine = {}
        for n, (values, step) in grids.items():
            width = values[1] - values[0] if len(values) > 1 else 0         # +-1 coarse step
            low, high = max(values.min(), coarse[n] - width), min(values.max(), coarse[n] + width)
            fine[n] = np.round(np.arange(low, high + step / 2, step), 4)
        return search(config, marks, fine, pool)


# --- window ---

class Finder:
    def __init__(self, config: Config, ctx: ImageContext, gray: np.ndarray):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import LassoSelector

        self.config, self.ctx, self.gray = config, ctx, gray
        self.marks: list[Mark] = []
        self.patches = []
        self.mode = "1"
        self.best: dict | None = None
        self.overlay = None
        rgb = ctx.original if ctx.original.dtype == np.uint8 else (ctx.original >> 8).astype(np.uint8)
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.figure.canvas.manager.set_window_title("Settings finder")
        self.picture = self.ax.imshow(rgb, interpolation="nearest")
        self.views = [(rgb, None), (gray, "gray")]
        self.view = 0
        self.ax.set_axis_off()
        self.figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
        self.lasso = LassoSelector(self.ax, self.on_lasso, props={"color": "white", "linewidth": 1})
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_title()
        plt.show()

    def update_title(self, extra: str = ""):
        kind, color = KINDS[self.mode]
        counts = {k: sum(m.kind == k for m in self.marks) for k, _ in KINDS.values()}
        self.ax.set_title(f"mode: {kind}  (1 pore · 2 microstructure · u undo · t picture · Enter search · e result)"
                          f"   –   {counts['pore']} pores, {counts['microstructure']} microstructure  {extra}",
                          color=color, fontsize=9)
        self.figure.canvas.draw_idle()

    def on_lasso(self, vertices):
        points = np.asarray(vertices, dtype=float)
        if len(points) < 3:
            return
        kind, color = KINDS[self.mode]
        self.marks.append(Mark(kind, points, self.ctx, self.gray))
        closed = np.vstack([points, points[:1]])
        line, = self.ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.2)
        number = self.ax.text(*points.max(axis=0), str(len(self.marks)), color=color, fontsize=8)
        self.patches.append((line, number))
        self.update_title()

    def on_key(self, event):
        if event.key in KINDS:
            self.mode = event.key
            self.update_title()
        elif event.key == "u" and self.marks:
            self.marks.pop()
            for artist in self.patches.pop():
                artist.remove()
            self.update_title()
        elif event.key == "t":
            self.view = 1 - self.view
            image, cmap = self.views[self.view]
            self.picture.set_data(image)
            self.picture.set_cmap(cmap or "viridis")
            self.picture.set_clim(0, 1 if cmap else 255)
            self.figure.canvas.draw_idle()
        elif event.key == "enter":
            self.run_search()
        elif event.key == "e":
            self.toggle_result()

    def run_search(self):
        if not any(m.kind == "pore" for m in self.marks):
            print("mark at least one pore first")
            return
        self.update_title("– searching …")
        self.figure.canvas.flush_events()
        start = time.perf_counter()
        current = {n: float(self.config.detection.to_dict()[n]) for n in GRIDS.get(self.config.detection.name, {})}
        self.best, correct = find(self.config, self.marks)
        report("config.yaml now", current, score(self.config, self.marks, current)[0], self.marks)
        report("best", self.best, correct, self.marks)
        print(f"\n  ({time.perf_counter() - start:.1f} s)  paste into config.yaml:\n")
        print(f"detection:\n  name: {self.config.detection.name}")
        for name, value in {**self.config.detection.to_dict(), **self.best}.items():
            if name != "name":
                print(f"  {name}: {value:g}" if isinstance(value, float) else f"  {name}: {value}")
        if self.overlay is not None:
            self.overlay.remove()
            self.overlay = None
        self.update_title(f"– best: {sum(correct)}/{len(correct)} correct (e: show)")

    def toggle_result(self):
        if self.overlay is not None:
            self.overlay.remove()
            self.overlay = None
        elif self.best is not None:
            found = detector_with(self.config, self.best).detect(self.ctx, self.gray) & self.ctx.roi_mask
            layer = np.zeros((*found.shape, 4), np.float32)
            layer[found] = (1.0, 0.85, 0.0, 0.6)                # detected pores: yellow
            self.overlay = self.ax.imshow(layer, interpolation="nearest")
        self.figure.canvas.draw_idle()


def report(title: str, values: dict, correct: list[bool], marks: list[Mark]):
    text = ", ".join(f"{n} {v:g}" for n, v in values.items())
    print(f"\n{title}: {text}  ->  {sum(correct)} of {len(correct)} marks correct")
    for i, (mark, ok) in enumerate(zip(marks, correct), start=1):
        if not ok:
            reason = "detected" if mark.kind == "microstructure" else "missed or reaches over the circle"
            print(f"    #{i} {mark.kind}: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", nargs="?", help="image file (default: first image of input.path)")
    args = parser.parse_args()
    config = load_config()
    if args.image:
        path = Path(args.image)
    else:
        files = io.find_images(resolve_path(config.input.path), config.input.extensions)
        if not files:
            raise SystemExit(f"no image in {config.input.path}")
        path = files[0]
    print(f"{path}: sample + preprocessing …")
    ctx, gray = prepare(config, path)
    import matplotlib
    matplotlib.use("QtAgg")
    print(CONTROLS)
    Finder(config, ctx, gray)


if __name__ == "__main__":
    main()
