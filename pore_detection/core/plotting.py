"""Plots: draw the plot modules (stage "plots") into PNG images – fixed size and look for all.

A plot module only draws into the matplotlib Axes it gets (see core/step.py, Plot). Everything
else is done here: figure, look, checking the needed values, saving as PNG. matplotlib is
imported only when a plot is drawn, so runs without plots do not need it.

A plot that cannot be drawn (values missing, error in the plot) is skipped and reported; the
image itself does not fail – plots only show values that are already computed.

Plots over the whole ID (optional Plot.plot_id) are drawn at the end of a run and after every
manual correction, from images.json and the pores.json of all images (read back, never Excel).

Extra plots (optional Plot.extra_plots, e.g. a plot for every q): drawn and saved with the others,
in a subfolder of plots/ (key "all_q/critical_pores_q5"); the viewer shows only plots/<name>.png.

Also here: log classes and kernel density for distribution plots, so that every plot (image and
ID) computes them the same way – plots never import each other.
"""

import io
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from . import geometry

from .step import Plot

log = logging.getLogger("pore_detection")

SIZE_IN = (6.4, 4.0)        # figure size in inch; with DPI -> 960 x 600 px (sharp when scaled down)
DPI = 150
PAD_IN = 0.04               # white border around the plot when saved (tight)
# colors for plot modules: first = main series (accent of the viewer), then further series
COLORS = ("#0f6d8c", "#d9480f", "#5c940d", "#7c5cd6", "#868e96")
# scientific look: black frame on all four sides, no grid, ticks inside on all sides
STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal", "axes.labelsize": 9,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.color": "black", "ytick.color": "black", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.fontsize": 8, "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "figure.facecolor": "white", "savefig.facecolor": "white",
}


class MissingData(Exception):
    """Raised by a plot when values it needs are missing: the plot is skipped (not the image)."""


def available() -> bool:
    """matplotlib installed (needed as soon as plots are configured)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def variant_key(name: str, variant: str) -> str:
    """Key (= file stem in plots/) of a plot view: "pore_areas" or "pore_areas.normalized"."""
    return f"{name}.{variant}" if variant else name


def render(plots: list[Plot], image: dict, pores: pd.DataFrame, picture=None) -> dict[str, bytes]:
    """PNG bytes per plot view (variant_key) of one image, in the order of `plots`; a plot with a
    variant gives two views. Skipped plots are logged and left out. picture: callable that makes
    the SamplePicture (or None) – called once, only if a plot needs_picture."""
    made = {}

    def get_picture():
        if "picture" not in made:
            made["picture"] = picture() if picture is not None else None
        return made["picture"]

    def missing(step):
        return ([c for c in step.columns if _empty(image.get(c))]
                + [c for c in step.pore_columns if c not in pores.columns])

    def draw(step, ax, **variant):
        if step.needs_picture:
            variant["picture"] = get_picture()
        step.plot(ax, image, pores, **variant)
    return _render(_with_extras(plots, pores), image.get("image"), missing, draw)


def render_id(plots: list[Plot], images: pd.DataFrame, pores: pd.DataFrame) -> dict[str, bytes]:
    """PNG bytes per plot view over the whole ID: only plots with plot_id. images: images.json,
    pores: pores.json of all images with column "image"."""
    plots = [step for step in plots if hasattr(step, "plot_id")]

    def missing(step):
        return ([c for c in step.columns if c not in images.columns or images[c].isna().all()]
                + [c for c in step.pore_columns if c not in pores.columns])
    return _render(_with_extras(plots, pores), "ID", missing,
                   lambda step, ax, **variant: step.plot_id(ax, images, pores, **variant))


def _with_extras(plots: list[Plot], pores: pd.DataFrame) -> list[tuple[str, Plot]]:
    """(key, plot) of the configured plots, then the extra plots (optional Plot.extra_plots: saved in
    a subfolder of plots/, not shown in the viewer)."""
    items = [(step.name, step) for step in plots]
    for step in plots:
        if hasattr(step, "extra_plots"):
            try:
                items += list(step.extra_plots(pores).items())
            except Exception as error:
                log.warning("plot %s: extra plots skipped – %s: %s", step.name, type(error).__name__, error)
    return items


def _render(plots: list[tuple[str, Plot]], where: str, missing, draw) -> dict[str, bytes]:
    """plots: (key, plot); the key is the file stem in plots/ (with a subfolder for extra plots)."""
    if not plots:
        return {}
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure     # object API: no global pyplot state, safe in threads

    result = {}
    for key, step in plots:
        try:
            gaps = missing(step)
            if gaps:
                raise MissingData(f"values missing: {', '.join(gaps)}")
            views = {}
            for variant in ("", step.variant) if step.variant else ("",):
                with matplotlib.rc_context(STYLE):
                    figure = Figure(figsize=step.size_in or SIZE_IN, dpi=DPI)
                    ax = figure.add_subplot()
                    if variant:
                        draw(step, ax, variant=variant)
                    else:
                        draw(step, ax)
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=PAD_IN)
                views[variant_key(key, variant)] = buffer.getvalue()
            result.update(views)
        except MissingData as reason:
            log.warning("%s: plot %s skipped – %s", where, step.name, reason)
        except Exception as error:
            log.warning("%s: plot %s skipped – %s: %s", where, step.name, type(error).__name__, error)
    return result


# --- picture of the sample for plots (Plot.needs_picture) ---

PICTURE_MAX_PX = 1400       # long edge of the picture
PICTURE_MARGIN = 0.04       # embedding resin shown around the sample, share of its long side


@dataclass(frozen=True)
class SamplePicture:
    """The sample straightened along its frame (outer surface on top, x to the right), cropped to
    the sample plus a small margin of embedding resin, scaled down."""

    rgb: np.ndarray             # (H, W, 3) uint8
    pores: pd.DataFrame         # pore_id, x, y: centroid of every counted pore in picture px
    scale: float                # picture px per image px


def sample_picture(original: np.ndarray, labels: np.ndarray, frame) -> SamplePicture | None:
    """Picture of the sample for plots; None without a sample frame. original: full image (RGB),
    labels: counted pores in full image size."""
    if frame is None:
        return None
    length, thickness = frame.length_px, frame.thickness_px
    margin = PICTURE_MARGIN * max(length, thickness)
    scale = min(1.0, PICTURE_MAX_PX / (max(length, thickness) + 2 * margin))
    width, height = round(scale * (length + 2 * margin)), round(scale * (thickness + 2 * margin))

    small = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else original
    if small.dtype != np.uint8:
        small = (small >> 8).astype(np.uint8)
    # picture (u, v) -> sample frame (x = u/s - m, z = T + m - v/s) -> small image (scale * image px)
    (xy, xx), (zy, zx) = frame.x_axis, frame.z_axis
    oy, ox = frame.origin_px
    shift_x = scale * (ox - margin * xx + (thickness + margin) * zx)
    shift_y = scale * (oy - margin * xy + (thickness + margin) * zy)
    to_small = np.array([[xx, -zx, shift_x], [xy, -zy, shift_y]], dtype=np.float64)
    rgb = cv2.warpAffine(small, to_small, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    flat, ids = geometry.pixels(labels)
    if len(ids):
        rows, cols = np.divmod(flat, labels.shape[1])
        unique, index, count = np.unique(ids, return_inverse=True, return_counts=True)
        y = np.bincount(index, weights=rows) / count
        x = np.bincount(index, weights=cols) / count
        sx, sz = frame.to_sample(y, x)
        pores = pd.DataFrame({"pore_id": unique, "x": scale * (sx + margin), "y": scale * (thickness + margin - sz)})
    else:
        pores = pd.DataFrame({"pore_id": [], "x": [], "y": []})
    return SamplePicture(rgb, pores, scale)


# --- distributions: log classes and kernel density (Pore_Detection_V2) ---

MIN_CLASSES, MAX_CLASSES = 5, 24
CURVE_POINTS = 96


def class_count(values: np.ndarray) -> int:
    """Number of log classes after Freedman-Diaconis (width 2 IQR n^(-1/3) on the logarithms)."""
    if len(values) < 4:
        return max(1, len(values))
    logs = np.log10(values)
    span = logs.max() - logs.min()
    if span <= 0:
        return 1
    q1, q3 = np.percentile(logs, [25, 75])
    if q3 - q1 <= 0:                # more than half of the values equal: square root rule
        return max(4, min(20, round(math.sqrt(len(logs)))))
    width = 2.0 * (q3 - q1) * len(logs) ** (-1 / 3)
    return max(MIN_CLASSES, min(MAX_CLASSES, math.ceil(span / width)))


def class_edges(values: np.ndarray, classes: int) -> np.ndarray:
    """Log class edges, widened to smooth quarter decades (round axis labels)."""
    low, high = values.min(), values.max()
    if high <= low:
        high = low * 1.05
    lo = math.floor(math.log10(low) * 4) / 4
    hi = math.ceil(math.log10(high) * 4) / 4
    return 10 ** np.linspace(lo, hi, classes + 1)


def density_curve(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Gaussian kernel density on the logarithms over the range of `edges`: a true density per
    decade – the area under the curve over a log10 axis is 1 (times 100: % of the pores per decade,
    times n: pores per decade). It may exceed 1 where the values lie closer than one decade.
    None with fewer than 5 values or no spread."""
    logs = np.log10(values)
    n = len(logs)
    if n < 5 or len(edges) < 2:
        return None
    spread = logs.std(ddof=1)
    q1, q3 = np.percentile(logs, [25, 75])
    robust = (q3 - q1) / 1.349
    sigma = min(spread, robust) if robust > 0 else spread
    if sigma <= 0:
        return None
    h = 0.9 * sigma * n ** (-1 / 5)                 # Silverman
    lo, hi = math.log10(edges[0]), math.log10(edges[-1])
    x = np.linspace(lo, hi, CURVE_POINTS)
    factor = 1.0 / (n * h * math.sqrt(2 * math.pi))     # density 1/(n h sqrt(2 pi)) sum exp(-u²/2)
    height = np.empty(CURVE_POINTS)
    for i, point in enumerate(x):
        u = (point - logs) / h
        u = u[np.abs(u) < 5.0]                       # beyond that the contribution is negligible
        height[i] = factor * np.exp(-0.5 * u * u).sum()
    return 10 ** x, height


def id_pores(images: pd.DataFrame, read_table, results_dir) -> pd.DataFrame:
    """pores.json of all images of the ID, with column "image" (read back from the results)."""
    tables = []
    for name in images["image"]:
        table = read_table(results_dir / name / "pores.json")
        tables.append(table.assign(image=name))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame({"image": []})


def _empty(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)
