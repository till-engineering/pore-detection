"""Reading the results of one run – the viewer's only data source (docs/output_format.md).

Nothing is imported from pore_detection. Per image everything is read in full resolution: the
original (from the input folder), labels, rejected labels and ROI mask (about 1.5 s for 100 MP,
in a background thread). The overview shown when the whole image is visible is computed from
them here and never saved. After a manual correction the new state comes from the correction
session (apply_state), not from disk.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# file names, see docs/output_format.md
IMAGES_JSON = "images.json"
ID_VALUES_JSON = "id_values.json"
ERRORS_JSON = "errors.json"
CONFIG_USED = "config_used.yaml"
PORES_JSON = "pores.json"
LABELS_TIF = "labels.tif"
ROI_MASK_TIF = "roi_mask.tif"
REJECTED_LABELS_TIF = "rejected_labels.tif"
REJECTED_JSON = "rejected.json"
CORRECTIONS_JSON = "corrections.json"
PLOTS_DIR = "plots"         # in the image folder: <plot name>.png
MANUAL = "manual"           # filter name of pores rejected by hand

OVERVIEW_MAX_PX = 2048      # long edge of the overview computed for the whole-image view
MISSING_GRAY = 128          # base image if the original cannot be read any more


def read_table(path: Path) -> pd.DataFrame:
    """Column-oriented JSON table ({column: [values]})."""
    with open(path, encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def _decode(path: Path, flags: int) -> np.ndarray:
    """cv2.imread for non-ASCII paths; raises if the file cannot be read."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def read_rgb(path: Path) -> np.ndarray:
    """Any image as RGB uint8 (16 bit reduced to 8 bit, alpha dropped)."""
    image = _decode(path, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB)


def read_bool(path: Path) -> np.ndarray:
    return _decode(path, cv2.IMREAD_GRAYSCALE) > 0


@dataclass
class Run:
    """Everything of the run that does not depend on the shown image."""

    results_dir: Path
    images: pd.DataFrame        # one row per successful image
    config: dict
    config_text: str
    id_values: dict | None      # None after a cancelled run
    errors: pd.DataFrame

    @classmethod
    def load(cls, results_dir: Path) -> "Run":
        results_dir = Path(results_dir)
        config_text = (results_dir / CONFIG_USED).read_text(encoding="utf-8")
        id_path = results_dir / ID_VALUES_JSON
        return cls(
            results_dir=results_dir,
            images=read_table(results_dir / IMAGES_JSON),
            config=yaml.safe_load(config_text),
            config_text=config_text,
            id_values=json.loads(id_path.read_text(encoding="utf-8")) if id_path.exists() else None,
            errors=read_table(results_dir / ERRORS_JSON),
        )

    def plot_names(self) -> list[str]:
        """Plots of the run in the order of the config (one place each in the plot grid)."""
        return [entry["name"] for entry in self.config.get("plots") or []]

    def chain(self) -> list[tuple[str, dict]]:
        """Main methods: scale -> sample -> detector."""
        c = self.config
        return [("scale", c["sample"]["scale"]), ("sample (ROI)", c["sample"]["roi"]), ("detection", c["detection"])]

    def row(self, index: int) -> dict:
        return self._rows[index] if index in self._rows else self.images.iloc[index].to_dict()

    def set_row(self, index: int, row: dict) -> None:
        """Values after a manual correction (images.json is rewritten in the background)."""
        self._rows[index] = row

    @property
    def _rows(self) -> dict:
        return self.__dict__.setdefault("_corrected_rows", {})


@dataclass(frozen=True)
class Scale:
    """Scale of one image as found by the pipeline (columns of images.json)."""

    um_per_px: float | None
    label: str | None                           # text read from the scale label
    bar: tuple[int, int, int, int] | None       # measured bar: top, bottom, left, right
    box: tuple[int, int, int, int] | None       # scale overlay (not sample)

    @classmethod
    def from_row(cls, row: dict) -> "Scale":
        def value(key):
            v = row.get(key)
            return None if v is None or (isinstance(v, float) and v != v) else v

        def box(key):
            v = value(key)
            return tuple(int(x) for x in v) if v is not None else None
        um = value("pixel_size_um")
        return cls(float(um) if um is not None else None, value("scale_label"), box("scale_bar_px"), box("scale_box_px"))


@dataclass(frozen=True)
class Frame:
    """Coordinate system of the sample (columns frame_* of images.json, docs/interfaces.md 2.6).

    Full image px. x along the outer (top) edge to the right, z perpendicular upwards; z = 0 is
    the inner (bottom) edge, z = thickness_px the outer (top) edge.
    """

    origin_px: tuple[float, float]  # (y, x)
    angle_deg: float                # direction of x, counter-clockwise from the image x axis (as seen)
    thickness_px: float
    length_px: float
    thickness_um: float | None

    @classmethod
    def from_row(cls, row: dict) -> "Frame | None":
        origin = row.get("frame_origin_px")
        if origin is None or (isinstance(origin, float) and origin != origin):
            return None
        um = row.get("frame_thickness_um")
        return cls((float(origin[0]), float(origin[1])), float(row["frame_angle_deg"]),
                   float(row["frame_thickness_px"]), float(row["frame_length_px"]),
                   float(um) if um is not None and um == um else None)

    def to_image(self, x: float, z: float) -> tuple[float, float]:
        """Point of the sample frame -> (x, y) in full image px."""
        a = np.radians(self.angle_deg)
        return (float(self.origin_px[1] + x * np.cos(a) - z * np.sin(a)),
                float(self.origin_px[0] - x * np.sin(a) - z * np.cos(a)))


@dataclass
class Image:
    """One image, everything in full resolution plus the computed overview."""

    index: int
    name: str
    row: dict                    # values of images.json for this image
    table: pd.DataFrame          # pores.json
    scale: Scale
    original: np.ndarray | None  # RGB uint8; None if the source image is not readable any more
    labels: np.ndarray           # int32, id of the counted pore per pixel
    rejected_labels: np.ndarray  # int32, id of the rejected pore per pixel
    rejected: pd.DataFrame       # rejected.json: pore_id, filter, reason
    added: frozenset[int]        # counted by hand (restored or drawn)
    removed: frozenset[int]      # rejected by hand
    roi: np.ndarray              # bool, True = sample
    overview: np.ndarray         # RGB uint8, long edge OVERVIEW_MAX_PX
    overview_roi: np.ndarray     # bool, overview size
    message: str                 # problem while loading, "" if none
    frame: Frame | None = None   # coordinate system of the sample, None = none in the run
    plots: dict = field(default_factory=dict)   # plot name -> PNG bytes (plots/<name>.png)
    overview_masks: dict = field(default_factory=dict, repr=False)   # pores, rejected, added, removed
    _boxes: dict = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape

    def pore_labels(self, rejected: bool) -> np.ndarray:
        return self.rejected_labels if rejected else self.labels

    def pore_box(self, pore_id: int, rejected: bool = False) -> tuple[int, int, int, int]:
        """Bounding box (top, bottom, left, right; bottom/right exclusive) of one pore, on demand:
        for all pores at once it would cost about 0.5 s per 100 MP image."""
        key = (pore_id, rejected)
        if key not in self._boxes:
            ys, xs = np.nonzero(self.pore_labels(rejected) == pore_id)
            self._boxes[key] = ((int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
                                if len(ys) else (0, 1, 0, 1))
        return self._boxes[key]

    def nearest_pore(self, x: int, y: int, radius: float, counted: bool = True,
                     rejected: bool = True) -> tuple[int, bool, int, int] | None:
        """The pore under (x, y), else the nearest one within `radius` image px:
        (pore id, is rejected, x, y of its pixel closest to the point); None = none. Only a small
        window around the point is searched."""
        r = max(0, int(np.ceil(radius)))
        height, width = self.shape
        top, bottom, left, right = max(0, y - r), min(height, y + r + 1), max(0, x - r), min(width, x + r + 1)
        best = None
        for is_rejected, wanted in ((False, counted), (True, rejected)):
            if not wanted:
                continue
            window = self.pore_labels(is_rejected)[top:bottom, left:right]
            ys, xs = np.nonzero(window)
            if not len(ys):
                continue
            distance = (ys + top - y) ** 2 + (xs + left - x) ** 2
            i = int(np.argmin(distance))
            if distance[i] <= radius ** 2 and (best is None or distance[i] < best[0]):
                best = (int(distance[i]), int(window[ys[i], xs[i]]), is_rejected, int(xs[i] + left), int(ys[i] + top))
        return None if best is None else best[1:]

    def pore_box_near(self, pore_id: int, rejected: bool, x: int, y: int) -> tuple[int, int, int, int]:
        """Bounding box of a pore that has a pixel at (x, y): searched in a window around the point
        that grows until the pore fits – fast for hovering, unlike pore_box over the whole image."""
        key = (pore_id, rejected)
        if key in self._boxes:
            return self._boxes[key]
        labels = self.pore_labels(rejected)
        height, width = labels.shape
        half = 32
        while True:
            top, bottom = max(0, y - half), min(height, y + half + 1)
            left, right = max(0, x - half), min(width, x + half + 1)
            ys, xs = np.nonzero(labels[top:bottom, left:right] == pore_id)
            whole = (top == 0 and bottom == height and left == 0 and right == width)
            touches = ((ys.min() == 0 and top > 0) or (ys.max() == bottom - top - 1 and bottom < height)
                       or (xs.min() == 0 and left > 0) or (xs.max() == right - left - 1 and right < width))
            if not touches or whole:
                break
            half *= 2
        self._boxes[key] = (int(ys.min()) + top, int(ys.max()) + top + 1, int(xs.min()) + left, int(xs.max()) + left + 1)
        return self._boxes[key]

    def status_masks(self, top: int, bottom: int, left: int, right: int) -> dict[str, np.ndarray]:
        """Full-resolution masks of a part: pores, rejected, added (by hand), removed (by hand)."""
        labels, rejected = self.labels[top:bottom, left:right], self.rejected_labels[top:bottom, left:right]
        return {"pores": labels > 0, "rejected": rejected > 0,
                "added": _member(labels, self.added), "removed": _member(rejected, self.removed)}

    def reason(self, pore_id: int) -> tuple[str, str] | None:
        """(filter, reason) of a rejected pore."""
        row = self.rejected[self.rejected["pore_id"] == pore_id]
        return (str(row.iloc[0]["filter"]), str(row.iloc[0]["reason"])) if len(row) else None

    def original_crop(self, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        if self.original is not None:
            return self.original[top:bottom, left:right]
        return np.full((bottom - top, right - left, 3), MISSING_GRAY, dtype=np.uint8)


def load_image(run: Run, index: int) -> Image:
    """Slow (about 1.5 s for 100 MP): run in a background thread. The files are read in parallel."""
    row = run.row(index)
    folder = run.results_dir / row["image"]

    # the image to draw on (output.draw_on); runs older than that only have the original
    shown = row["draw_path"] if isinstance(row.get("draw_path"), str) else row["path"]

    def original():
        try:
            return read_rgb(Path(shown)), ""
        except (OSError, ValueError):
            return None, f"original not found: {shown}"

    with ThreadPoolExecutor(4) as pool:
        jobs = (pool.submit(_decode, folder / LABELS_TIF, cv2.IMREAD_UNCHANGED),
                pool.submit(_decode, folder / REJECTED_LABELS_TIF, cv2.IMREAD_UNCHANGED),
                pool.submit(read_bool, folder / ROI_MASK_TIF), pool.submit(original))
        labels, rejected_labels, roi, (rgb, message) = (job.result() for job in jobs)
    plots = {path.stem: path.read_bytes() for path in (folder / PLOTS_DIR).glob("*.png")}
    corrections = folder / CORRECTIONS_JSON
    saved = json.loads(corrections.read_text(encoding="utf-8")) if corrections.exists() else {}

    height, width = labels.shape
    factor = max(1.0, max(height, width) / OVERVIEW_MAX_PX)
    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    overview = (cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if rgb is not None
                else np.full((size[1], size[0], 3), MISSING_GRAY, dtype=np.uint8))
    image = Image(index, row["image"], row, read_table(folder / PORES_JSON), Scale.from_row(row), rgb, labels,
                  rejected_labels, read_table(folder / REJECTED_JSON), frozenset(saved.get("added", [])),
                  frozenset(saved.get("removed", [])), roi, overview, shrink_mask(roi, size), message,
                  Frame.from_row(row), plots)
    update_overview(image)
    return image


def apply_state(image: Image, state) -> None:
    """Take over the state after a manual correction (a State of the correction session)."""
    image.labels, image.rejected_labels, image.rejected = state.labels, state.rejected_labels, state.rejected
    image.table, image.row = state.per_pore, state.row
    image.added, image.removed = state.added, state.removed
    if getattr(state, "plots", None) is not None:     # drawn again after the change
        image.plots = state.plots
    image._boxes.clear()
    update_overview(image)


def update_overview(image: Image) -> None:
    """Overview masks of pores, rejected pores and the pores changed by hand."""
    size = (image.overview.shape[1], image.overview.shape[0])
    height, width = image.shape
    image.overview_masks = {name: shrink_mask(mask, size)
                            for name, mask in image.status_masks(0, height, 0, width).items()}


def _member(labels: np.ndarray, ids) -> np.ndarray:
    """Pixels whose id is in `ids` (bool mask)."""
    if not ids:
        return np.zeros(labels.shape, dtype=bool)
    table = np.zeros(max(int(labels.max()), max(ids)) + 1, dtype=bool)
    table[list(ids)] = True
    return table[labels]


def shrink_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bool mask to (width, height); a pixel with any covered pixel counts (small pores stay visible)."""
    return cv2.resize(mask.view(np.uint8) * np.uint8(255), size, interpolation=cv2.INTER_AREA) > 0

