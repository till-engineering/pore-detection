"""Data types passed between modules. Binding specification: docs/interfaces.md, section 2."""

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a read-only view; the caller's array keeps its flags."""
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class ImageInfo:
    """Metadata of the source image."""

    path: Path
    image_id: str               # name of the input folder (sample ID)
    name: str                   # <image_id>_<file stem>, used in all outputs
    height_px: int              # full image height
    width_px: int               # full image width
    bit_depth: int              # 8 or 16
    offset_px: tuple[int, int]  # (y, x) of the current data inside the full image


@dataclass(frozen=True)
class ScaleResult:
    """Output of the sample.scale stage."""

    pixel_size_um: float | None             # micrometres per pixel, None = unknown
    scale_mask: np.ndarray | None = None    # (H, W) bool, True = scale overlay (no sample), None = none
    bar_px: tuple[int, int, int, int] | None = None  # measured bar (top, bottom, left, right), for the check
    label: str | None = None                # text read from the scale label, for the check


@dataclass(frozen=True)
class ImageContext:
    """Read-only context, passed to every step from preprocessing on."""

    info: ImageInfo
    original: np.ndarray        # (H, W, 3) RGB, uint8 / uint16, never modified
    roi_mask: np.ndarray        # (H, W) bool, True = sample
    pixel_size_um: float | None # micrometres per pixel, None = unknown

    def __post_init__(self):
        object.__setattr__(self, "original", _read_only(self.original))
        object.__setattr__(self, "roi_mask", _read_only(self.roi_mask))


REJECTED_COLUMNS = ("pore_id", "filter", "reason")


def _no_rejected() -> pd.DataFrame:
    return pd.DataFrame({"pore_id": pd.Series(dtype=np.int32), "filter": pd.Series(dtype=str),
                         "reason": pd.Series(dtype=str)})


@dataclass(frozen=True)
class LabeledPores:
    """Output of postprocessing, input of evaluation.

    Counted and rejected pores share one id space; ids are unique and ascending but may have
    gaps (a rejected pore keeps its id, so it is the same pore when it is counted again).
    Evaluation modules only see the counted pores (labels, properties).
    """

    labels: np.ndarray          # (H, W) int32, 0 = no counted pore, else pore_id
    count: int                  # number of counted pores
    properties: pd.DataFrame    # one row per counted pore: pore_id (ascending) + columns added by label steps
    rejected_labels: np.ndarray | None = None   # (H, W) int32, pore_id of rejected pores; None = none
    rejected: pd.DataFrame = field(default_factory=_no_rejected)  # pore_id, filter, reason

    def get(self, column: str) -> np.ndarray:
        """One value per counted pore, ordered by pore_id."""
        if column not in self.properties.columns:
            raise KeyError(
                f"pore property '{column}' not found (available: {list(self.properties.columns)}); "
                f"add the label step that computes it to postprocessing.label_steps"
            )
        return self.properties[column].to_numpy()

    @cached_property
    def pixels(self) -> "geometry.Pixels":
        """Flat indices and ids of all counted pore pixels (computed once, see core/geometry.py)."""
        return geometry.pixels(self.labels)

    @property
    def ids(self) -> np.ndarray:
        """pore_id of the counted pores, ascending."""
        return self.properties["pore_id"].to_numpy()

    @property
    def next_id(self) -> int:
        """First id not used by a counted or rejected pore."""
        used = [int(self.properties["pore_id"].max()) if self.count else 0,
                int(self.rejected["pore_id"].max()) if len(self.rejected) else 0]
        return max(used) + 1

    def reject(self, ids, filter_name: str, reasons) -> "LabeledPores":
        """Move counted pores to the rejected ones (they keep their id). `reasons`: one text per id
        or one text for all. Properties of the moved pores are dropped."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return self
        reasons = [reasons] * ids.size if isinstance(reasons, str) else list(reasons)
        flat, values = self.pixels
        moved = _lookup(ids, self.labels)[values]
        labels = self.labels.copy()
        labels.ravel()[flat[moved]] = 0
        rejected_labels = (np.zeros_like(self.labels) if self.rejected_labels is None
                           else self.rejected_labels.copy())
        rejected_labels.ravel()[flat[moved]] = values[moved]
        rows = pd.DataFrame({"pore_id": ids.astype(np.int32), "filter": filter_name, "reason": reasons})
        properties = self.properties[~self.properties["pore_id"].isin(ids)].reset_index(drop=True)
        return replace(self, labels=labels, count=len(properties),
                       properties=properties, rejected_labels=rejected_labels,
                       rejected=pd.concat([self.rejected, rows]).sort_values("pore_id").reset_index(drop=True))

    def relabeled(self, labels: np.ndarray) -> "LabeledPores":
        """New counted pores (e.g. after merging or splitting); the rejected ones stay.

        Properties start again with pore_id only – value steps come after steps that change pores.
        """
        ids = geometry.ids_of(labels).astype(np.int32)
        if len(self.rejected) and np.isin(ids, self.rejected["pore_id"].to_numpy()).any():
            raise ValueError("new pore ids collide with rejected pores – use ids from next_id on")
        return replace(self, labels=labels.astype(np.int32, copy=False), count=len(ids),
                       properties=pd.DataFrame({"pore_id": ids}))


def _lookup(ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Bool table indexed by pore id: True for `ids`."""
    table = np.zeros(max(int(labels.max()), int(ids.max())) + 1, dtype=bool)   # labels.max: cheap
    table[ids] = True
    return table


@dataclass(frozen=True)
class EvaluationResult:
    """Output of one evaluation module for one image."""

    per_image: dict[str, float | int | str]  # one value per column of the per-image results
    per_pore: pd.DataFrame | None            # one row per pore, must contain column pore_id
    images: dict[str, np.ndarray]            # images to save, key = output name
