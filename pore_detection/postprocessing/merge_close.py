"""Merge pore parts that are at most max_gap_px apart into one pore.

A threshold that sits right at the bright rim of a pore or at a brightness gradient on its floor
tears a one pixel wide gap through the mask: one pore becomes two objects that do not touch.
Merging only happens across gaps inside the sample (roi_mask) – between two parts with embedding
medium or the scale overlay in between there are two cavities. With bridge_gaps the gap itself is
added to the pore, otherwise one pore would have two outlines. (As in Pore_Detection_V2.)

Runs before split_touching: otherwise merging would undo every split at once.
"""

import cv2
import numpy as np

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("merge_close")
class MergeClose(LabelStep):
    """Pores at most max_gap_px apart (inside the sample) become one; optionally the gap is filled."""

    params = {
        "max_gap_px": "largest gap between two parts that is bridged, 0 = only touching parts",
        "bridge_gaps": "true = the gap pixels are added to the merged pore",
    }

    def process(self, ctx, pores):
        if pores.count < 2:
            return pores
        mask = pores.labels > 0
        allowed = ctx.roi_mask
        reach = mask
        if self.max_gap_px > 0:
            # square of edge gap + 1: two pixels end up 8-connected iff at most gap px lie between them
            kernel = np.ones((self.max_gap_px + 1, self.max_gap_px + 1), dtype=np.uint8)
            reach = cv2.dilate(mask.view(np.uint8), kernel).view(bool) & (allowed | mask)
        _, groups = cv2.connectedComponents(reach.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        merged = np.where(mask, groups, 0)
        composite = _composite(merged, pores.labels)
        if composite.size == 0:
            return pores                        # nothing merged
        if self.bridge_gaps and self.max_gap_px > 0:
            merged = _bridge(merged, composite, self.max_gap_px, allowed)
        return pores.relabeled(geometry.renumber(merged, start=_first_id(pores)))


def _first_id(pores) -> int:
    """New ids must not collide with rejected pores."""
    return pores.next_id if len(pores.rejected) else 1


def _composite(merged: np.ndarray, parts: np.ndarray) -> np.ndarray:
    """Ids in `merged` that cover more than one id of `parts`."""
    flat, part_ids = geometry.pixels(parts)
    # every part lies in exactly one merged pore: count the parts per merged pore
    group_of_part = np.zeros(int(part_ids.max()) + 1, dtype=np.int64)
    group_of_part[part_ids] = merged.ravel()[flat]
    counts = np.bincount(group_of_part[np.flatnonzero(np.bincount(part_ids))])
    return np.flatnonzero(counts > 1)


def _bridge(merged: np.ndarray, composite: np.ndarray, gap: int, allowed: np.ndarray) -> np.ndarray:
    """Add the gap between the merged parts to the pore: closing of the pore, only free sample pixels."""
    out = merged.copy()
    kernel = np.ones((gap + 1, gap + 1), dtype=np.uint8)
    pad = gap + 1
    height, width = out.shape
    for pore_id, (top, bottom, left, right) in geometry.boxes(merged, composite).items():
        ys = slice(max(0, top - pad), min(height, bottom + pad))
        xs = slice(max(0, left - pad), min(width, right + pad))
        closed = cv2.morphologyEx((merged[ys, xs] == pore_id).view(np.uint8), cv2.MORPH_CLOSE, kernel).view(bool)
        window = out[ys, xs]                    # view: changes go to out
        window[closed & (window == 0) & allowed[ys, xs]] = pore_id
    return out
