"""Split pores that grew together into single pores (distance transform + watershed).

Two touching pores counted as one distort the count, the size distribution and the largest pore –
often the actual acceptance criterion. In a nest of two merged pores the distance to the border
has two local maxima; flooding from them puts the dividing line into the constriction.

Safeguard: shrinkage cavities are ragged and have many small side maxima. A part smaller than
min_area_ratio of the original pore is such a fringe and is merged back into the largest part –
otherwise one branched pore would be cut into a dozen pieces. A pore is never lost: pixels no
flood reaches stay with the pore. (As in Pore_Detection_V2, with OpenCV instead of scikit-image.)

Ids: the largest part keeps the id of the pore, further parts get new ids. Runs after merge_close.
"""

import cv2
import numpy as np

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("split_touching")
class SplitTouching(LabelStep):
    """Splits pores with several distance maxima at least min_distance_px apart."""

    params = {
        "min_distance_px": "smallest distance between two pore centres in px",
        "min_area_ratio": "parts smaller than this share of the pore are merged back (0..1)",
    }

    def __init__(self, **params):
        super().__init__(**params)
        size = 2 * self.min_distance_px + 1
        self.peak_kernel = np.ones((size, size), dtype=np.uint8)   # a peak is the maximum in this square

    def process(self, ctx, pores):
        if pores.count == 0:
            return pores
        labels = pores.labels.copy()
        next_id = pores.next_id
        for pore_id, (top, bottom, left, right) in geometry.boxes(pores.labels, None, pores.pixels).items():
            if max(bottom - top, right - left) <= 2 * self.min_distance_px:
                continue                        # too small for two centres min_distance_px apart
            window = labels[top:bottom, left:right]     # view: changes go to labels
            parts = self._split(window == pore_id)
            if parts is None:
                continue
            for part in parts[1:]:
                window[part] = next_id
                next_id += 1
        return pores.relabeled(labels) if next_id > pores.next_id else pores

    def _centres(self, distance: np.ndarray) -> np.ndarray:
        """Local maxima of the distance, at least min_distance_px apart (chessboard distance);
        among close ones the highest wins (like skimage.feature.peak_local_max)."""
        candidates = (distance > 0) & (distance == cv2.dilate(distance, self.peak_kernel))
        ys, xs = np.nonzero(candidates)
        order = np.argsort(-distance[ys, xs], kind="stable")
        blocked = np.zeros(distance.shape, dtype=bool)  # closer than min_distance_px to a kept centre
        reach = self.min_distance_px - 1
        kept = []
        for y, x in zip(ys[order].tolist(), xs[order].tolist()):
            if not blocked[y, x]:
                kept.append((y, x))
                blocked[max(0, y - reach):y + reach + 1, max(0, x - reach):x + reach + 1] = True
        return np.array(kept, dtype=np.int64).reshape(-1, 2)

    def _split(self, pore: np.ndarray) -> list[np.ndarray] | None:
        """Parts of one pore (bool masks, largest first), None if it stays one pore."""
        padded = np.pad(pore, 1).view(np.uint8)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        centres = self._centres(distance)
        if len(centres) < 2:
            return None
        n_markers = len(centres) + 1            # marker ids 1 … len(centres), n_markers = outside
        markers = np.zeros(padded.shape, dtype=np.int32)
        markers[centres[:, 0], centres[:, 1]] = np.arange(1, n_markers, dtype=np.int32)

        # cv2.watershed floods an 8-bit colour image; lowest = deepest inside the pore
        relief = 255 - cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        markers[padded == 0] = n_markers        # outside the pore: its own basin
        cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers)
        # watershed lines (-1) and unreached pixels go to the neighbouring part, never lost
        inside = padded.astype(bool)
        lines = inside & ((markers <= 0) | (markers == n_markers))
        if lines.any():
            grown = cv2.dilate(np.where(inside & ~lines, markers, 0).astype(np.float32), np.ones((3, 3), np.uint8))
            markers[lines] = grown[lines].astype(np.int32)
        markers = markers[1:-1, 1:-1]

        total = int(pore.sum())
        ids, counts = np.unique(markers[pore], return_counts=True)
        keep = (ids > 0) & (ids < n_markers)
        ids, counts = ids[keep], counts[keep]
        largest = ids[np.argmax(counts)]
        for part_id, count in zip(ids, counts):
            if count < self.min_area_ratio * total:
                markers[pore & (markers == part_id)] = largest
        markers[pore & ((markers <= 0) | (markers >= n_markers))] = largest
        ids, counts = np.unique(markers[pore], return_counts=True)
        if len(ids) < 2:
            return None
        order = np.argsort(-counts)
        return [pore & (markers == ids[i]) for i in order]
