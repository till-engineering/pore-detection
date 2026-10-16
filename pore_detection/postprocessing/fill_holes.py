"""Fill holes in pores – only holes that are small compared to the pore around them.

A pore is brighter towards its wall and often has a brightness gradient on its floor; both tear
holes into the mask. A pore with a hole is one pore, not two with a ring. Without a limit,
however, a ring of dark grain boundaries around a whole grain would turn the grain into a
compact "pore" that no shape filter removes any more: the ring is thin, the grain large. A hole
in a real pore is small compared to the pore. (As in Pore_Detection_V2.)

Not tileable: whether an area is a hole depends on the whole pore around it.
"""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import MaskStep


@register("fill_holes")
class FillHoles(MaskStep):
    """Fills holes up to max_ratio x the area of the surrounding pore (0 = every hole)."""

    params = {"max_ratio": "largest filled hole as multiple of the area of its pore, 0 = fill every hole"}

    def process(self, ctx, mask):
        # background reachable from the border (4-connected, via a 1 px frame) is outside
        outside = cv2.copyMakeBorder((~mask).view(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
        cv2.floodFill(outside, None, (0, 0), 2, flags=4)
        filled = outside[1:-1, 1:-1] != 2
        holes = filled & ~mask
        if not holes.any() or self.max_ratio <= 0:
            return filled

        n_holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(holes.view(np.uint8), connectivity=4)
        _, owners = cv2.connectedComponents(filled.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        pore_area = np.bincount(owners[mask], minlength=int(owners.max()) + 1)
        owner_of_hole = np.zeros(n_holes, dtype=np.int64)
        owner_of_hole[hole_labels[holes]] = owners[holes]       # every hole lies inside one pore
        fill = hole_stats[:, cv2.CC_STAT_AREA] <= self.max_ratio * pore_area[owner_of_hole]
        fill[0] = False
        return mask | fill[hole_labels]
