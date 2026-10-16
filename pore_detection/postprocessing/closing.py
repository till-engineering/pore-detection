"""Morphological closing of the pore mask: closes narrow gaps inside pores.

Pores may grow into the embedding medium here; core removes everything outside roi_mask before
labeling.
"""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import MaskStep


@register("close")
class Closing(MaskStep):
    """Closing with a disc of radius_px (dilate, then erode)."""

    params = {"radius_px": "radius of the disc-shaped structuring element in px"}
    tileable = True

    def __init__(self, **params):
        super().__init__(**params)
        self.tile_overlap_px = 2 * self.radius_px       # dilate + erode each reach radius_px
        size = 2 * self.radius_px + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def process(self, ctx, mask):
        return cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_CLOSE, self.kernel).view(bool)
