"""Morphological opening of the pore mask: removes single noise pixels and thin spurs."""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import MaskStep


@register("open")
class Opening(MaskStep):
    """Opening with a disc of radius_px (erode, then dilate)."""

    params = {"radius_px": "radius of the disc-shaped structuring element in px"}
    tileable = True

    def __init__(self, **params):
        super().__init__(**params)
        self.tile_overlap_px = 2 * self.radius_px       # erode + dilate each reach radius_px
        size = 2 * self.radius_px + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def process(self, ctx, mask):
        return cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_OPEN, self.kernel).view(bool)
