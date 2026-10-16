"""Gaussian blur of the gray image (noise reduction before detection).

CPU only: a GPU path was measured slower (copying to the GPU and back costs more than
OpenCV's blur), see docs/good_to_know.md.
"""

import cv2

from ..core.registry import register
from ..core.step import Filter


@register("gauss")
class Gauss(Filter):
    """Gaussian blur with standard deviation sigma_px."""

    params = {"sigma_px": "standard deviation of the gaussian kernel in px"}
    tileable = True
    tile_overlap_px = 0          # depends on sigma_px, set per instance

    def __init__(self, **params):
        super().__init__(**params)
        self.radius_px = max(1, round(4 * self.sigma_px))  # kernel covers +-4 sigma
        self.tile_overlap_px = self.radius_px

    def process(self, ctx, image):
        size = 2 * self.radius_px + 1
        return cv2.GaussianBlur(image, (size, size), self.sigma_px, borderType=cv2.BORDER_REFLECT_101)
