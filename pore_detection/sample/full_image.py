"""ROI = the whole image (no embedding medium detection)."""

import numpy as np

from ..core.registry import register
from ..core.step import RoiDetector


@register("full_image")
class FullImage(RoiDetector):
    """The whole image is sample."""

    params = {}

    def detect_roi(self, info, original):
        return np.ones(original.shape[:2], dtype=bool)
