"""Local threshold: every pixel gets a threshold from mean and spread of the window around it.

    niblack   T = mean + k * std                  (k < 0, e.g. -2: clearly darker than around)
    sauvola   T = mean * (1 + k * (std / r - 1))  (k > 0, e.g. 0.3; r = dynamic range of std, 0.5
              for gray 0..1). In flat matrix std is small -> T well below the mean, so noise is no
              pore (Niblack's weakness); near a pore std is large -> T close to the mean.

Adapts to illumination and contrast by itself, no background_correction needed. The window
(window_px / window_um) should be clearly larger than the largest pore – inside a large pore
the window sees only pore and the centre gets lost.
Not tileable (window in µm). Exactly one of window_px / window_um.
"""

import cv2
import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Detector


@register("local_threshold")
class LocalThreshold(Detector):
    """Sauvola or Niblack threshold from mean and std in a window around every pixel."""

    params = {
        "method": "sauvola | niblack",
        "k": "sauvola: 0.1-0.5 (higher = stricter); niblack: negative, e.g. -2",
        "r": "sauvola: dynamic range of the std (0.5 for gray 0..1); unused by niblack",
        "window_px": "window edge length in px, larger than the largest pore (or window_um)",
        "window_um": "window edge length in µm (or window_px; needs a scale)",
    }
    choices = {"method": ["sauvola", "niblack"]}

    def detect(self, ctx, image):
        size = max(3, round(adaptive.length_px(self, "window", ctx)) | 1)
        border = cv2.BORDER_REFLECT_101
        mean = cv2.boxFilter(image, cv2.CV_32F, (size, size), borderType=border)
        mean_sq = cv2.sqrBoxFilter(image, cv2.CV_32F, (size, size), borderType=border)
        std = cv2.sqrt(np.maximum(mean_sq - mean * mean, np.float32(0), out=mean_sq))
        if self.method == "sauvola":
            threshold = mean * (np.float32(1) + np.float32(self.k) * (std / np.float32(self.r) - np.float32(1)))
        else:
            threshold = mean + np.float32(self.k) * std
        return (image < threshold) & ctx.roi_mask
