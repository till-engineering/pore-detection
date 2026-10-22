"""Automatic global threshold per image, from the histogram of the sample (comparison baseline).

    otsu      splits the histogram into two classes with the smallest spread each. Assumes two
              similar-sized peaks – with few pores (1 %) it tends to cut into the matrix
    triangle  line from the matrix peak to the end of the dark tail; threshold where the histogram
              lies farthest below it. Made for one large peak + a small tail (= few pores)
    kapur     maximum entropy of the two classes; also works for a small dark class

offset is added to the found threshold (gray units 0..1; + = more pore). Only sample pixels count.
Measured threshold logged (DEBUG). Not tileable (one threshold per image).
"""

import logging

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import Detector

log = logging.getLogger("pore_detection")


@register("auto_threshold")
class AutoThreshold(Detector):
    """One threshold per image from the histogram of the sample: otsu | triangle | kapur."""

    params = {
        "method": "otsu | triangle | kapur",
        "offset": "added to the found threshold (gray 0..1, + = more pore, e.g. 0)",
    }
    choices = {"method": ["otsu", "triangle", "kapur"]}

    def detect(self, ctx, image):
        values = np.clip(image[ctx.roi_mask] * 255 + 0.5, 0, 255).astype(np.uint8)
        if not values.size:
            return np.zeros(image.shape, bool)
        if self.method == "kapur":
            level = _kapur(np.bincount(values, minlength=256))
        else:
            flag = cv2.THRESH_OTSU if self.method == "otsu" else cv2.THRESH_TRIANGLE
            level, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + flag)
        threshold = (level + 0.5) / 255 + self.offset
        log.debug("%s: %s threshold %.3f", ctx.info.name, self.method, threshold)
        return (image < threshold) & ctx.roi_mask


def _kapur(hist: np.ndarray) -> int:
    """Gray level (0..255) with the largest sum of the entropies of both classes."""
    p = hist / hist.sum()
    cumulative = np.cumsum(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
        cum_plogp = np.cumsum(plogp)
        low = np.log(cumulative) - cum_plogp / cumulative
        high = np.log(1 - cumulative) - (cum_plogp[-1] - cum_plogp) / (1 - cumulative)
        total = np.where((cumulative > 0) & (cumulative < 1), low + high, -np.inf)
    return int(np.argmax(total))
