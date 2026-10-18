"""Hysteresis threshold: dark pore cores grow into their soft rim; optionally plus small dark spots.

A blurred pore has a dark core and a rim that fades into the matrix. A single threshold either
misses the rim (low) or picks up dark matrix pixels and noise (high). Here the low threshold only
finds the cores; every connected area below the high threshold that contains a core is pore. A
dark network pixel without a core nearby stays matrix.

Spots (micro pores): a pore of 1-3 px is brightened so much by the optical blur that its core
stays above seed_threshold. It is still a dark *spot*, curved upwards in every direction, while
the equally thin dark network is a *line*, curved in one direction only. The smaller eigenvalue
of the Hessian (gray image smoothed with spot_sigma_px, scale-normalised with sigma²) is large
only for spots. Pixels where it exceeds spot_threshold are pore cores as well and belong to the
pore themselves; they also grow over connected pixels below grow_threshold. spot_threshold 0 =
off. A spot is as large as the area above spot_threshold – the size of a 1-3 px pore is not
measurable more exactly.

Cores and growth stay inside the sample (ctx.roi_mask): the dark embedding resin would otherwise be
one huge "pore" reaching a pixel or two across the sample edge, leaving crumbs there after core
cuts the mask to the sample.

Works on whatever gray image it gets (with or without a preprocessing filter); the thresholds
have to fit that image. Not tileable: a pore must be connected to its core as a whole.
"""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import Detector


@register("hysteresis")
class Hysteresis(Detector):
    """Areas below grow_threshold that contain a pixel below seed_threshold or a dark spot are pores."""

    params = {
        "seed_threshold": "gray value 0..1; pixels below it are certain pore cores",
        "grow_threshold": "gray value 0..1; a core grows over connected pixels below it (>= seed_threshold)",
        "spot_sigma_px": "smoothing in px for the spot measure, about the size of the optical blur",
        "spot_threshold": "spot measure (smaller Hessian eigenvalue x sigma², gray units); above it = "
                          "pore core of a micro pore; 0 = no spot detection",
    }

    def detect(self, ctx, image):
        seeds = image < self.seed_threshold
        candidates = image < self.grow_threshold
        if self.spot_threshold > 0:
            # measure = 2 x smaller eigenvalue (unscaled); compare with the threshold converted to it
            spots = self._spot_measure(image) > np.float32(2 * self.spot_threshold / self.spot_sigma_px ** 2)
            seeds |= spots
            candidates |= spots
        candidates &= ctx.roi_mask
        n, labels = cv2.connectedComponents(candidates.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        keep = np.zeros(n, dtype=bool)
        keep[labels[seeds]] = True
        keep[0] = False
        return keep[labels]

    def _spot_measure(self, image):
        """2 x smaller eigenvalue of the Hessian (not scaled with sigma²): large where the image is a
        dark spot. OpenCV functions only (multi-threaded; NumPy element-wise is single-threaded)."""
        border = cv2.BORDER_REFLECT_101
        smooth = cv2.GaussianBlur(image, (0, 0), self.spot_sigma_px, borderType=border)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3, borderType=border)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3, borderType=border)
        dxy2 = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3, scale=0.5, borderType=border)  # 2 x dxy
        del smooth
        trace = cv2.add(dxx, dyy)
        difference = cv2.magnitude(cv2.subtract(dxx, dyy), dxy2)   # eigenvalue difference
        return cv2.subtract(trace, difference)
