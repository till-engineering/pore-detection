"""Hysteresis with micro pores found on several scales given in µm.

hysteresis finds micro pores with one smoothing spot_sigma_px. A pore of 1 px and one of 4 px
need different smoothing, and with another objective the same pore covers another number of px.
Here the spot measure (scale-normalised smaller Hessian eigenvalue, see hysteresis) is computed
for every sigma in spot_sigmas_um (or spot_sigmas_px) and the largest value counts: each spot is
measured at the scale that fits it. Given in µm, the same setting fits every magnification.

Otherwise as hysteresis: pixels below seed_threshold are cores, connected pixels below
grow_threshold belong to them; spots are cores and pore themselves. Cores and growth stay inside
the sample. Not tileable. Exactly one of spot_sigmas_px / spot_sigmas_um.
"""

import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Detector


@register("multiscale_spots")
class MultiscaleSpots(Detector):
    """Hysteresis + micro pores from the spot measure on several scales (µm or px)."""

    params = {
        "seed_threshold": "gray value 0..1; pixels below it are certain pore cores",
        "grow_threshold": "gray value 0..1; a core grows over connected pixels below it",
        "spot_sigmas_px": "list of smoothings in px for the spot measure (or spot_sigmas_um)",
        "spot_sigmas_um": "list of smoothings in µm (or spot_sigmas_px; needs a scale), e.g. [0.8, 1.5, 3]",
        "spot_threshold": "spot measure above it = micro pore core (e.g. 0.2)",
    }

    def detect(self, ctx, image):
        if hasattr(self, "spot_sigmas_um"):
            sigmas = [s / ctx.pixel_size_um for s in self.spot_sigmas_um]
        else:
            sigmas = list(self.spot_sigmas_px)
        measure = np.zeros(image.shape, np.float32)
        for sigma in sigmas:
            np.maximum(measure, adaptive.spot_measure(image, max(sigma, 0.5)), out=measure)
        spots = measure > self.spot_threshold
        del measure
        seeds = (image < self.seed_threshold) | spots
        candidates = (image < self.grow_threshold) | spots
        return adaptive.grow(seeds & ctx.roi_mask, candidates & ctx.roi_mask)
