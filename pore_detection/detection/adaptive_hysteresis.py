"""Adaptive hysteresis: like hysteresis, but the thresholds are measured per image.

hysteresis uses fixed gray values. If images differ in noise or in their matrix level (also after
background_correction, e.g. other exposure or camera gain), a fixed value is too strict on one
and too loose on another. Here every image measures
    level   typical matrix brightness (median of the sample)
    noise   pixel noise (robust spread of image - 3x3 median filter, see core/adaptive.py)
and the thresholds are "k times the noise darker than the matrix":
    seed  = level - seed_k * noise      certain pore cores
    grow  = level - grow_k * noise      a core grows over connected pixels below it (grow_k < seed_k)
So the same seed_k / grow_k mean "clearly darker than this image's noise" on every image.
Measured values are logged (DEBUG).

Right if images differ in contrast and noise together (exposure, gain). Wrong if the noise varies
but the pore contrast does not: on the synthetic images the noise varies 3x at equal contrast and
the fitting k ranges from 20 to 66 (median F1 0.82 against 0.89 of hysteresis).

Micro pores (spot_sigma_px / spot_sigma_um, spot_threshold; 0 = off) as in hysteresis: dark spots
whose scale-normalised Hessian measure exceeds spot_threshold are pore cores too.
Cores and growth stay inside the sample. Not tileable (a pore must be connected to its core).
"""

import logging

from ..core import adaptive
from ..core.registry import register
from ..core.step import Detector

log = logging.getLogger("pore_detection")


@register("adaptive_hysteresis")
class AdaptiveHysteresis(Detector):
    """Hysteresis with thresholds = matrix level - k x noise, measured per image."""

    params = {
        "seed_k": "cores: darker than the matrix by this many noise sigmas (e.g. 30)",
        "grow_k": "growth: darker than the matrix by this many noise sigmas (< seed_k, e.g. 20)",
        "spot_sigma_px": "micro pores: smoothing in px for the spot measure (or spot_sigma_um)",
        "spot_sigma_um": "micro pores: smoothing in µm (or spot_sigma_px; needs a scale)",
        "spot_threshold": "spot measure (smaller Hessian eigenvalue x sigma²) above it = pore core; 0 = off",
    }

    def detect(self, ctx, image):
        level = adaptive.matrix_level(image, ctx.roi_mask)
        noise = max(adaptive.noise_level(image, ctx.roi_mask), 1e-4)
        seed, grow = level - self.seed_k * noise, level - self.grow_k * noise
        log.debug("%s: matrix %.3f, noise %.4f -> seed %.3f, grow %.3f", ctx.info.name, level, noise, seed, grow)
        seeds = image < seed
        candidates = image < grow
        if self.spot_threshold > 0:
            spots = adaptive.spot_measure(image, adaptive.length_px(self, "spot_sigma", ctx)) > self.spot_threshold
            seeds |= spots
            candidates |= spots
        return adaptive.grow(seeds & ctx.roi_mask, candidates & ctx.roi_mask)
