"""Half contrast: every pore gets its own edge threshold, half way between its core and the matrix.

A blurred pore fades from its dark core into the matrix; its true edge lies where the gray value is
half way between core and matrix (edge of a blurred step). hysteresis uses one grow threshold for
all pores: a shallow pore (small, or less dark) is cut too small, a deep pore too large – and how
much depends on blur and contrast of the image. Here:
    background   local matrix brightness (closing with window_px / window_um, as background_correction)
    cores        pixels at least seed_depth darker than the background (relative: 0.6 = 60 % darker)
    micro pores  optional spots (as hysteresis) are pore with their spot area, but do not grow: their
                 core is only a little darker than the matrix, half way to it would reach far into
                 the microstructure (measured: F1 0.63 instead of 0.85 on the synthetic images)
    per core     its darkest value (image smoothed with sigma 1 px against single noisy pixels)
    edge         every pixel belongs to the nearest core; it is pore if it is darker than
                 core + edge_level * (background - core)   (0.5 = half contrast)
                 and connected to that core
Only seed_depth decides what is a pore; the size follows from the image itself, with no edge
parameter that depends on blur or contrast.

Neighbouring pores sharing one dark area each keep their own threshold up to the middle between
their cores (nearest core). Works with or without background_correction before it (relative depth).
Not tileable. Exactly one of window_px / window_um and of spot_sigma_px / spot_sigma_um.
"""

import cv2
import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Detector


@register("half_contrast")
class HalfContrast(Detector):
    """Cores by relative depth; each pore's edge at edge_level between its core and the local matrix."""

    params = {
        "seed_depth": "cores: at least this much darker than the local matrix, relative (0..1, e.g. 0.6)",
        "edge_level": "edge between core (0) and matrix (1); 0.5 = half contrast",
        "window_px": "window for the local matrix brightness in px, larger than the largest pore (or window_um)",
        "window_um": "window for the local matrix brightness in µm (or window_px; needs a scale)",
        "work_scale": "matrix brightness computed on a copy shrunk by this factor (e.g. 4)",
        "spot_sigma_px": "micro pores: smoothing in px for the spot measure (or spot_sigma_um)",
        "spot_sigma_um": "micro pores: smoothing in µm (or spot_sigma_px; needs a scale)",
        "spot_threshold": "spot measure (smaller Hessian eigenvalue x sigma²) above it = pore core; 0 = off",
    }

    def detect(self, ctx, image):
        roi = ctx.roi_mask
        background = adaptive.matrix_background(image, roi, adaptive.length_px(self, "window", ctx), self.work_scale)
        seeds = (image < background * np.float32(1 - self.seed_depth)) & roi
        spots = None
        if self.spot_threshold > 0:
            spots = (adaptive.spot_measure(image, adaptive.length_px(self, "spot_sigma", ctx))
                     > self.spot_threshold) & roi
        if not seeds.any():
            return seeds if spots is None else spots

        # darkest value of every core
        n, core_labels = cv2.connectedComponents(seeds.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        smooth = cv2.GaussianBlur(image, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
        core = adaptive.group_quantile(core_labels, smooth, n - 1, 0.0).astype(np.float32)
        del smooth

        # nearest core of every pixel: labels of the zero pixels (= cores) of the distance transform,
        # mapped to the core labels (several zero components may belong to one core)
        _, nearest = cv2.distanceTransformWithLabels((~seeds).view(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_CCOMP)
        to_core = np.zeros(int(nearest.max()) + 1, np.int32)
        to_core[nearest[seeds]] = core_labels[seeds]
        nearest_core = core[to_core[nearest]]
        del nearest
        threshold = nearest_core + np.float32(self.edge_level) * (background - nearest_core)
        pores = adaptive.grow(seeds, (image < threshold) & roi)
        return pores if spots is None else pores | spots
