"""Top-hat correction: subtract the local matrix brightness instead of dividing by it.

background_correction divides by the background: right if illumination scales all gray values of
an area by the same factor. If an area is mainly shifted (stray light, offset of the camera, dark
images with little contrast), dividing over- or under-corrects the pores there. Here the
background is subtracted (black top-hat: how much darker than the matrix around it) and the result
is put back on the typical matrix level of the whole image:

    result = 1 - (background - image) / matrix level     matrix ~1, pore keeps its contrast

So a pore of the same depth in gray levels gets the same value everywhere; thresholds of the
detection fit as after background_correction. Background as in background_correction (closing on a
shrunk copy, resin set to the matrix value). Not tileable (large window), CPU only.

Exactly one of window_px / window_um is set in config.yaml (µm needs a scale).
"""

import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Filter


@register("tophat_correction")
class TophatCorrection(Filter):
    """Subtracts the local matrix brightness (black top-hat), matrix ~1; result clipped to 0..1."""

    params = {
        "window_px": "closing window in px, larger than the largest pore (or window_um)",
        "window_um": "closing window in µm (or window_px; needs a scale)",
        "work_scale": "background is computed on a copy shrunk by this factor (e.g. 4)",
    }

    def process(self, ctx, image):
        window = adaptive.length_px(self, "window", ctx)
        background = adaptive.matrix_background(image, ctx.roi_mask, window, self.work_scale)
        level = max(adaptive.matrix_level(background, ctx.roi_mask), 1e-3)
        result = (image - background) / np.float32(level) + np.float32(1.0)
        return np.clip(result, 0.0, 1.0, out=result)
