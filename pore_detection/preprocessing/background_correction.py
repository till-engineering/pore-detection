"""Background correction: divide the gray image by the local brightness of the microstructure.

Illumination, vignetting, stitching seams and exposure shift all gray values of an image area by
the same factor. Dividing by the local matrix brightness (the "background") removes that: the
matrix becomes ~1, a pore keeps its relative darkness (~0.2) everywhere, so one threshold fits
the whole image.

Background: a grey closing with a window larger than the largest pore fills pores and the dark
network with the surrounding grain brightness; a blur smooths it. It is computed on a copy
shrunk by work_scale (the background varies slowly). Embedding resin is set to the typical
matrix value first, so the dark resin does not pull the background down at the sample edge.

Not tileable: the window is large, the shrunk copy of the whole image is cheaper than tiles with
a large overlap. CPU only (cheap filter, see docs/good_to_know.md).
"""

import cv2
import numpy as np

from ..core.imaging import downscale
from ..core.registry import register
from ..core.step import Filter


@register("background_correction")
class BackgroundCorrection(Filter):
    """Divides the image by the local matrix brightness; result clipped to 0..1."""

    params = {
        "window_px": "closing window in px (full resolution), larger than the largest pore",
        "work_scale": "background is computed on a copy shrunk by this factor (e.g. 4)",
    }

    def process(self, ctx, image):
        small = downscale(image, self.work_scale)
        sample = downscale(ctx.roi_mask, self.work_scale)
        if not sample.any():
            return image.copy()
        small[~sample] = np.median(small[sample])

        size = max(3, round(self.window_px / self.work_scale) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT_101)
        small = cv2.GaussianBlur(small, (0, 0), size / 2, borderType=cv2.BORDER_REFLECT_101)

        height, width = image.shape
        background = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        np.maximum(background, np.float32(1e-3), out=background)
        return np.clip(image / background, 0.0, 1.0, out=background)
