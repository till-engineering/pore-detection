"""CLAHE: contrast limited adaptive histogram equalization of the gray image.

Areas with low contrast (dim illumination, a weak objective, slight defocus) get stretched to a
similar contrast as the rest, so one threshold sees pores everywhere at a similar depth. The image
is split into tiles of tile_px / tile_um; in each the histogram is equalized, but at most by
clip_limit (higher = stronger, noise is amplified too). Tiles are blended bilinearly (no seams).

Computed in 8 bit (256 histogram classes; with 16 bit the clip limit per class is so small that
clip_limit has no effect), result back to 0..1. Changes the gray levels of matrix and pores:
the thresholds of the detection have to be set anew for it.
Not tileable by core (CLAHE keeps its own tile grid over the whole image), CPU only.

Exactly one of tile_px / tile_um is set in config.yaml (µm needs a scale).
"""

import cv2
import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Filter


@register("clahe")
class Clahe(Filter):
    """Contrast limited adaptive histogram equalization (tiles of tile_px / tile_um)."""

    params = {
        "clip_limit": "strength: limit of the histogram per tile (OpenCV, e.g. 2.0; 1 = hardly any change)",
        "tile_px": "edge length of the equalization tiles in px (or tile_um)",
        "tile_um": "edge length of the equalization tiles in µm (or tile_px; needs a scale)",
    }

    def process(self, ctx, image):
        tile = max(8.0, adaptive.length_px(self, "tile", ctx))
        height, width = image.shape
        grid = (max(1, round(width / tile)), max(1, round(height / tile)))
        clahe = cv2.createCLAHE(clipLimit=float(self.clip_limit), tileGridSize=grid)
        result = clahe.apply(np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8))
        return result.astype(np.float32) / np.float32(255)
