"""Comparison picture: the original next to the same picture with the counted pores in red.

Saved as comparison.jpg in the image folder, so the result can be checked without the viewer (a
picture for a rough look by a human, hence JPG and shrunk to MAX_PX).
Cropped to the sample (as every evaluation module sees the image); rewritten after every manual
correction. Pores are blended in red over the original, as the full overlay of the writer.
"""

import numpy as np

from ..core import imaging
from ..core.registry import register
from ..core.step import Evaluator
from ..core.types import EvaluationResult

RED = (255, 0, 0)               # RGB, pores
ALPHA = 0.5                     # share of red on pore pixels
MAX_PX = 4000                   # long edge of the saved picture in px (panoramas would be huge)
GAP_SHARE = 0.01                # white gap between both pictures, share of the picture width


@register("comparison")
class Comparison(Evaluator):
    """Original | original with the counted pores in red, side by side -> comparison.jpg."""

    params = {}

    def evaluate(self, ctx, pores):
        original = ctx.original
        color = np.array(RED, dtype=np.float32) * (np.iinfo(original.dtype).max / 255)
        mask = pores.labels > 0
        marked = original.copy()
        marked[mask] = (original[mask] * (1 - ALPHA) + color * ALPHA).astype(original.dtype)

        height, width = mask.shape
        gap = max(1, round(GAP_SHARE * width))
        picture = np.full((height, 2 * width + gap, 3), np.iinfo(original.dtype).max, dtype=original.dtype)
        picture[:, :width] = original
        picture[:, width + gap:] = marked
        # shrunk only after marking: small pores stay visible as reddish pixels
        if max(picture.shape[:2]) > MAX_PX:
            picture = imaging.downscale(picture, max(picture.shape[:2]) / MAX_PX)
        if picture.dtype != np.uint8:               # JPG: 8 bit only
            picture = (picture >> 8).astype(np.uint8)
        return EvaluationResult(per_image={}, per_pore=None, images={"comparison.jpg": picture})
