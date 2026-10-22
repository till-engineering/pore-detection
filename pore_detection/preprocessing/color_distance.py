"""Color-aware gray image: brightness along the matrix color, plus a penalty for other colors.

The fixed gray weights (input.gray_weights) treat a dark colored phase like a pore of the same
brightness. A pore is a hole: it is dark and has no color of its own. Here every pixel's RGB is
split into
    along   its component along the typical matrix color (brightness in matrix units: matrix ~1,
            black pore ~0)
    across  its component perpendicular to it (how much its color differs from the matrix color)
and the gray value is  along + color_weight * |across|  (clipped to 0..1). A pore stays dark, a
dark phase with its own color becomes lighter and no longer reaches the pore threshold.

Matrix color: median RGB of the brighter half of the sample (pores and dark phases left out),
measured once on every SAMPLE_STEP-th pixel. Uses ctx.original, ignores the incoming gray image –
put it first in the chain; put background_correction after it for uneven illumination.
color_weight 0 = only brightness along the matrix color. Not tileable (one matrix color for the
whole image); computed in bands of BAND_ROWS rows to keep the memory small.
"""

import numpy as np

from ..core.registry import register
from ..core.step import Filter

SAMPLE_STEP = 7         # every n-th sample pixel for the matrix color
BAND_ROWS = 512


@register("color_distance")
class ColorDistance(Filter):
    """Brightness along the matrix color + color_weight x color difference to it."""

    params = {"color_weight": "how much a color differing from the matrix color brightens a pixel (0 = none, e.g. 1.0)"}

    def process(self, ctx, image):
        sample = ctx.original[ctx.roi_mask][::SAMPLE_STEP].astype(np.float32)
        if not sample.size:
            return image.copy()
        brightness = sample.sum(axis=1)
        matrix = np.median(sample[brightness >= np.median(brightness)], axis=0)
        norm2 = float(matrix @ matrix)
        if norm2 <= 0:
            return image.copy()
        direction = (matrix / norm2).astype(np.float32)
        result = np.empty(image.shape, np.float32)
        for top in range(0, image.shape[0], BAND_ROWS):
            rgb = ctx.original[top:top + BAND_ROWS].astype(np.float32)
            along = rgb @ direction                         # 1 = matrix color, 0 = black
            across = rgb - along[..., None] * matrix.astype(np.float32)
            across = np.sqrt(np.einsum("...c,...c->...", across, across)) / np.float32(np.sqrt(norm2))
            result[top:top + BAND_ROWS] = along + np.float32(self.color_weight) * across
        return np.clip(result, 0.0, 1.0, out=result)
