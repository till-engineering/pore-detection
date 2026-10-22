"""Filter: color – a pore has no color of its own, a colored inclusion or phase has.

A pore is a hole: it shows the (dark) color of the light, like the matrix around it, only
darker. A colored inclusion, an oxide or an etched phase has a hue of its own. Compared is the
chromaticity (color without brightness: R, G, B divided by their sum) of
    the pore    mean RGB of its pixels
    around it   mean RGB of a ring of matrix around it (from gap to gap + ring, without other pores)
Their distance (0 = same hue, sqrt(2/3) ~ 0.82 = most different) above max_color_difference ->
rejected. Very dark pores have little light and a noisy hue: pores whose mean brightness is below
min_brightness (share of the ring's brightness) are not judged and stay.

Not tileable. Exactly one of ring_px / ring_um and of gap_px / gap_um.
"""

import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import LabelStep


@register("color_filter")
class ColorFilter(LabelStep):
    """Rejects pores whose hue differs from the matrix around them."""

    params = {
        "max_color_difference": "largest chromaticity distance pore - surroundings (0..0.82, e.g. 0.05)",
        "min_brightness": "pores darker than this share of the ring brightness are not judged (e.g. 0.15)",
        "gap_px": "ring starts this far outside the pore in px (or gap_um)",
        "gap_um": "ring start in µm (or gap_px; needs a scale)",
        "ring_px": "ring width in px (or ring_um)",
        "ring_um": "ring width in µm (or ring_px; needs a scale)",
    }

    def process(self, ctx, pores):
        if pores.count == 0:
            return pores
        own, ring = adaptive.pore_rings(pores, adaptive.length_px(self, "gap", ctx),
                                        adaptive.length_px(self, "ring", ctx))
        rgb = ctx.original.astype(np.float32)
        pore_rgb = np.stack([adaptive.group_mean(own, rgb[..., c], pores.count)[1:] for c in range(3)], axis=1)
        ring_rgb = np.stack([adaptive.group_mean(ring, rgb[..., c], pores.count)[1:] for c in range(3)], axis=1)
        del rgb
        with np.errstate(invalid="ignore", divide="ignore"):
            pore_sum, ring_sum = pore_rgb.sum(axis=1), ring_rgb.sum(axis=1)
            difference = np.linalg.norm(pore_rgb / pore_sum[:, None] - ring_rgb / ring_sum[:, None], axis=1)
            judged = pore_sum >= self.min_brightness * ring_sum
        hit = judged & (difference > self.max_color_difference)    # NaN (no ring) -> False
        reasons = [f"color difference {d:.3f} above {self.max_color_difference:g}" for d in difference[hit]]
        return pores.reject(pores.ids[hit], self.name, reasons)
