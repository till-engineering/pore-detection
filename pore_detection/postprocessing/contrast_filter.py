"""Filter: contrast to the surroundings – a pore is a hole, clearly darker than the matrix around it.

Dark phases, shadows at grain boundaries or slightly etched areas can reach the pore threshold
but are only a little darker than the matrix next to them. A pore reflects (almost) nothing.
Per counted pore:
    dark        its 10 % darkest value (robust against the light rim of a blurred pore)
    around      median of a ring of matrix around it: from gap to gap + ring outside the pore,
                without other pores (the soft rim right at the pore is skipped by the gap)
    contrast    1 - dark / around   (0 = as bright as the matrix, 1 = black)
Pores with contrast below min_contrast are rejected. A pore without matrix around it (ring
completely covered by other pores) stays.

Gray value from the original with the standard weights (0.299, 0.587, 0.114) – the ratio hardly
depends on the weights. Not tileable. Exactly one of ring_px / ring_um and of gap_px / gap_um.
"""

import numpy as np

from ..core import adaptive
from ..core.imaging import to_gray
from ..core.registry import register
from ..core.step import LabelStep

GRAY_WEIGHTS = (0.299, 0.587, 0.114)
DARK_QUANTILE = 0.1


@register("contrast_filter")
class ContrastFilter(LabelStep):
    """Rejects pores that are not clearly darker than the matrix around them."""

    params = {
        "min_contrast": "smallest contrast 1 - dark / around (0..1, e.g. 0.4)",
        "gap_px": "ring starts this far outside the pore in px, skips the soft rim (or gap_um)",
        "gap_um": "ring start in µm (or gap_px; needs a scale)",
        "ring_px": "ring width in px (or ring_um)",
        "ring_um": "ring width in µm (or ring_px; needs a scale)",
    }

    def process(self, ctx, pores):
        if pores.count == 0:
            return pores
        own, ring = adaptive.pore_rings(pores, adaptive.length_px(self, "gap", ctx),
                                        adaptive.length_px(self, "ring", ctx))
        gray = to_gray(ctx.original, GRAY_WEIGHTS)
        dark = adaptive.group_quantile(own, gray, pores.count, DARK_QUANTILE)[1:]
        around = adaptive.group_quantile(ring, gray, pores.count, 0.5)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            contrast = 1 - dark / around
        hit = contrast < self.min_contrast          # NaN (no ring) -> False, the pore stays
        reasons = [f"contrast {c:.2f} below {self.min_contrast:g} (dark {d:.3f}, around {a:.3f})"
                   for c, d, a in zip(contrast[hit], dark[hit], around[hit])]
        return pores.reject(pores.ids[hit], self.name, reasons)
