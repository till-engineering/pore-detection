"""Filter: grinding scratches – long, thin and straight.

All three conditions must hold, each for a reason:
- elongated (min_aspect_ratio): the obvious feature, but alone too coarse – shrinkage cavities
  are elongated too
- absolutely thin (max_width_px or max_width_um): the real distinction – a scratch of the abrasive
  grain is a few micrometres wide however long it is; an elongated cavity is much wider
- straight (min_solidity): a scratch lies close to its convex hull, a branched cavity does not.
  This protects cavities: what is ragged is not rejected as a scratch even if long and narrow.
(As in Pore_Detection_V2.)

Exactly one of max_width_px / max_width_um is set in config.yaml (µm needs a scale).
"""

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("scratch")
class Scratch(LabelStep):
    """Rejects long, thin, straight objects (aspect ratio, width, solidity)."""

    params = {
        "min_aspect_ratio": "elongated from this ratio of major to minor axis",
        "max_width_px": "thin up to this minor axis in px (or max_width_um)",
        "max_width_um": "thin up to this minor axis in µm (or max_width_px; needs a scale)",
        "min_solidity": "straight from this solidity (area / convex hull area) – ragged objects stay",
    }

    def process(self, ctx, pores):
        if pores.count == 0:
            return pores
        shape = geometry.shapes(pores.labels, pores.ids, pores.pixels)
        if hasattr(self, "max_width_um"):
            width, limit, unit = shape["minor_axis_px"] * ctx.pixel_size_um, self.max_width_um, "µm"
        else:
            width, limit, unit = shape["minor_axis_px"], self.max_width_px, "px"
        hit = ((shape["aspect_ratio"] >= self.min_aspect_ratio) & (width <= limit)
               & (shape["solidity"] >= self.min_solidity))
        reasons = [f"scratch: width {w:.2f} {unit}, aspect ratio {a:.1f}, solidity {s:.2f}"
                   for w, a, s in zip(width[hit], shape["aspect_ratio"][hit], shape["solidity"][hit])]
        return pores.reject(shape["pore_id"][hit].to_numpy(), self.name, reasons)
