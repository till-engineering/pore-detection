"""Filter: solidity (area / convex hull area) – against networks of grain boundaries.

A network of dark grain boundaries that the detection joins into one object encloses a lot of
empty area and has a very low solidity. A pore – even a ragged shrinkage cavity – is solid.
Objects below geometry.MIN_PIXELS_FOR_SHAPE are skipped (their shape is pixel grid only).
(As in Pore_Detection_V2.)
"""

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("compactness")
class Compactness(LabelStep):
    """Rejects pores with a solidity below min_solidity."""

    params = {"min_solidity": "smallest solidity (area / convex hull area, 1 = convex)"}

    def process(self, ctx, pores):
        ids = pores.ids[geometry.areas(pores.labels, pores.ids, pores.pixels) >= geometry.MIN_PIXELS_FOR_SHAPE]
        if ids.size == 0:
            return pores
        shape = geometry.shapes(pores.labels, ids, pores.pixels)
        hit = shape["solidity"] < self.min_solidity
        reasons = [f"solidity {s:.2f} below {self.min_solidity:g}" for s in shape["solidity"][hit]]
        return pores.reject(shape["pore_id"][hit].to_numpy(), self.name, reasons)
