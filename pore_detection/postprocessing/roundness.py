"""Filter: roundness 4 A / (pi feret_max²) – against elongated and ragged objects.

Robust against pixelation (no perimeter involved), unlike the usual circularity 4 pi A / U²,
which wrongly rejects small round pores. Objects below geometry.MIN_PIXELS_FOR_SHAPE are skipped.
Choose the limit with care: gas pores lie above 0.7, shrinkage cavities reach down to 0.2.
(As in Pore_Detection_V2.)
"""

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("roundness")
class Roundness(LabelStep):
    """Rejects pores with a roundness below min_roundness."""

    params = {"min_roundness": "smallest roundness 4 A / (pi feret_max²), 1 = circle"}

    def process(self, ctx, pores):
        ids = pores.ids[geometry.areas(pores.labels, pores.ids, pores.pixels) >= geometry.MIN_PIXELS_FOR_SHAPE]
        if ids.size == 0:
            return pores
        shape = geometry.shapes(pores.labels, ids, pores.pixels)
        hit = shape["roundness"] < self.min_roundness
        reasons = [f"roundness {r:.2f} below {self.min_roundness:g}" for r in shape["roundness"][hit]]
        return pores.reject(shape["pore_id"][hit].to_numpy(), self.name, reasons)
