"""Filter: lower limit of measurability via the equivalent circle diameter.

In pixels on purpose: the question is from which size anything can be recognised at all – a spot
of three pixels has no shape and no reliable size, what one assigns to it comes from the pixel
grid. That depends on the sampling, not on the scale. A limit in micrometres would be an
acceptance rule ("pores below x µm are not counted"), not part of the measurement.
(As in Pore_Detection_V2.)
"""

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("min_diameter")
class MinDiameter(LabelStep):
    """Rejects pores whose equivalent circle diameter is below min_diameter_px."""

    params = {"min_diameter_px": "smallest equivalent circle diameter in px"}

    def process(self, ctx, pores):
        diameter = geometry.equivalent_diameter(geometry.areas(pores.labels, pores.ids, pores.pixels))
        small = diameter < self.min_diameter_px
        reasons = [f"diameter {d:.1f} px below {self.min_diameter_px:g} px" for d in diameter[small]]
        return pores.reject(pores.ids[small], self.name, reasons)
