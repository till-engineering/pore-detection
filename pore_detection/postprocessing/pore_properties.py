"""Basic values per pore: currently only the area."""

from dataclasses import replace

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("pore_properties")
class PoreProperties(LabelStep):
    """Adds area_px2 (pixel count) per pore."""

    params = {}
    changes_pores = False

    def process(self, ctx, pores):
        area = geometry.areas(pores.labels, pores.ids, pores.pixels)
        return replace(pores, properties=pores.properties.assign(area_px2=area))
