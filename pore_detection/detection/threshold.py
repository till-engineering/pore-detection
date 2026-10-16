"""Fixed gray threshold: everything darker than `threshold` is a pore."""

from ..core.registry import register
from ..core.step import Detector


@register("threshold")
class Threshold(Detector):
    """Pixels darker than a fixed gray value are pores."""

    params = {"threshold": "gray value 0..1; pixels below it are pores"}
    tileable = True
    tile_overlap_px = 0

    def detect(self, ctx, image):
        return image < self.threshold
