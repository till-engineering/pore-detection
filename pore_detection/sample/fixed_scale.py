"""Scale from config.yaml, until the scale module of the other project is integrated."""

from ..core.registry import register
from ..core.step import ScaleDetector
from ..core.types import ScaleResult


@register("fixed_scale")
class FixedScale(ScaleDetector):
    """Same pixel size for every image, taken from the config."""

    params = {"um_per_px": "micrometres per pixel, null = unknown"}

    def detect_scale(self, info, original):
        return ScaleResult(self.um_per_px)
