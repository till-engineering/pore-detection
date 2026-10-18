"""No sample frame: for runs that do not need one (e.g. round sections, only porosity)."""

from ..core.registry import register
from ..core.step import FrameDetector


@register("none")
class NoFrame(FrameDetector):
    """No coordinate system; steps that need one make the image fail."""

    params = {}

    def detect_frame(self, info, roi_mask):
        return None
