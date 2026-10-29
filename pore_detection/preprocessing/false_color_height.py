"""False color height: the relative height from a height image shown in false colors (e.g. the
depth composition of a Keyence VHX: blue = deep, cyan, green, yellow, red = high).

Reads the color original (ctx.original), not the gray image: a gray conversion of a rainbow scale
is not monotonic (blue and red both turn dark). The hue alone tells the position on the scale; it
does not depend on brightness and hardly on JPEG compression, and the exact color table of the
microscope is not needed.

1. Hue of every pixel (0..360°).
2. Position on the scale: the scale runs from hue_deep_deg to hue_high_deg through decreasing hue
   (240 blue -> 120 green -> 0 red); deep 0, high 1. Hues outside that arc (magenta, purple) go to
   the nearer end of the scale.
3. Pixels without color (saturation below min_saturation or value below min_value: gray, white,
   black, e.g. gaps of the measurement or overlays) carry no height: they get the median height of
   the sample. Their share is logged if it exceeds 1 %.

The result is a gray image 0 (deep) .. 1 (high): height_map with deep: dark follows. The microscope
stretches its scale per image from the lowest to the highest point, so the value is relative only;
height_map with depth_unit: noise makes it comparable between images. Hue is not exactly linear in
the height of a rainbow scale - a fixed, monotonic distortion that does not differ between images.

Per pixel, but not tileable: the median for colorless pixels is one value of the whole sample.
CPU only.
"""

import logging

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import Filter

log = logging.getLogger("pore_detection")


@register("false_color_height")
class FalseColorHeight(Filter):
    """Relative height 0 (deep) .. 1 (high) from the hue of a false color height image."""

    params = {
        "hue_deep_deg": "hue of the deepest color of the scale in degrees (blue: 240)",
        "hue_high_deg": "hue of the highest color in degrees (red: 0); the scale runs from deep to high "
                        "through decreasing hue",
        "min_saturation": "saturation 0..1; less saturated pixels (gray, white) carry no height",
        "min_value": "value (brightness) 0..1; darker pixels carry no height",
    }

    def process(self, ctx, image):
        original = ctx.original
        rgb = original.astype(np.float32) * np.float32(1 / np.iinfo(original.dtype).max)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)         # float input: hue 0..360, s and v 0..1
        del rgb
        hue, saturation, value = cv2.split(hsv)
        del hsv

        arc = (self.hue_deep_deg - self.hue_high_deg) % 360
        if arc == 0:
            raise ValueError("false_color_height: hue_deep_deg and hue_high_deg must differ")
        # degrees from the deep end towards the high end (decreasing hue), 0..360
        along = np.subtract(np.float32(self.hue_deep_deg), hue, out=hue)
        np.mod(along, np.float32(360), out=along)
        # outside the arc: the nearer end (beyond high by along - arc, before deep by 360 - along)
        beyond = along > arc
        along[beyond & (along - arc <= 360 - along)] = arc
        along[beyond & (along - arc > 360 - along)] = 0
        height = np.multiply(along, np.float32(1 / arc), out=along)

        colorless = (saturation < self.min_saturation) | (value < self.min_value)
        if colorless.any():
            valid = ctx.roi_mask & ~colorless
            share = colorless[ctx.roi_mask].mean() if ctx.roi_mask.any() else 0.0
            if share > 0.01:
                log.warning("%s: %.1f %% of the sample has no color (no height) -> median height",
                            ctx.info.name, 100 * share)
            height[colorless] = np.median(height[valid]) if valid.any() else 0.5
        return height
