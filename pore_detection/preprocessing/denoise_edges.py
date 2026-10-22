"""Edge-preserving noise reduction: less noise below the threshold, the pore edge stays sharp.

A gaussian blur (gauss) also blurs the pore edge and makes small pores lighter. Two methods that
smooth only between similar gray values:

    bilateral   a pixel is averaged with neighbours of similar gray value (fast)
    nl_means    non-local means: averaged with pixels whose surroundings look alike (stronger,
                slower – about 10x bilateral)

strength: how different two gray values (0..1) may be and still be averaged (bilateral: sigma of
the gray value, nl_means: filter strength h). Should be about the noise of the image, e.g. 0.03;
much more blurs low-contrast pore edges too.
radius_px / radius_um: neighbourhood (bilateral: spatial sigma; nl_means: search window radius).

Tileable (local). Exactly one of radius_px / radius_um is set in config.yaml (µm needs a scale).
"""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import Filter


@register("denoise_edges")
class DenoiseEdges(Filter):
    """Bilateral filter or non-local means: smooths noise, keeps edges."""

    params = {
        "method": "bilateral | nl_means",
        "strength": "gray difference (0..1) that is still smoothed, about the image noise (e.g. 0.03)",
        "radius_px": "neighbourhood radius in px (or radius_um)",
        "radius_um": "neighbourhood radius in µm (or radius_px; needs a scale)",
    }
    choices = {"method": ["bilateral", "nl_means"]}
    tileable = True
    tile_overlap_px = 16        # covers radius 5 (bilateral 3 sigma, nl_means search window 2 x radius + patch)

    def process(self, ctx, image):
        radius = self.radius_um / ctx.pixel_size_um if hasattr(self, "radius_um") else self.radius_px
        radius = max(1, min(5, round(radius)))     # larger is slow and needs a larger tile overlap
        if self.method == "bilateral":
            return cv2.bilateralFilter(image, 2 * radius + 1, float(self.strength), float(radius))
        # nl_means works on 8 / 16 bit; 16 bit needs the L1 norm, h is in gray levels of it
        image16 = (image * 65535).astype(np.uint16)
        result = cv2.fastNlMeansDenoising(image16, h=[float(self.strength) * 65535], templateWindowSize=5,
                                          searchWindowSize=2 * radius + 1, normType=cv2.NORM_L1)
        return result.astype(np.float32) / np.float32(65535)
