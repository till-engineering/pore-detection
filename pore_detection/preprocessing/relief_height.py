"""Relief height: the depth below the surroundings from a shaded-relief image (several light
directions merged into one picture).

In such an image the gray value tells the slope of the surface along the light, not its height:
a pore is a dark shadow (wall falling away from the light), then a bright highlight (wall rising
towards it). Adding up the slope along the light gives the height, and a pore becomes a
depression - also where a line crosses it near its side, whose walls only show a weak, but long
shadow. This filter computes that depth, after it any detection for dark pores works (e.g.
hysteresis).

1. Light direction (core/adaptive.py, light_angle): base light_angle_deg, checked on every image
   on a copy shrunk by estimate_work_scale; the estimate is used if its certainty reaches
   min_certainty and it lies more than angle_tolerance_deg off the base. Logged.
2. Slope: image / its local mean - 1 (gaussian of 2 x integration_length on a shrunk copy), so
   stripes of a stitched panorama and the exposure do not count.
3. Height z with the least squares error: slope of z along the light = slope of the image,
   slope of z across the light small (weight cross_smoothing), z small (weight 1 /
   integration_length²: an error of the slope does not add up over more than about this length).
   The image holds no slope across the light; cross_smoothing only fills it in smoothly. Solved by
   FFT in blocks with a margin of 3 decay lengths (the solution is local), depth = -z.
4. Reference: depth minus its grey opening with reference_radius (larger than the largest pore)
   on a shrunk copy, at least 0: depth below the surroundings; slow waves of the surface and
   bumps -> 0.
5. Gray value 1 / (1 + depth): matrix 1, the deeper the darker (depth 1 -> 0.5, 3 -> 0.25), never
   clipped. Depth unit: relative gray contrast x px along the light.

Outside the sample the slope is 0 (flat), so the dark resin gives no relief at the sample edge;
there the result is 1. Not tileable (one light direction per image, blocks inside), CPU only.
Exactly one of integration_length_px / _um and of reference_radius_px / _um (µm needs a scale).
"""

import math

import cv2
import numpy as np

from ..core import adaptive
from ..core.imaging import downscale
from ..core.registry import register
from ..core.step import Filter

BLOCK_PX = 2048             # core edge of an FFT block (plus margin, rounded up to a fast FFT size)


@register("relief_height")
class ReliefHeight(Filter):
    """Depth below the surroundings from the slope along the light; gray 1 / (1 + depth), matrix 1."""

    params = {
        "light_angle_deg": "base direction the light shines in (shadow -> highlight of a pore) in "
                           "degrees, 0 = to the right, 90 = up (as the image is shown)",
        "angle_tolerance_deg": "estimate within this many degrees of the base -> the base is used",
        "min_certainty": "estimate certainty 0..1 below it -> the base is used (e.g. 0.05; pure noise 0)",
        "estimate_work_scale": "the light direction is estimated on a copy shrunk by this factor",
        "integration_length_px": "slope errors add up over about this length in px, about the largest "
                                 "pore (or integration_length_um)",
        "integration_length_um": "as integration_length_px in µm (needs a scale)",
        "cross_smoothing": "weight of a smooth height across the light (e.g. 1; 0 = every line on its own)",
        "reference_radius_px": "depth is measured below the surroundings within this radius in px, larger "
                               "than the largest pore (or reference_radius_um)",
        "reference_radius_um": "as reference_radius_px in µm (needs a scale)",
    }

    def process(self, ctx, image):
        length = adaptive.length_px(self, "integration_length", ctx)
        radius = adaptive.length_px(self, "reference_radius", ctx)
        roi = ctx.roi_mask
        level = np.float32(adaptive.matrix_level(image, roi))
        image = np.where(roi, image, level)
        angle = adaptive.light_angle(self, ctx, image, level, length)
        slope = self._slope(image, length)
        slope[~roi] = 0
        depth = self._depth(slope, angle, length)
        adaptive.depth_below_surroundings(depth, radius)
        depth[~roi] = 0
        depth += 1
        return np.divide(np.float32(1), depth, out=depth)

    @staticmethod
    def _slope(image, length):
        """image / local mean - 1; the mean (gaussian, sigma 2 x length) on a shrunk copy."""
        factor = max(1.0, length / 8)
        small = cv2.GaussianBlur(downscale(image, factor), (0, 0), 2 * length / factor,
                                 borderType=cv2.BORDER_REFLECT_101)
        height, width = image.shape
        mean = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        np.maximum(mean, np.float32(1e-3), out=mean)
        slope = np.divide(image, mean, out=mean)
        slope -= 1
        return slope

    def _depth(self, slope, angle_deg, length):
        """-z of min |slope of z along the light - slope|² + cross_smoothing |slope across|² + |z|² / length²,
        per FFT block: D = i k_along / (k_along² + cross_smoothing k_across² + 1 / length²) x S."""
        height, width = slope.shape
        margin = math.ceil(3 * length * max(1.0, math.sqrt(self.cross_smoothing)))
        size_y = cv2.getOptimalDFTSize(min(BLOCK_PX, height) + 2 * margin)
        size_x = cv2.getOptimalDFTSize(min(BLOCK_PX, width) + 2 * margin)
        core_y, core_x = size_y - 2 * margin, size_x - 2 * margin
        rows, cols = math.ceil(height / core_y), math.ceil(width / core_x)
        padded = cv2.copyMakeBorder(slope, margin, margin + rows * core_y - height,
                                    margin, margin + cols * core_x - width, cv2.BORDER_CONSTANT, value=0)
        dx, dy = math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))   # y points down
        ky = (2 * np.pi * np.fft.fftfreq(size_y))[:, None]
        kx = (2 * np.pi * np.fft.fftfreq(size_x))[None, :]
        along, across = kx * dx + ky * dy, ky * dx - kx * dy
        gain = (along / (along ** 2 + self.cross_smoothing * across ** 2 + 1 / length ** 2)).astype(np.float32)
        depth = np.empty_like(slope)
        for row in range(rows):
            for col in range(cols):
                y, x = row * core_y, col * core_x
                spectrum = cv2.dft(padded[y:y + size_y, x:x + size_x], flags=cv2.DFT_COMPLEX_OUTPUT)
                real, imag = spectrum[..., 0].copy(), spectrum[..., 1]
                spectrum[..., 0] = -imag * gain               # (real + i imag) x i gain
                spectrum[..., 1] = real * gain
                block = cv2.idft(spectrum, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
                bottom, right = min(core_y, height - y), min(core_x, width - x)
                depth[y:y + bottom, x:x + right] = block[margin:margin + bottom, margin:margin + right]
        return depth
