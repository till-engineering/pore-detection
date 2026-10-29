"""Height map: the depth below the surroundings from a height image exported by the microscope.

Input is a gray image (8 or 16 bit, e.g. TIFF, or the result of false_color_height) whose gray value
is the height of the surface; deep says which end of the gray scale is deep. height_um_full_scale
is the height between gray 0 and the largest gray value (255 / 65535) in µm, so depths are in µm;
if the export gives no scale, 1 measures depths in shares of the gray range.

1. Depth: (1 - gray) or gray (deep = dark / bright) x height_um_full_scale.
2. Reference, within reference_radius (larger than the largest pore); tilt of the sample and slow
   waves of the surface -> 0:
   opening  depth minus its grey opening (core/adaptive.py, depth_below_surroundings): below the
            highest points around; for clean height data, bumps -> 0. On noisy data the highest
            points are noise peaks, the whole matrix then lies "deep".
   median   depth minus its local median (core/adaptive.py, local_median): below the typical
            height around, robust against noise; for noisy data (e.g. depth composition).
3. Unit (depth_unit):
   um       as it is (µm, or shares of the gray range).
   noise    divided by the noise of the matrix (core/adaptive.py, spread: robust sigma of the
            depth inside the sample): depth in sigma. For height images whose scale is stretched
            per image (false colors from min to max): stretch and height_um_full_scale scale depth
            and noise alike and cancel out, so one threshold fits every image.
4. At least 0; gray value 1 / (1 + depth): matrix 1, the deeper the darker (1 -> 0.5, 3 -> 0.25,
   5 -> 0.167), never clipped; as after relief_height, so the same detection fits (e.g. hysteresis).

Outside the sample the result is 1. Not tileable (large reference window), CPU only. Exactly one of
reference_radius_px / _um (µm needs a scale). height_um_full_scale is a height, not a length in the
image: it does not end in _um, which would demand a scale of the image.
"""

import logging

import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Filter

log = logging.getLogger("pore_detection")


@register("height_map")
class HeightMap(Filter):
    """Depth below the surroundings from a height image; gray 1 / (1 + depth), matrix 1."""

    params = {
        "deep": "dark | bright: which end of the gray scale is deep in the exported height image",
        "height_um_full_scale": "height in µm between gray 0 and the largest gray value (1 = unknown: "
                                "depth in shares of the gray range; cancels out with depth_unit noise)",
        "reference": "opening | median: depth below the highest points (clean data) or below the typical "
                     "height (noisy data) of the surroundings",
        "reference_radius_px": "depth is measured below the surroundings within this radius in px, larger "
                               "than the largest pore (or reference_radius_um)",
        "reference_radius_um": "as reference_radius_px in µm (needs a scale)",
        "depth_unit": "um | noise: depth as it is, or in multiples of the matrix noise (sigma)",
    }
    choices = {"deep": ["dark", "bright"], "reference": ["opening", "median"], "depth_unit": ["um", "noise"]}

    def __init__(self, **params):
        super().__init__(**params)
        for key, allowed in self.choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"height_map: {key} must be one of {', '.join(allowed)}, not {getattr(self, key)!r}")

    def process(self, ctx, image):
        depth = 1 - image if self.deep == "dark" else image.copy()
        depth *= np.float32(self.height_um_full_scale)
        radius = adaptive.length_px(self, "reference_radius", ctx)
        if self.reference == "opening":
            adaptive.depth_below_surroundings(depth, radius)
        else:
            depth -= adaptive.local_median(depth, ctx.roi_mask, radius)
        if self.depth_unit == "noise":
            sigma = adaptive.spread(depth, ctx.roi_mask)
            if sigma <= 0:
                raise ValueError("height_map: the height image has no noise inside the sample (flat or "
                                 "quantised to one level), depth in sigma is undefined")
            log.info("%s: height noise sigma %.4g (height_map unit)", ctx.info.name, sigma)
            depth /= np.float32(sigma)
        np.maximum(depth, 0, out=depth)
        depth[~ctx.roi_mask] = 0
        depth += 1
        return np.divide(np.float32(1), depth, out=depth)
