"""Height map: the depth below the surroundings from a height image exported by the microscope.

Input is a gray image (8 or 16 bit, e.g. TIFF) whose gray value is the height of the surface;
deep says which end of the gray scale is deep. height_um_full_scale is the height between gray 0
and the largest gray value (255 / 65535) in µm, so depths are in µm; if the export gives no
scale, 1 measures depths in shares of the gray range.

1. Depth: (1 - gray) or gray (deep = dark / bright) x height_um_full_scale.
2. Reference (core/adaptive.py, depth_below_surroundings): depth minus its grey opening with
   reference_radius (larger than the largest pore), at least 0: depth below the surroundings; tilt
   of the sample, slow waves of the surface and bumps -> 0.
3. Gray value 1 / (1 + depth): matrix 1, the deeper the darker (1 µm -> 0.5, 3 µm -> 0.25), never
   clipped; as after relief_height, so the same detection fits (e.g. hysteresis).

Outside the sample the result is 1. Not tileable (large opening), CPU only. Exactly one of
reference_radius_px / _um (µm needs a scale). height_um_full_scale is a height, not a length in the
image: it does not end in _um, which would demand a scale of the image.
"""

import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Filter


@register("height_map")
class HeightMap(Filter):
    """Depth below the surroundings from a height image; gray 1 / (1 + depth), matrix 1."""

    params = {
        "deep": "dark | bright: which end of the gray scale is deep in the exported height image",
        "height_um_full_scale": "height in µm between gray 0 and the largest gray value (1 = unknown: "
                                "depth in shares of the gray range)",
        "reference_radius_px": "depth is measured below the surroundings within this radius in px, larger "
                               "than the largest pore (or reference_radius_um)",
        "reference_radius_um": "as reference_radius_px in µm (needs a scale)",
    }
    choices = {"deep": ["dark", "bright"]}

    def __init__(self, **params):
        super().__init__(**params)
        if self.deep not in self.choices["deep"]:
            raise ValueError(f"height_map: deep must be dark or bright, not {self.deep!r}")

    def process(self, ctx, image):
        depth = 1 - image if self.deep == "dark" else image.copy()
        depth *= np.float32(self.height_um_full_scale)
        adaptive.depth_below_surroundings(depth, adaptive.length_px(self, "reference_radius", ctx))
        depth[~ctx.roi_mask] = 0
        depth += 1
        return np.divide(np.float32(1), depth, out=depth)
