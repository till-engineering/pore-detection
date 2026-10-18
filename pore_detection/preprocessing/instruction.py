"""INSTRUCTION – template for a preprocessing filter (stage "preprocessing", interface: core/step.py Filter).

Not an active method: @register below is commented out, so nothing of this file is used.
To make your own filter:
  1. copy this file, e.g. preprocessing/my_filter.py, rename the class
  2. uncomment @register("...") and choose a name
  3. add it to config.yaml:   preprocessing: [{name: my_filter, strength: 1.0, blur_um: 2}]
     every key besides "name" becomes an attribute (self.strength); "params" describes them for the launcher

What a filter gets and returns:
  process(ctx, image) -> image
    image: gray working image, (H, W) float32, 0.0 … 1.0, cropped to the sample's bounding box (or a tile)
    ctx:   ImageContext, read only (see _context_values below for everything in it)
    return a NEW (H, W) float32 array 0 … 1 – never change `image` in place
"""

import cv2
import numpy as np

from ..core.registry import register  # noqa: F401 – needed once @register is active
from ..core.step import Filter


# @register("example_filter")
class ExampleFilter(Filter):
    """Example: contrast stretch inside the sample, then a light blur of blur_um (first docstring line = launcher text)."""

    params = {"strength": "share of the contrast stretch, 0 = off, 1 = full",
              "blur_um": "blur (standard deviation) in µm (needs a scale – without one the image is skipped)"}
    choices = {}                # e.g. {"mode": ["a", "b"]} -> selection in the launcher
    tileable = False            # True = local operation: core may cut the image into tiles (parallel, GPU)
    tile_overlap_px = 0         # with tileable: border each tile needs (e.g. kernel radius)

    def __init__(self, **params):
        super().__init__(**params)          # sets self.strength, self.blur_um from config.yaml
        # check / prepare parameters once here (errors abort the run with a clear message)
        if not 0 <= self.strength <= 1:
            raise ValueError(f"strength must be 0 … 1, got {self.strength}")

    def process(self, ctx, image):
        values = _context_values(ctx)       # everything the interface passes – see below
        sample = ctx.roi_mask               # True = sample, False = embedding medium / background

        # 1) contrast stretch with the gray values of the sample only (embedding medium ignored)
        result = image.copy()
        if sample.any():
            low, high = np.percentile(image[sample], [1, 99])
            if high > low:
                stretched = np.clip((image - low) / (high - low), 0.0, 1.0)
                result = (1 - self.strength) * image + self.strength * stretched

        # 2) a length in µm -> px: needs the scale (pixel_size_um = µm per px, None = unknown)
        if values["pixel_size_um"] is None:
            raise ValueError("blur_um needs a scale, but none was found")     # -> image skipped, logged
        sigma_px = self.blur_um / values["pixel_size_um"]
        if sigma_px >= 0.5:
            size = 2 * round(4 * sigma_px) + 1
            result = cv2.GaussianBlur(result, (size, size), sigma_px, borderType=cv2.BORDER_REFLECT_101)

        return result.astype(np.float32, copy=False)

    # optional GPU path (only for tileable = True; gets a batch of tiles, NumPy in and out):
    # def process_gpu(self, ctxs, items):
    #     from ..core import device
    #     tensors = [device.to_gpu(item) for item in items]
    #     ...
    #     return [device.to_numpy(t) for t in tensors]


def _context_values(ctx) -> dict:
    """All values of the ImageContext – remove what you do not need."""
    info = ctx.info                         # ImageInfo: metadata of the source image
    values = {
        "path": info.path,                  # Path of the source file
        "image_id": info.image_id,          # name of the input folder (= sample ID)
        "name": info.name,                  # <image_id>_<file stem>, as in all outputs
        "full_height_px": info.height_px,   # size of the FULL original image
        "full_width_px": info.width_px,
        "bit_depth": info.bit_depth,        # 8 or 16
        "offset_px": info.offset_px,        # (y, x) of this crop / tile inside the full image (tile: may be < 0)
        "original": ctx.original,           # (H, W, 3) RGB uint8/uint16, same area as `image`, read only
        "roi_mask": ctx.roi_mask,           # (H, W) bool, True = sample
        "pixel_size_um": ctx.pixel_size_um, # µm per px, None = no scale
        "frame": ctx.frame,                 # SampleFrame or None (sample.frame: none)
    }
    if ctx.frame is not None:
        frame = ctx.frame                   # all in px of the FULL image
        values.update({
            "frame_origin_px": frame.origin_px,         # (y, x): inner (bottom) edge meets left edge
            "frame_angle_deg": frame.angle_deg,         # direction of x, counter-clockwise as seen
            "frame_thickness_px": frame.thickness_px,   # inner edge (z = 0) to outer edge (z = thickness)
            "frame_length_px": frame.length_px,
            "frame_wedge_deg": frame.wedge_deg,         # only a check value
            "frame_x_axis": frame.x_axis,               # unit vector (dy, dx) along the outer edge
            "frame_z_axis": frame.z_axis,               # unit vector (dy, dx) towards the outer side
        })
        x_px, z_px = ctx.sample_coords()                # (H, W) float32 each: every pixel in the frame
        values["zeta"] = np.clip(z_px / frame.thickness_px, 0, 1)   # relative height, 0 = inner, 1 = outer
        # single points: frame.to_sample(y_full, x_full) -> (x_px, z_px); y/x = full image coordinates
        # = index in this array + offset_px. ctx.require_frame() raises (image skipped) if there is no frame
    return values
