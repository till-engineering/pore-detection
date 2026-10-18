"""INSTRUCTION – templates for the three kinds of postprocessing steps (interface: core/step.py).

Not active: every @register below is commented out, so nothing of this file is used.
To make your own step: copy this file (e.g. postprocessing/my_step.py), keep the class you need,
uncomment @register("...") and add the name to config.yaml under

  postprocessing:
    mask_steps:  [...]   MaskStep  – mask in, mask out          (any number, in this order)
    labeling:    {...}   Labeler   – mask in, numbered pores out (exactly one)
    label_steps: [...]   LabelStep – pores in, pores out         (any number, in this order)

Every key besides "name" becomes an attribute (self.edge_um). The base class (MaskStep, Labeler,
LabelStep) decides the stage. Inputs are never changed in place – always return new data.

ctx: ImageContext, read only – all its values are listed in _context_values at the end.
All arrays have the size of the cropped area (bounding box of the sample), not of the full image.
"""

from dataclasses import replace

import cv2
import numpy as np
import pandas as pd

from ..core import geometry
from ..core.registry import register  # noqa: F401 – needed once @register is active
from ..core.step import LabelStep, Labeler, MaskStep
from ..core.types import LabeledPores


# --- 1. MaskStep: works on the pore mask (before the pores get numbers) ---

# @register("example_mask_step")
class ExampleMaskStep(MaskStep):
    """Example: deletes mask pixels closer than edge_um to the edge of the sample."""

    params = {"edge_um": "width of the strip along the sample edge that is cleared, in µm"}
    tileable = False            # True = local operation, may run on tiles (then set tile_overlap_px)

    def process(self, ctx, mask):
        # mask: (H, W) bool, True = pore pixel (what the detector found, maybe changed by earlier mask steps)
        values = _context_values(ctx)
        if values["pixel_size_um"] is None:
            raise ValueError("edge_um needs a scale")          # -> image skipped, logged
        radius = max(1, round(self.edge_um / values["pixel_size_um"]))
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        inner = cv2.erode(ctx.roi_mask.view(np.uint8), kernel, borderValue=0).view(bool)
        return mask & inner                                    # new array, `mask` stays unchanged


# --- 2. Labeler: gives every pore a number (exactly one in config.yaml) ---

# @register("example_labeler")
class ExampleLabeler(Labeler):
    """Example: 8-connected mask pixels form one pore (like connected_components)."""

    params = {}

    def label(self, ctx, mask):
        # mask: (H, W) bool; core has already removed everything outside the sample (mask & roi_mask)
        n, labels = cv2.connectedComponents(mask.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        count = n - 1
        return LabeledPores(
            labels=labels,                      # (H, W) int32, 0 = background, else pore_id
            count=count,                        # number of counted pores
            properties=pd.DataFrame({"pore_id": np.arange(1, count + 1, dtype=np.int32)}),  # only pore_id
        )                                       # rejected_labels / rejected: none yet (defaults)


# --- 3a. LabelStep that only ADDS values per pore ---

# @register("example_values")
class ExampleValues(LabelStep):
    """Example: adds the centroid of every pore (full image px) and its relative height zeta."""

    params = {}
    changes_pores = False       # only adds columns -> runs again after a manual correction in the viewer

    def process(self, ctx, pores):
        values = _context_values(ctx)
        _pore_values(pores)                     # everything in LabeledPores – see below
        flat, ids = pores.pixels                # all counted pore pixels: flat index + pore id (cached)
        rows, cols = np.divmod(flat, pores.labels.shape[1])
        index = np.searchsorted(pores.ids, ids)                 # pixel -> row in properties
        count = np.bincount(index, minlength=pores.count)
        top, left = values["offset_px"]                         # crop -> full image coordinates
        y = np.bincount(index, weights=rows, minlength=pores.count) / np.maximum(count, 1) + top
        x = np.bincount(index, weights=cols, minlength=pores.count) / np.maximum(count, 1) + left
        columns = {"centroid_y_px": y, "centroid_x_px": x}
        if ctx.frame is not None:
            _, z = ctx.frame.to_sample(y, x)
            columns["centroid_zeta"] = np.clip(z / ctx.frame.thickness_px, 0, 1)
        # internal values for later steps / evaluation modules (pores.get("centroid_y_px")).
        # They reach pores.json only if an evaluation module returns them in per_pore.
        return replace(pores, properties=pores.properties.assign(**columns))


# --- 3b. LabelStep that CHANGES the pores (filter: reject with a reason) ---

# @register("example_filter")
class ExampleFilter(LabelStep):
    """Example: rejects pores that touch the edge of the sample (they are cut off)."""

    params = {}
    changes_pores = True        # changes which pores count -> must come BEFORE value steps in config.yaml

    def process(self, ctx, pores):
        if pores.count == 0:
            return pores
        edge = ctx.roi_mask & ~cv2.erode(ctx.roi_mask.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
        touching = np.unique(pores.labels[edge])
        touching = touching[touching > 0]
        # reject: the pores keep their id, appear in the "rejected" layer of the viewer with this reason
        # and can be counted again by hand; rejected_labels / rejected are filled by reject()
        return pores.reject(touching, self.name, "touches the sample edge")
        # pores with new shapes (merge, split): pores.relabeled(new_labels) – new ids from pores.next_id on
        # delete without a trace (noise): set the pixels to 0 in a copy of labels, then pores.relabeled(...)


def _pore_values(pores) -> dict:
    """All values of LabeledPores – remove what you do not need."""
    return {
        "labels": pores.labels,                     # (H, W) int32, 0 = no counted pore, else pore_id
        "count": pores.count,                       # number of counted pores
        "properties": pores.properties,             # DataFrame, one row per counted pore: pore_id + added columns
        "ids": pores.ids,                           # pore_id of the counted pores, ascending (gaps possible)
        "next_id": pores.next_id,                   # first free id (for new pores)
        "pixels": pores.pixels,                     # (flat index, pore_id) of all counted pore pixels, cached
        "rejected_labels": pores.rejected_labels,   # (H, W) int32 of rejected pores, None = none
        "rejected": pores.rejected,                 # DataFrame pore_id, filter, reason
        # one column: pores.get("area_px2") -> ndarray (clear error if the column is missing)
        # helpers: geometry.areas(labels, ids, pixels), geometry.boxes(...), geometry.shapes(...)
        "areas_px2": geometry.areas(pores.labels, pores.ids, pores.pixels),
    }


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
        "offset_px": info.offset_px,        # (y, x) of this crop / tile inside the full image
        "original": ctx.original,           # (H, W, 3) RGB uint8/uint16, same area as the mask, read only
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
        # single points: frame.to_sample(y_full, x_full) -> (x_px, z_px)
        # ctx.require_frame() raises (image skipped) if there is no frame
    return values
