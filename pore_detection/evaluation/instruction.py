"""INSTRUCTION – template for an evaluation module (stage "evaluation", interface: core/step.py Evaluator).

Not active: @register below is commented out, so nothing of this file is used.
To make your own module: copy this file (e.g. evaluation/my_values.py), rename the class,
uncomment @register("...") and add it to config.yaml:

  evaluation:
    - name: my_values
      min_area_um2: 50          # every key besides "name" becomes an attribute (self.min_area_um2)

What a module gets and returns (once per image, modules of one image run in parallel):
  evaluate(ctx, pores) -> EvaluationResult(per_image, per_pore, images)
    ctx:   ImageContext, read only (all its values: _context_values below)
    pores: LabeledPores, only the COUNTED pores (all values: _pore_values below)
  The module never writes files – core writes everything it returns:
    per_image -> images.json + results Excel (one row per image)
    per_pore  -> pores.json (one row per pore, joined on pore_id)
    images    -> <image folder>/<name>.tif (key "<name>.jpg" -> <name>.jpg, 8 bit: pictures for humans)
  Column names are free, but must not clash with another module (the run stops with an error).
  It runs again after every manual correction in the viewer.
"""

import numpy as np

from ..core import geometry
from ..core.registry import register  # noqa: F401 – needed once @register is active
from ..core.step import Evaluator
from ..core.types import EvaluationResult


# @register("example_values")
class ExampleValues(Evaluator):
    """Example: number of large pores, pore density and a map of the large pores."""

    params = {"min_area_um2": "pores from this area on count as large (µm², needs a scale)"}
    choices = {}                        # e.g. {"mode": ["a", "b"]} -> selection in the launcher
    excel_sheet = "Example"             # sheet in <ID>_results.xlsx (None = module name)
    excel_hidden = {"large_area_px2"}   # per-image values written to JSON only, not to the Excel

    def evaluate(self, ctx, pores):
        values = _context_values(ctx)
        pore = _pore_values(pores)

        # values of other steps: pores.get(column) – e.g. area_px2 from the label step pore_properties
        area_px2 = pores.get("area_px2")
        scale = values["pixel_size_um"]
        if scale is None:
            raise ValueError("min_area_um2 needs a scale")     # -> image skipped, logged with this text
        area_um2 = area_px2 * scale ** 2
        large = area_um2 >= self.min_area_um2
        sample_mm2 = np.count_nonzero(ctx.roi_mask) * scale ** 2 / 1e6

        # images: e.g. a map of the large pores, (H, W) of the cropped area; uint8 / uint16 / float32
        large_ids = pore["ids"][large]
        large_map = (np.isin(pores.labels, large_ids) * 255).astype(np.uint8)

        return EvaluationResult(
            per_image={                                         # numbers or texts, one per column
                "large_pore_count": int(large.sum()),
                "large_area_px2": int(area_px2[large].sum()),
                "pore_density_per_mm2": pores.count / sample_mm2 if sample_mm2 > 0 else float("nan"),
            },
            per_pore=pores.properties[["pore_id"]].assign(is_large=large),   # must contain pore_id
            images={"large_pores": large_map},                  # -> large_pores.tif ({} = none)
        )

    def finalize(self, per_image, per_pore):
        """Optional: values over the whole ID, once at the end of a COMPLETE run (and after corrections).
        per_image: DataFrame, one row per image (all per_image values of all modules + core columns)
        per_pore:  iterator of DataFrames, pores.json of one image after the other (read lazily)
        returns {name: value} -> id_values.json + sheet "ID" (delete the method if not needed)."""
        large = sum(int(table["is_large"].sum()) for table in per_pore if "is_large" in table)
        return {"large_pore_count_id": large, "large_pore_count_mean": float(per_image["large_pore_count"].mean())}


def _pore_values(pores) -> dict:
    """All values of LabeledPores (only counted pores) – remove what you do not need."""
    return {
        "labels": pores.labels,                     # (H, W) int32, 0 = no counted pore, else pore_id
        "count": pores.count,                       # number of counted pores
        "properties": pores.properties,             # DataFrame: pore_id + values of the label steps (e.g. area_px2)
        "ids": pores.ids,                           # pore_id, ascending (gaps possible)
        "pixels": pores.pixels,                     # (flat index, pore_id) of all pore pixels, cached
        "rejected": pores.rejected,                 # DataFrame pore_id, filter, reason (not counted)
        "rejected_labels": pores.rejected_labels,   # (H, W) int32 of rejected pores, None = none
        # one column: pores.get("area_px2"); helpers: geometry.areas / boxes / shapes / equivalent_diameter
        "equivalent_diameter_px": geometry.equivalent_diameter(geometry.areas(pores.labels, pores.ids, pores.pixels)),
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
        "offset_px": info.offset_px,        # (y, x) of the cropped area inside the full image
        "original": ctx.original,           # (H, W, 3) RGB uint8/uint16 of the cropped area, read only
        "roi_mask": ctx.roi_mask,           # (H, W) bool, True = sample (count_nonzero = sample area in px)
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
        # single points: frame.to_sample(y_full, x_full) -> (x_px, z_px); full = array index + offset_px
        # ctx.require_frame() raises FrameMissing if there is no frame: only this module is skipped – see evaluation/pore_index.py
    return values
