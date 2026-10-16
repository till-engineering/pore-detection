"""Porosity: pore count and share of pore area in the sample area."""

import numpy as np

from ..core.registry import register
from ..core.step import Evaluator
from ..core.types import EvaluationResult


@register("porosity")
class Porosity(Evaluator):
    """Pore count, porosity per image and pore areas."""

    params = {}

    def evaluate(self, ctx, pores):
        area_px2 = pores.get("area_px2")
        scale = ctx.pixel_size_um
        area_um2 = area_px2 * scale**2 if scale is not None else np.full(pores.count, np.nan)
        pore_area_px2 = int(area_px2.sum())
        sample_area_px2 = int(np.count_nonzero(ctx.roi_mask))
        return EvaluationResult(
            per_image={
                "pore_count": pores.count,
                "pore_area_px2": pore_area_px2,
                "sample_area_px2": sample_area_px2,
                "porosity_percent": 100.0 * pore_area_px2 / sample_area_px2,
            },
            per_pore=pores.properties[["pore_id"]].assign(area_px2=area_px2, area_um2=area_um2),
            images={},
        )

    def finalize(self, per_image, per_pore):
        """Porosity of the whole ID: all images taken together as one large image."""
        return {"porosity_percent": float(100.0 * per_image["pore_area_px2"].sum() / per_image["sample_area_px2"].sum())}
