"""Pore index (Porenkonzept): pore area weighted with the local stress, per image.

    phi_q     = sum_i A_i g_i^q / A_Mess
    phi_eff_q = w_o sum_(i in o) A_i g_i^q / A_Mess,o + w_u sum_(i in u) A_i g_i^q / A_Mess,u
    w_o = (n_o + 1) / (n_o + n_u + 2),  w_u = (n_u + 1) / (n_o + n_u + 2)

g is the relative stress (0..1) over the relative height zeta = z / thickness of the sample frame
(0 = inner / bottom edge, 1 = outer / top edge, see sample.frame). Zones: outer (o) = zeta >= 0.5,
inner (u) = zeta < 0.5; A_Mess, A_Mess,o, A_Mess,u = sample area in total / per zone. n_o, n_u =
crack origins at the outer / inner surface found in tests (smoothed to the weights w_o, w_u).

A pore reaching across the zone boundary is split pixel by pixel; each part counts as a pore of
its own. pore_stress selects how a part gets its stress:
    centroid  g at the centroid of the part, times its area
    area      every pixel with the g at its place (sum of g^q over the pixels)
    max       highest g in the part, times its area

Stress profiles (stress): bending = alternating bending, g = |2 zeta - 1| (1 at both surfaces,
0 in the neutral fibre). Further profiles (any stress field) later.

Values per image only (no values over the ID yet). Areas cancel in phi: px and µm give the same.
Needs the sample frame (sample.frame), else the image fails.
"""

import numpy as np

from ..core.registry import register
from ..core.step import Evaluator
from ..core.types import EvaluationResult

STRESS_PROFILES = {
    "bending": lambda zeta: np.abs(2 * zeta - 1),
}
PORE_STRESS = ("centroid", "area", "max")
ZONE_BOUNDARY = 0.5         # relative height of the boundary between inner and outer zone
BAND_ROWS = 1024            # rows per band when the sample area per zone is counted


def q_name(q: float) -> str:
    """1 -> "q1", 1.5 -> "q1_5" (column names)."""
    return "q" + f"{q:g}".replace(".", "_")


@register("pore_index")
class PoreIndex(Evaluator):
    """Stress-weighted pore index phi_q and phi_eff_q per image (Porenkonzept)."""

    params = {
        "stress": f"stress profile over the height of the sample ({' | '.join(STRESS_PROFILES)})",
        "pore_stress": "stress of a pore (part): centroid | area | max",
        "q_values": "exponents q, e.g. [1, 1.5, 2, 3, 5, 10]",
        "cracks_outer": "n_o: crack origins at the outer surface (tests)",
        "cracks_inner": "n_u: crack origins at the inner surface (tests)",
    }
    choices = {"stress": list(STRESS_PROFILES), "pore_stress": list(PORE_STRESS)}
    excel_sheet = "Stress"
    excel_hidden = {"sample_area_outer_px2", "sample_area_inner_px2"}

    def __init__(self, **params):
        super().__init__(**params)
        if self.stress not in STRESS_PROFILES:
            raise ValueError(f"unknown stress profile '{self.stress}' (available: {', '.join(STRESS_PROFILES)})")
        if self.pore_stress not in PORE_STRESS:
            raise ValueError(f"unknown pore_stress '{self.pore_stress}' (available: {', '.join(PORE_STRESS)})")
        self.profile = STRESS_PROFILES[self.stress]
        total = self.cracks_outer + self.cracks_inner + 2
        self.w_outer = (self.cracks_outer + 1) / total
        self.w_inner = (self.cracks_inner + 1) / total

    def evaluate(self, ctx, pores):
        frame = ctx.require_frame()
        area_outer, area_inner = self._zone_areas(ctx, frame)

        # every counted pore pixel: its pore, relative height and zone
        flat, ids = pores.pixels
        rows, cols = np.divmod(flat, pores.labels.shape[1])
        top, left = ctx.info.offset_px
        _, z = frame.to_sample(rows + top, cols + left)
        zeta = np.clip(z / frame.thickness_px, 0.0, 1.0)
        outer = zeta >= ZONE_BOUNDARY
        g = self.profile(zeta)

        # pore parts: (pore, zone) -> index
        index = np.searchsorted(pores.ids, ids)
        part = 2 * index + outer
        n_parts = 2 * pores.count
        part_area = np.bincount(part, minlength=n_parts).astype(np.float64)
        part_zone_outer = np.arange(n_parts) % 2 == 1

        # the inputs first: what the values below were computed with
        per_image = {"stress_profile": self.stress, "pore_stress": self.pore_stress,
                     "cracks_outer": self.cracks_outer, "cracks_inner": self.cracks_inner}
        for q in self.q_values:
            weighted = self._weighted_parts(part, part_area, zeta, g, q, n_parts)     # A g^q per part
            sum_outer, sum_inner = weighted[part_zone_outer].sum(), weighted[~part_zone_outer].sum()
            phi = (sum_outer + sum_inner) / (area_outer + area_inner)
            phi_eff = self.w_outer * _ratio(sum_outer, area_outer) + self.w_inner * _ratio(sum_inner, area_inner)
            per_image[f"phi_{q_name(q)}"] = float(phi)
            per_image[f"phi_eff_{q_name(q)}"] = float(phi_eff)
        per_image.update({"w_outer": self.w_outer, "w_inner": self.w_inner,
                          "sample_area_outer_px2": int(area_outer), "sample_area_inner_px2": int(area_inner)})

        # per pore (whole pore, independent of pore_stress): height and stress at the centroid, highest stress
        pore_area = np.bincount(index, minlength=pores.count)
        pore_zeta = np.bincount(index, weights=zeta, minlength=pores.count) / np.maximum(pore_area, 1)
        pore_max = np.zeros(pores.count)
        np.maximum.at(pore_max, index, g)
        per_pore = pores.properties[["pore_id"]].assign(
            zeta=pore_zeta, stress_centroid=self.profile(pore_zeta), stress_max=pore_max)
        return EvaluationResult(per_image=per_image, per_pore=per_pore, images={})

    def _weighted_parts(self, part, part_area, zeta, g, q, n_parts):
        """A g^q of every pore part for the selected pore_stress."""
        if self.pore_stress == "area":
            return np.bincount(part, weights=g ** q, minlength=n_parts)
        if self.pore_stress == "centroid":
            part_zeta = np.bincount(part, weights=zeta, minlength=n_parts) / np.maximum(part_area, 1)
            return part_area * self.profile(part_zeta) ** q
        part_max = np.zeros(n_parts)
        np.maximum.at(part_max, part, g)
        return part_area * part_max ** q

    @staticmethod
    def _zone_areas(ctx, frame) -> tuple[int, int]:
        """Sample pixels in the outer / inner zone, counted in bands (bounded memory)."""
        height, width = ctx.roi_mask.shape
        top, left = ctx.info.offset_px
        x = np.arange(left, left + width, dtype=np.float64)[None, :]
        boundary = ZONE_BOUNDARY * frame.thickness_px
        outer = inner = 0
        for start in range(0, height, BAND_ROWS):
            stop = min(start + BAND_ROWS, height)
            y = np.arange(top + start, top + stop, dtype=np.float64)[:, None]
            _, z = frame.to_sample(y, x)
            sample = ctx.roi_mask[start:stop]
            is_outer = z >= boundary
            outer += int(np.count_nonzero(sample & is_outer))
            inner += int(np.count_nonzero(sample & ~is_outer))
        return outer, inner


def _ratio(numerator: float, area: int) -> float:
    return numerator / area if area > 0 else float("nan")
