"""Pore index (Porenkonzept): pore area weighted with the local stress, per image.

    phi_q     = sum_i A_i g_i^q / A_Mess
    phi_eff_q = w_o sum_(i in o) A_i g_i^q / A_Mess,o + w_u sum_(i in u) A_i g_i^q / A_Mess,u
    w_o = (n_o + 1) / (n_o + n_u + 2),  w_u = (n_u + 1) / (n_o + n_u + 2)

g is the stress field at the place of the pore. The field is one YAML file of the folder
stress_fields_dir (same format as the Fatigue_crack_labeler, see core/stress.py), selected by its
file name (`stress`) – the same files in both programs give the same stress at the same place.
Variables of the field: xi = x / length, zeta = z / thickness of the sample frame (0 = inner /
bottom edge, 1 = outer / top edge, see sample.frame), both limited to 0..1. The values are used
as they are (no normalisation, the user's choice; a field in MPa simply gives other numbers).

Zones: outer (o) = zeta >= 0.5, inner (u) = zeta < 0.5; A_Mess, A_Mess,o, A_Mess,u = sample area
in total / per zone. n_o, n_u = crack origins at the outer / inner surface found in tests (smoothed
to the weights w_o, w_u).

A pore reaching across the zone boundary is split pixel by pixel; each part counts as a pore of
its own. pore_stress selects how a part gets its stress:
    centroid  g at the centroid of the part, times its area
    area      every pixel with the g at its place (sum of g^q over the pixels)
    max       highest g in the part, times its area

g is defined in one place only (PoreIndex.stress_at). Hardness outer / inner (HBW, slide
"Porenkonzept") is an input that is recorded but not used in the model yet.

Per pore also its contribution A g^q (weighted_area_q…_px2, both zones, selected pore_stress) – the
sum over the pores is the numerator of phi_q. Image stress_field.tif: the field over the sample
rectangle (float32, row 0 = outer edge, column 0 = left edge; the viewer shows it as a layer).

Values per image only (no values over the ID yet). Areas cancel in phi: px and µm give the same.
Needs the sample frame (sample.frame), else the image fails.
"""

from pathlib import Path

import numpy as np

from ..core.config import PROJECT_DIR
from ..core.registry import register
from ..core.step import Evaluator
from ..core.stress import load_field, usable_keys
from ..core.types import EvaluationResult

PORE_STRESS = ("centroid", "area", "max")
ZONE_BOUNDARY = 0.5         # relative height of the boundary between inner and outer zone
BAND_ROWS = 1024            # rows per band when the sample area per zone is counted
MAP_LONG_PX = 768           # stress_field.tif: cells along the longer side of the sample (as the labeler)
MAP_MIN_PX = 8
# results of older runs (config_used.yaml without stress_fields_dir) used bending / pulsating, which
# are files of this folder now
DEFAULT_FIELDS_DIR = "stress_fields"


def q_name(q: float) -> str:
    """1 -> "q1", 1.5 -> "q1_5" (column names)."""
    return "q" + f"{q:g}".replace(".", "_")


def _fields_dir(value) -> Path:
    path = Path(value or DEFAULT_FIELDS_DIR)
    return path if path.is_absolute() else PROJECT_DIR / path


def _field_choices(params: dict) -> list[str]:
    """Selection of `stress` in the launcher: the usable fields of stress_fields_dir."""
    return usable_keys(_fields_dir(params.get("stress_fields_dir")))


@register("pore_index")
class PoreIndex(Evaluator):
    """Stress-weighted pore index phi_q and phi_eff_q per image (Porenkonzept)."""

    params = {
        "stress_fields_dir": "folder of the stress fields (one YAML file each, as the Fatigue_crack_labeler); "
                             "relative to the project folder",
        "stress": "stress field: file name in stress_fields_dir without .yaml",
        "pore_stress": "stress of a pore (part): centroid | area | max",
        "q_values": "exponents q, e.g. [1, 1.5, 2, 3, 5, 10]",
        "cracks_outer": "n_o: crack origins at the outer surface (tests)",
        "cracks_inner": "n_u: crack origins at the inner surface (tests)",
        "hardness_outer": "hardness at the outer surface in HBW (empty = not given; not used in the model yet)",
        "hardness_inner": "hardness at the inner surface in HBW (empty = not given; not used in the model yet)",
    }
    # stress: read from the folder whenever the launcher builds its form (new files appear without restart)
    choices = {"stress": _field_choices, "pore_stress": list(PORE_STRESS)}
    # test data of every single section: entered per image in the launcher (empty = config.yaml)
    image_params = {"cracks_outer", "cracks_inner", "hardness_outer", "hardness_inner"}
    excel_sheet = "Stress"
    excel_hidden = {"sample_area_outer_px2", "sample_area_inner_px2"}

    def __init__(self, **params):
        super().__init__(**params)
        if self.pore_stress not in PORE_STRESS:
            raise ValueError(f"unknown pore_stress '{self.pore_stress}' (available: {', '.join(PORE_STRESS)})")
        self.field = load_field(_fields_dir(getattr(self, "stress_fields_dir", None)), self.stress)
        # HBW or None; optional, so that results of older runs (config_used.yaml without it) can be corrected
        self.hardness_outer = _hardness(getattr(self, "hardness_outer", None), "hardness_outer")
        self.hardness_inner = _hardness(getattr(self, "hardness_inner", None), "hardness_inner")
        total = self.cracks_outer + self.cracks_inner + 2
        self.w_outer = (self.cracks_outer + 1) / total
        self.w_inner = (self.cracks_inner + 1) / total

    def evaluate(self, ctx, pores):
        frame = ctx.require_frame()
        area_outer, area_inner = self._zone_areas(ctx, frame)

        # every counted pore pixel: its pore, place in the frame and zone
        flat, ids = pores.pixels
        rows, cols = np.divmod(flat, pores.labels.shape[1])
        top, left = ctx.info.offset_px
        x, z = frame.to_sample(rows + top, cols + left)
        xi = np.clip(x / frame.length_px, 0.0, 1.0)
        zeta = np.clip(z / frame.thickness_px, 0.0, 1.0)
        outer = zeta >= ZONE_BOUNDARY
        g = self.stress_at(xi, zeta)

        # pore parts: (pore, zone) -> index
        index = np.searchsorted(pores.ids, ids)
        part = 2 * index + outer
        n_parts = 2 * pores.count
        part_area = np.bincount(part, minlength=n_parts).astype(np.float64)
        part_zone_outer = np.arange(n_parts) % 2 == 1

        # the inputs first: what the values below were computed with
        per_image = {"stress_field": self.stress, "stress_unit": self.field.unit, "pore_stress": self.pore_stress,
                     "cracks_outer": self.cracks_outer, "cracks_inner": self.cracks_inner,
                     "hardness_outer_hbw": _nan_if_none(self.hardness_outer),
                     "hardness_inner_hbw": _nan_if_none(self.hardness_inner)}
        weighted_per_pore = {}
        for q in self.q_values:
            weighted = self._weighted_parts(part, part_area, xi, zeta, g, q, n_parts)     # A g^q per part
            weighted_per_pore[f"weighted_area_{q_name(q)}_px2"] = weighted[0::2] + weighted[1::2]
            sum_outer, sum_inner = weighted[part_zone_outer].sum(), weighted[~part_zone_outer].sum()
            phi = (sum_outer + sum_inner) / (area_outer + area_inner)
            phi_eff = self.w_outer * _ratio(sum_outer, area_outer) + self.w_inner * _ratio(sum_inner, area_inner)
            per_image[f"phi_{q_name(q)}"] = float(phi)
            per_image[f"phi_eff_{q_name(q)}"] = float(phi_eff)
        per_image.update({"w_outer": self.w_outer, "w_inner": self.w_inner,
                          "sample_area_outer_px2": int(area_outer), "sample_area_inner_px2": int(area_inner)})

        # per pore (whole pore, independent of pore_stress): place and stress at the centroid, highest stress
        pore_area = np.maximum(np.bincount(index, minlength=pores.count), 1)
        pore_xi = np.bincount(index, weights=xi, minlength=pores.count) / pore_area
        pore_zeta = np.bincount(index, weights=zeta, minlength=pores.count) / pore_area
        pore_max = np.full(pores.count, -np.inf)
        np.maximum.at(pore_max, index, g)
        # + its contribution A g^q to phi_q (both zones, selected pore_stress): which pores matter
        per_pore = pores.properties[["pore_id"]].assign(
            xi=pore_xi, zeta=pore_zeta, stress_centroid=self.stress_at(pore_xi, pore_zeta), stress_max=pore_max,
            **weighted_per_pore)
        return EvaluationResult(per_image=per_image, per_pore=per_pore,
                                images={"stress_field": self._field_map(frame)})

    def stress_at(self, xi: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        """g at (xi, zeta) – the only place where g is defined; every value of this module (phi,
        phi_eff, per-pore columns), the plots and the layer of the viewer follow it. Also at hand
        for changes of the model: self.hardness_outer / self.hardness_inner (HBW, None = not given)."""
        return self.field.evaluate(xi, zeta)

    def _weighted_parts(self, part, part_area, xi, zeta, g, q, n_parts):
        """A g^q of every pore part for the selected pore_stress."""
        if self.pore_stress == "area":
            return np.bincount(part, weights=g ** q, minlength=n_parts)
        if self.pore_stress == "centroid":
            count = np.maximum(part_area, 1)
            part_xi = np.bincount(part, weights=xi, minlength=n_parts) / count
            part_zeta = np.bincount(part, weights=zeta, minlength=n_parts) / count
            return part_area * self.stress_at(part_xi, part_zeta) ** q
        part_max = np.zeros(n_parts)
        np.maximum.at(part_max, part, g)
        return part_area * part_max ** q

    def _field_map(self, frame) -> np.ndarray:
        """The field over the sample rectangle as the labeler draws it: float32, row 0 = outer edge."""
        length, thickness = frame.length_px, frame.thickness_px
        if length >= thickness:
            width = MAP_LONG_PX
            height = int(np.clip(round(MAP_LONG_PX * thickness / length), MAP_MIN_PX, MAP_LONG_PX))
        else:
            height = MAP_LONG_PX
            width = int(np.clip(round(MAP_LONG_PX * length / thickness), MAP_MIN_PX, MAP_LONG_PX))
        xi = (np.arange(width) + 0.5) / width
        zeta = 1.0 - (np.arange(height) + 0.5) / height
        return np.ascontiguousarray(self.stress_at(xi[None, :], zeta[:, None]), dtype=np.float32)

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


def _hardness(value, name: str) -> float | None:
    """Hardness input: number, or empty / null = not given. Text from the launcher may use a decimal comma."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        hardness = float(str(value).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{name}: '{value}' is not a number (HBW)") from None
    if hardness <= 0:
        raise ValueError(f"{name}: {hardness:g} HBW – hardness must be positive")
    return hardness


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else value
