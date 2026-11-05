"""Stress fields: one YAML file per field in a folder (pore_index.stress_fields_dir), same format as
the Fatigue_crack_labeler (copied from there, keep both in step).

Variables: xi = x / length, zeta = z / thickness, both 0..1 in the sample frame (zeta = 0 inner /
bottom edge, 1 outer / top edge). A file has either `formula` (NumPy expression in xi and zeta)
or `grid` (support points, interpolated bilinearly):

    name: Bending 300 MPa       # optional, else the file name
    unit: MPa
    formula: "300 * abs(2*zeta - 1)"

    grid:
      xi:   [0.0, 0.5, 1.0]
      zeta: [0.0, 1.0]
      sigma: [[0, 10, 0], [300, 320, 300]]   # rows = zeta, columns = xi

For the stress, xi and zeta are limited to 0..1 (a crack exactly on the edge may lie a little
outside the fitted frame).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("pore_detection")

# names a formula may use besides xi and zeta
FORMULA_NAMES = {name: getattr(np, name) for name in (
    "abs", "sqrt", "exp", "log", "log10", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh",
    "tanh", "minimum", "maximum", "clip", "where", "sign", "floor", "ceil", "pi", "e")}
FORMULA_NAMES["min"], FORMULA_NAMES["max"] = np.minimum, np.maximum


@dataclass(frozen=True)
class StressField:
    key: str                    # file name without extension (stored in the report)
    name: str                   # shown name
    unit: str
    formula: str | None = None
    grid_xi: np.ndarray | None = None
    grid_zeta: np.ndarray | None = None
    grid_sigma: np.ndarray | None = None    # (len(zeta), len(xi))
    error: str = ""             # file not usable: why (the field cannot be selected)

    @property
    def ok(self) -> bool:
        return not self.error

    def evaluate(self, xi, zeta) -> np.ndarray:
        """Stress at (xi, zeta) (numbers or arrays of the same shape), limited to the sample."""
        xi = np.clip(np.asarray(xi, dtype=np.float64), 0.0, 1.0)
        zeta = np.clip(np.asarray(zeta, dtype=np.float64), 0.0, 1.0)
        if self.formula is not None:
            value = eval(self.formula, {"__builtins__": {}}, {**FORMULA_NAMES, "xi": xi, "zeta": zeta})
            return np.broadcast_to(np.asarray(value, dtype=np.float64), np.broadcast(xi, zeta).shape)
        return _bilinear(self.grid_xi, self.grid_zeta, self.grid_sigma, xi, zeta)

    def describe(self) -> str:
        if self.error:
            return f"not usable: {self.error}"
        if self.formula is not None:
            return f"σ = {self.formula} [{self.unit}]"
        return f"grid {len(self.grid_zeta)} × {len(self.grid_xi)} (zeta × xi) [{self.unit}]"


def load_fields(folder: Path) -> dict[str, StressField]:
    """All fields of the folder by key (sorted by name); unusable files with their error."""
    fields = {}
    for path in sorted(Path(folder).glob("*.y*ml"), key=lambda p: p.stem.lower()):
        try:
            fields[path.stem] = _load(path)
        except Exception as error:
            log.warning("stress field %s not usable: %s", path.name, error)
            fields[path.stem] = StressField(path.stem, path.stem, "", error=f"{type(error).__name__}: {error}")
    return fields


def usable_keys(folder: Path) -> list[str]:
    """Keys of the usable fields of the folder (selection in the launcher); [] if it does not exist."""
    if not Path(folder).is_dir():
        return []
    return [key for key, field in load_fields(folder).items() if field.ok]


def load_field(folder: Path, key: str) -> StressField:
    """One field by key (file name without extension); raises if it is missing or not usable."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"stress field folder {folder} does not exist")
    fields = load_fields(folder)
    if key not in fields:
        raise ValueError(f"stress field '{key}' not found in {folder} (available: {', '.join(fields) or 'none'})")
    if not fields[key].ok:
        raise ValueError(f"stress field '{key}' not usable: {fields[key].error}")
    return fields[key]


def _load(path: Path) -> StressField:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name, unit = str(data.get("name", path.stem)), str(data.get("unit", ""))
    if ("formula" in data) == ("grid" in data):
        raise ValueError("exactly one of 'formula' and 'grid' is needed")
    if "formula" in data:
        field = StressField(path.stem, name, unit, formula=str(data["formula"]))
    else:
        grid = data["grid"]
        xi, zeta = np.asarray(grid["xi"], dtype=np.float64), np.asarray(grid["zeta"], dtype=np.float64)
        sigma = np.asarray(grid["sigma"], dtype=np.float64)
        if sigma.shape != (len(zeta), len(xi)):
            raise ValueError(f"sigma has shape {sigma.shape}, expected (len(zeta), len(xi)) = ({len(zeta)}, {len(xi)})")
        if np.any(np.diff(xi) <= 0) or np.any(np.diff(zeta) <= 0):
            raise ValueError("xi and zeta must be strictly ascending")
        field = StressField(path.stem, name, unit, grid_xi=xi, grid_zeta=zeta, grid_sigma=sigma)
    field.evaluate(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))   # fails here, not during a run
    return field


def _bilinear(xs: np.ndarray, zs: np.ndarray, values: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a rectilinear grid; outside the support points the edge value."""
    x, z = np.broadcast_arrays(x, z)
    if len(xs) == 1 and len(zs) == 1:
        return np.full(x.shape, values[0, 0])
    if len(xs) == 1:
        return np.interp(z, zs, values[:, 0])
    if len(zs) == 1:
        return np.interp(x, xs, values[0])
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    tx = np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
    tz = np.clip((z - zs[j]) / (zs[j + 1] - zs[j]), 0.0, 1.0)
    bottom = values[j, i] * (1 - tx) + values[j, i + 1] * tx
    top = values[j + 1, i] * (1 - tx) + values[j + 1, i + 1] * tx
    return bottom * (1 - tz) + top * tz
