"""JSON-Export des kompletten Ergebnisbaums.

Die maschinenlesbare Fassung: sie enthaelt alles, was die CSV-Dateien zeigen, und
zusaetzlich die Herkunftsangaben - welches Verfahren, welcher Massstab, welche Warnung.
Daraus entsteht spaeter der Bericht, und daraus laesst sich ein Lauf rekonstruieren,
ohne die Bilder erneut zu rechnen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...core.models import BatchResult, ImageResult, Pore


def write(batch: BatchResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(as_dict(batch), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def as_dict(batch: BatchResult) -> dict[str, Any]:
    return {
        "input_dir": str(batch.input_dir),
        "started_at": batch.started_at.isoformat(timespec="seconds"),
        "finished_at": (batch.finished_at.isoformat(timespec="seconds")
                        if batch.finished_at else None),
        "duration_s": batch.duration_s,
        "config": batch.config_summary,
        "total_pores": batch.total_pores,
        "mean_porosity_pct": batch.mean_porosity_pct,
        "images": [_image(r) for r in batch.results],
        "failures": [{"path": str(p), "reason": g} for p, g in batch.failures],
    }


def _image(result: ImageResult) -> dict[str, Any]:
    groesste = result.largest_pore
    return {
        "image": result.name,
        "width": result.width,
        "height": result.height,
        "scale": _scale(result),
        "specimen_area_px": result.specimen_area_px,
        "specimen_area_mm2": result.specimen_area_mm2,
        "excluded_area_px": result.excluded_area_px,
        "pore_count": result.pore_count,
        "pore_area_px": result.pore_area_px,
        "porosity_pct": result.porosity_pct,
        "porosity_pct_excl_edge": result.porosity_pct_excl_edge,
        "pore_density_per_mm2": result.pore_density_per_mm2,
        "largest_pore_um": groesste.equivalent_diameter_um if groesste else None,
        "stages": result.stages,
        "warnings": result.warnings,
        "pores": [_pore(p) for p in result.pores],
        "rejected": [
            {"label": r.pore.label, "filter": r.filter_name, "reason": r.reason}
            for r in result.rejected
        ],
    }


def _scale(result: ImageResult) -> dict[str, Any] | None:
    scale = result.scale
    if scale is None:
        return None
    return {
        "um_per_px": scale.um_per_px,
        "label_text": scale.label_text,
        "value_um": scale.value_um,
        "bar_length_px": scale.bar_length_px,
        "source": str(scale.source),
        "confidence": scale.confidence,
        "engine": scale.engine,
    }


def _pore(pore: Pore) -> dict[str, Any]:
    return {
        "label": pore.label,
        "area_px": pore.area_px,
        "area_um2": pore.area_um2,
        "equivalent_diameter_px": pore.equivalent_diameter_px,
        "equivalent_diameter_um": pore.equivalent_diameter_um,
        "perimeter_px": pore.perimeter_px,
        "feret_max_px": pore.feret_max_px,
        "feret_max_um": pore.feret_max_um,
        "major_axis_px": pore.major_axis_px,
        "minor_axis_px": pore.minor_axis_px,
        "circularity": pore.circularity,
        "aspect_ratio": None if pore.aspect_ratio == float("inf") else pore.aspect_ratio,
        "solidity": pore.solidity,
        "eccentricity": pore.eccentricity,
        "orientation_deg": pore.orientation_deg,
        "centroid_px": list(pore.centroid_px),
        "mean_intensity": pore.mean_intensity,
        "min_intensity": pore.min_intensity,
        "contrast": pore.contrast,
        "touches_image_edge": pore.touches_image_edge,
        "touches_specimen_edge": pore.touches_specimen_edge,
    }
