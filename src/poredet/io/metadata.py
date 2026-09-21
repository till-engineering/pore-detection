"""TIFF-/ImageJ-Kalibrierung auslesen.

BEWUSST AUSSERHALB DES PRODUKTIVPFADS: der Maßstab kommt im Betrieb ausschließlich aus
dem eingebrannten Balken. Diese Tags dienen allein als Ground Truth im Benchmark der
Balkenerkennung - sie sagen, was herauskommen *müsste*, und machen aus "sieht richtig
aus" eine Zahl.

Wer hier eine Abkürzung für den Produktivbetrieb sucht: es ist keine. Die meisten
Schliffbilder kommen als JPG ohne jede Kalibrierung an, und ein Modul, das sich still
auf Metadaten verlässt, versagt genau dann, wenn niemand hinsieht.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..core.units import UNIT_FACTORS_UM, normalize_unit

#: ImageJ legt die Einheit als ASCII-escapten Text ab: "unit=µm".
_ESCAPE_RE = re.compile(r"\\u([0-9A-Fa-f]{4})")
_UNIT_RE = re.compile(r"unit=(\S+)")

_TAG_DESCRIPTION = 270
_TAG_X_RESOLUTION = 282


@dataclass(frozen=True)
class ImageJCalibration:
    """Die in einem TIFF hinterlegte ImageJ-Kalibrierung."""

    um_per_px: float
    unit: str
    pixels_per_unit: float


def read_imagej_calibration(path: str | Path) -> ImageJCalibration | None:
    """Kalibrierung aus den TIFF-Tags, oder ``None``, wenn keine vorhanden ist."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ist eine harte Abhängigkeit
        return None

    try:
        with Image.open(path) as image:
            tags = getattr(image, "tag_v2", None)
            if tags is None:
                return None
            description = str(tags.get(_TAG_DESCRIPTION, ""))
            resolution = tags.get(_TAG_X_RESOLUTION)
    except Exception:
        return None

    if resolution is None or "ImageJ" not in description:
        return None

    match = _UNIT_RE.search(description)
    if not match:
        return None

    raw_unit = _ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), match.group(1))
    unit, _corrections = normalize_unit(raw_unit)
    if unit is None or unit not in UNIT_FACTORS_UM:
        return None

    pixels_per_unit = float(resolution)
    if pixels_per_unit <= 0:
        return None

    return ImageJCalibration(
        um_per_px=UNIT_FACTORS_UM[unit] / pixels_per_unit,
        unit=unit,
        pixels_per_unit=pixels_per_unit,
    )
