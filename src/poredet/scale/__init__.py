"""Maßstab aus dem eingebrannten Balken.

Das Ergebnis dieser Schicht ist ein :class:`~poredet.core.models.ScaleInfo` - und das ist
die einzige Schnittstelle, die spätere Module brauchen::

    outcome = ScaleResolver().resolve(image.gray, image.path)
    if outcome.scale:
        durchmesser_um = outcome.scale.to_um(durchmesser_px)
        flaeche_um2    = outcome.scale.to_um2(flaeche_px)
        nicht_werten   = overlay_mask(image.gray.shape, outcome.scale, pad=4)

Ohne erkannten Maßstab ist ``outcome.scale`` ``None``. Dann wird in Pixeln weiter
gemessen und die physikalischen Werte bleiben leer - es wird nirgends ein Faktor
unterstellt.
"""

from __future__ import annotations

from ..core.models import ScaleInfo, ScaleSource
from .base import OcrEngine, OcrResult, OverlayCandidate, ScaleDetector
from .detectors.box_overlay import BoxOverlayDetector
from .exclusion import exclude, excluded_area_px, overlay_mask
from .resolver import ScaleOutcome, ScaleResolver

__all__ = [
    "BoxOverlayDetector",
    "OcrEngine",
    "OcrResult",
    "OverlayCandidate",
    "ScaleDetector",
    "ScaleInfo",
    "ScaleOutcome",
    "ScaleResolver",
    "ScaleSource",
    "exclude",
    "excluded_area_px",
    "overlay_mask",
]
