"""Protokolle: ScaleDetector liefert Kandidaten, OcrEngine liest den Text.

Die Trennung ist bewusst: **Geometrie und Lesen sind zwei Probleme.** Ein Detektor findet
den Balken und den Kasten - er weiß nichts von OCR. Eine Engine liest einen kleinen
Bildausschnitt - sie weiß nichts von Maßstäben. Zusammengeführt werden beide erst im
:mod:`poredet.scale.resolver`.

Dadurch lässt sich die OCR austauschen, ohne die Bildverarbeitung anzufassen, und ein
neuer Detektor für eine andere Overlay-Bauform kommt ohne eigene OCR aus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config.schema import ScaleConfig
from ..core.models import BBox

# OcrEngine und OcrResult sind in scale.ocr.base definiert und werden hier nur
# weitergereicht: wer gegen die Maßstabsschicht programmiert, soll alle ihre Protokolle
# an einer Stelle finden.
from .ocr.base import OcrEngine, OcrResult

# --------------------------------------------------------------------------------------
# Was ein Detektor findet
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlayCandidate:
    """Ein gefundenes Maßstabs-Overlay - noch ungelesen.

    ``bar`` ist die Referenzlänge, ``box`` der weiße Kasten und ``label`` der Bereich, in
    dem die Beschriftung steht. Erst der Resolver macht daraus einen Maßstab.
    """

    bar: BBox
    box: BBox
    label: BBox | None
    label_side: str | None          # "above" | "below" - auf welcher Seite des Balkens
    geometry_score: float           # 0..1, allein aus der Form begründet
    detector: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bar_length_px(self) -> float:
        """Die gemessene Referenzlänge. Die Balkenbreite, nicht die Kastenbreite."""
        return float(self.bar.w)


@runtime_checkable
class ScaleDetector(Protocol):
    """Findet Maßstabs-Overlays in einem Graustufenbild."""

    name: str

    def detect(self, gray: np.ndarray, cfg: ScaleConfig) -> list[OverlayCandidate]:
        """Liefert die Kandidaten, absteigend nach :attr:`geometry_score`."""
        ...


__all__ = ["OcrEngine", "OcrResult", "OverlayCandidate", "ScaleDetector"]
