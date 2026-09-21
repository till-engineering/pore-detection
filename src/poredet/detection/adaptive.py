"""Sauvola - lokal adaptive Schwelle.

Jedes Pixel bekommt seine eigene Schwelle aus Mittelwert und Streuung seiner Umgebung.
Das macht das Verfahren unempfindlich gegen Helligkeitsverläufe, ohne dass ein Untergrund
geschätzt werden müsste - und anders als reine Mittelwertverfahren berücksichtigt Sauvola
die lokale Streuung: in einer gleichmäßigen Fläche ohne Struktur wird kaum etwas
gefunden, was Fehlalarme im homogenen Grundgefüge dämpft.

Die Schwäche ist die Kehrseite: das Fenster muss deutlich größer sein als die Poren.
Ist es zu klein, sieht es *innerhalb* einer großen Pore nur Poreninneres, hält das für
den Normalzustand und findet dort nichts mehr - die Pore wird zum Ring.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import PoreConfig
from .base import DetectionInput, DetectionResult


class AdaptiveDetector:
    """Lokal adaptive Schwelle nach Sauvola."""

    name = "adaptive"

    def detect(self, data: DetectionInput, cfg: PoreConfig) -> DetectionResult:
        from skimage.filters import threshold_sauvola

        window = cfg.adaptive.window_px | 1
        level = threshold_sauvola(data.gray, window_size=window, k=cfg.adaptive.k)
        mask = (data.gray < level) & data.specimen
        return DetectionResult(
            mask=mask, method=self.name,
            thresholds={"fenster_px": float(window), "k": float(cfg.adaptive.k)},
        )
