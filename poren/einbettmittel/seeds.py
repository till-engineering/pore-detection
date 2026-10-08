"""Saatpunkte: was sicher Harz ist und was sicher Probe.

Die ausbreitenden Verfahren - Random Walker, Watershed, GrabCut - brauchen einen
Startpunkt. Dessen Qualität entscheidet mehr über das Ergebnis als das Verfahren selbst,
deshalb steht er hier an einer Stelle und nicht sechsmal leicht verschieden.

Zwei Entwurfsentscheidungen:

**Die Saat ist zurückhaltend, nicht vollständig.** Sie muss das Harz nicht abdecken - nur
darin liegen. Die Ausbreitung besorgt den Rest. Eine großzügige Saat nimmt dem Verfahren
genau die Arbeit ab, die geprüft werden soll.

**Harzsaat wird nur im Randstreifen gesucht.** Das Sichtfeld liegt innerhalb des
Einbettlings, Harz berührt also den Bildrand. Ein dunkles Gefügeband mitten in der Probe
kann so gar nicht erst zur Saat werden - der Fehlerfall, an dem das Schwellverfahren
scheitert, ist hier durch die Konstruktion ausgeschlossen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..einstellungen import Abschnitt
from . import signals as sig
from .signals import Signals


@dataclass(frozen=True)
class Seeds:
    """Sichere Anfangspunkte beider Klassen, in Arbeitsauflösung."""

    resin: np.ndarray            # bool
    specimen: np.ndarray         # bool

    @property
    def usable(self) -> bool:
        """Ohne beide Klassen lässt sich nichts ausbreiten."""
        return bool(self.resin.any() and self.specimen.any())

    def as_labels(self, unknown: int = 0, resin: int = 1, specimen: int = 2) -> np.ndarray:
        """Label-Bild, wie es Random Walker und Watershed erwarten."""
        labels = np.full(self.resin.shape, unknown, dtype=np.int32)
        labels[self.specimen] = specimen
        labels[self.resin] = resin
        return labels


def from_signals(signals: Signals, cfg: Abschnitt) -> Seeds:
    """Saat aus Helligkeit, Textur und Lage am Bildrand."""
    height, width = signals.shape
    band_px = max(2, int(cfg.border_band_frac * min(height, width)))

    # -- Harz: dunkel, strukturlos, am Rand ---------------------------------------------
    candidate = signals.is_dark & signals.is_smooth
    band = sig.border_band((height, width), band_px)
    resin = sig.erode(candidate & band, cfg.erode_px)
    resin = _drop_small(resin, cfg.min_seed_frac * height * width)

    # -- Probe: hell oder strukturiert, weg vom Rand ------------------------------------
    # "oder" ist wichtig: dunkles Gefüge ist Probe, solange es Struktur hat. Genau diese
    # Bereiche sind die schwierigen, und sie gehören auf die richtige Seite der Saat.
    bright_or_textured = ~signals.is_dark | ~signals.is_smooth
    specimen = sig.erode(bright_or_textured & ~band, cfg.erode_px)
    specimen = _drop_small(specimen, cfg.min_seed_frac * height * width)

    return Seeds(resin=resin, specimen=specimen)


def _drop_small(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Saatflecken unterhalb einer Mindestgröße verwerfen - sie sind meist Rauschen."""
    import cv2

    if not mask.any():
        return mask
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    # Nachschlagetabelle Label -> behalten, statt je Fleck einmal übers ganze Bild.
    behalten = stats[:, cv2.CC_STAT_AREA] >= min_area
    behalten[0] = False
    out = behalten[labels]
    return out if out.any() else mask
