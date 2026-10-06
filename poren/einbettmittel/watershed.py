"""Marker-Watershed: Wasserscheide auf dem Gradientenbild.

Das Bild wird als Landschaft gelesen, in der die Höhe die Kantenstärke ist. An den
Saatpunkten beider Klassen wird geflutet; wo die Wasser aufeinandertreffen, liegt die
Grenze. Sie fällt dadurch **immer auf einen Kamm** - also auf eine echte Kante im Bild -
und nie mitten in eine gleichmäßige Fläche.

Das ist die Stärke und zugleich die Schwäche des Verfahrens: die Grenze sitzt genau, wo
ein Kontrastsprung ist, aber sie muss irgendwo hin. Zwischen zwei Saaten entsteht immer
eine Trennlinie, auch wenn in Wahrheit gar kein Übergang existiert. Das Verfahren kann
also nicht "kein Einbettmittel" sagen - dafür ist die Materialabnahme im gemeinsamen
Abschluss zuständig.

Ohne Marker wäre Watershed hier unbrauchbar: jedes lokale Minimum würde ein eigenes
Becken erzeugen und das Bild in hunderte Flecken zerlegen.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..einstellungen import Abschnitt
from . import postprocess, seeds
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals


class WatershedSegmenter:
    """Trennt Probe und Einbettmittel über die Wasserscheide zwischen zwei Saaten."""

    name = "watershed"

    def segment(self, gray: np.ndarray, cfg: Abschnitt) -> SpecimenMask:
        return self.from_signals(sig.compute(gray, cfg), cfg)

    def from_signals(self, signals: Signals, cfg: Abschnitt) -> SpecimenMask:
        from skimage.segmentation import watershed

        seed = seeds.from_signals(signals, cfg.seeds)
        if not seed.usable:
            return postprocess.empty(
                signals, self.name,
                ("keine brauchbare Saat - am Bildrand ist nichts, was nach Harz aussieht",),
            )

        elevation = _elevation(signals)
        result = watershed(elevation, seed.as_labels())
        resin = result == 1

        return postprocess.finish(resin, signals, cfg, self.name,
                                  debug={"seed_resin": seed.resin,
                                         "seed_specimen": seed.specimen,
                                         "elevation": elevation})


def _elevation(signals: Signals) -> np.ndarray:
    """Die Landschaft, auf der geflutet wird: Kantenstärke der Helligkeit.

    Geglättet, bevor der Gradient gebildet wird - sonst ist das geätzte Gefüge selbst ein
    Gebirge aus tausend Kämmen, und die Wasserscheide bleibt an der erstbesten Korngrenze
    hängen statt am Harzübergang.
    """
    blurred = cv2.GaussianBlur(signals.smoothed, (0, 0), 2.0)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)
    high = float(np.percentile(magnitude, 99)) or 1.0
    return np.clip(magnitude / high, 0, 1).astype(np.float32)
