"""GrabCut: Graph-Schnitt mit gelernten Farbmodellen. **Das Standardverfahren.**

Ausgewählt am 12 Bilder umfassenden Vergleichssatz: GrabCut deckt den Saum am
vollständigsten ab und fängt auch die beiden Fälle, an denen das Schwellverfahren
``lasso`` zu wenig findet (``asdasd``: 48,7 % statt 14,6 %; ``Schweissnaht``: 25,7 %
statt 11,3 % - dort übersieht ``lasso`` das untere Harzband ganz).

Der Preis steht in ``tests/benchmark/test_specimen_real.py``: in der Gegenprobe an
Aufnahmen **ohne** Einbettmittel meldet GrabCut in 5 von 14 Fällen welches, ``lasso`` nur
in einem. Wer auf diese Seite empfindlich ist, stellt ``SpecimenConfig.method`` um - die
Verfahren sind austauschbar, und der Test hält den Stand aller sechs fest.

Das einzige Verfahren hier, das seine Entscheidungsgrundlage selbst lernt. Aus der Saat
schätzt es je ein Mischmodell (Gaußsche Mischverteilung) für Harz und Probe, bewertet
damit jedes Pixel, und schneidet dann den Graphen so, dass die Summe aus zwei Kosten
minimal wird:

* **Datenkosten** - wie schlecht passt das Pixel zu der Klasse, der es zugeschlagen wird,
* **Nachbarschaftskosten** - wie teuer ist es, zwischen zwei ähnlichen Nachbarn eine
  Grenze zu ziehen.

Der zweite Term ist der entscheidende Unterschied zum Schwellverfahren: eine Grenze
**kostet Länge**. Ein dünnes dunkles Band mitten in der Probe müsste zwei lange Grenzen
bezahlen und lohnt sich nicht - es bleibt bei der Probe, obwohl seine Grauwerte für Harz
sprächen. Genau das ist der Fehlerfall, an dem `lasso` scheitert.

Danach wird das Ganze wiederholt: aus dem Schnitt neue Modelle, aus den Modellen ein
neuer Schnitt. Drei Durchgänge genügen in der Regel.

GrabCut arbeitet auf Farbe. Wo ein Bild farbig ist, bekommt das Verfahren damit eine
Information, die alle anderen hier nicht nutzen - auch wenn sie sich an diesen Bildern
als schwach erwiesen hat (Farbabstand 1,4 bis 8,2 gegenüber 70 bis 146 Graustufen).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..config.schema import SpecimenConfig
from . import postprocess, seeds
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals

log = logging.getLogger(__name__)


class GrabCutSegmenter:
    """Trennt Probe und Einbettmittel durch Graph-Schnitt mit gelernten Modellen."""

    name = "grabcut"

    #: Das einzige Verfahren hier, das von einem Farbbild profitiert.
    uses_color = True

    def segment(self, gray: np.ndarray, cfg: SpecimenConfig) -> SpecimenMask:
        return self.from_signals(sig.compute(gray, cfg), cfg, color=None)

    def segment_color(
        self, gray: np.ndarray, color: np.ndarray | None, cfg: SpecimenConfig
    ) -> SpecimenMask:
        """Einstieg mit Farbbild - GrabCut ist das einzige Verfahren, das davon profitiert."""
        return self.from_signals(sig.compute(gray, cfg), cfg, color=color)

    def from_signals(
        self, signals: Signals, cfg: SpecimenConfig, color: np.ndarray | None = None
    ) -> SpecimenMask:
        seed = seeds.from_signals(signals, cfg.seeds)
        if not seed.usable:
            return postprocess.empty(
                signals, self.name,
                ("keine brauchbare Saat - am Bildrand ist nichts, was nach Harz aussieht",),
            )

        image = _bgr_at(signals, color)
        # GrabCut-Sprache: "sicher" für die Saat, "wahrscheinlich" für alles andere.
        # Alles Unbekannte kommt als "wahrscheinlich Probe" hinein, damit das Verfahren
        # den Zweifel selbst auflösen muss statt ihn geschenkt zu bekommen.
        mask = np.full(signals.shape, cv2.GC_PR_FGD, dtype=np.uint8)
        mask[seed.specimen] = cv2.GC_FGD
        mask[seed.resin] = cv2.GC_BGD

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, None, background_model, foreground_model,
                        cfg.grabcut.iterations, cv2.GC_INIT_WITH_MASK)
        except cv2.error as exc:
            log.warning("GrabCut ist gescheitert: %s", exc)
            return postprocess.empty(
                signals, self.name, (f"GrabCut ist gescheitert: {exc}",)
            )

        resin = (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)
        return postprocess.finish(resin, signals, cfg, self.name,
                                  debug={"seed_resin": seed.resin,
                                         "seed_specimen": seed.specimen})


def _bgr_at(signals: Signals, color: np.ndarray | None) -> np.ndarray:
    """Farbbild in Arbeitsauflösung, notfalls aus den Graustufen aufgebaut."""
    if color is None:
        return cv2.cvtColor(signals.smoothed, cv2.COLOR_GRAY2BGR)
    height, width = signals.shape
    resized = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)
