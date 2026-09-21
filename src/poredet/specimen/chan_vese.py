"""Chan-Vese: eine Kontur, die sich auf Regionenstatistik einpendelt.

Das Verfahren kennt gar keine Schwelle. Es zieht eine geschlossene Kontur durchs Bild und
verschiebt sie so lange, bis innen und außen jeweils möglichst **einheitlich** sind - es
minimiert die Streuung innerhalb der beiden Gebiete, zuzüglich eines Preises für die
Länge der Kontur. Eine Kante im Bild braucht es dafür nicht; entscheidend ist allein, wo
sich zwei Bereiche im Mittelwert unterscheiden.

Das macht es unempfindlich gegen verwaschene Übergänge - und gegen die Kantenabrundung am
Harzübergang, an der kantenbasierte Verfahren ins Rutschen kommen. Der Preis: es ist ein
**globales Zwei-Mittelwert-Modell**. Hat die Probe selbst stark unterschiedliche
Helligkeiten, teilt die Kontur womöglich das Metall statt Metall von Harz zu trennen -
dasselbe Problem wie beim Zweiklassen-Otsu, nur mit Geometrie.

Verwendet wird die morphologische Variante: statt partieller Differentialgleichungen nur
Dilatation und Erosion. Das ist um Größenordnungen schneller und numerisch nicht kaputt
zu bekommen.

Gerechnet wird auf Helligkeit und Textur zugleich - ohne die Textur unterscheidet das
Modell dunkles Gefüge nicht von Harz.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import SpecimenConfig
from . import postprocess, seeds
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals


class ChanVeseSegmenter:
    """Trennt Probe und Einbettmittel über eine Kontur, die Regionen vereinheitlicht."""

    name = "chan_vese"

    def segment(self, gray: np.ndarray, cfg: SpecimenConfig) -> SpecimenMask:
        return self.from_signals(sig.compute(gray, cfg), cfg)

    def from_signals(self, signals: Signals, cfg: SpecimenConfig) -> SpecimenMask:
        from skimage.segmentation import morphological_chan_vese

        chan = cfg.chan_vese
        seed = seeds.from_signals(signals, cfg.seeds)
        if not seed.resin.any():
            return postprocess.empty(
                signals, self.name,
                ("keine Startkontur - am Bildrand ist nichts, was nach Harz aussieht",),
            )

        small_shape = _target_shape(signals.shape, chan.max_px)
        data = _feature_image(signals)
        small = sig.downscale_values(data, small_shape)
        # Die Saat dient hier nur als Startkontur. Wohin sie sich bewegt, entscheidet
        # allein die Regionenstatistik - das ist der eigentliche Unterschied zu den
        # ausbreitenden Verfahren, bei denen die Saat festgenagelt bleibt.
        init = sig.upscale(seed.resin, small_shape)

        result = morphological_chan_vese(
            small,
            num_iter=chan.iterations,
            init_level_set=init.astype(np.uint8),
            smoothing=chan.smoothing,
            lambda1=chan.lambda1,
            lambda2=chan.lambda2,
        )

        resin = sig.upscale(result.astype(bool), signals.shape)
        notes: tuple[str, ...] = ()
        if resin.mean() > 0.9:
            notes += (
                ("die Kontur hat fast das ganze Bild eingeschlossen - das "
                 "Zwei-Mittelwert-Modell passt zu diesem Bild nicht"),
            )
        return postprocess.finish(resin, signals, cfg, self.name, notes,
                                  debug={"seed_resin": seed.resin})


def _feature_image(signals: Signals) -> np.ndarray:
    """Helligkeit und Textur zu einem Bild, auf 0…1 normiert - klein heißt "eher Harz"."""
    gray = signals.smoothed.astype(np.float32) / 255.0
    texture = signals.texture
    scale = float(np.percentile(texture, 99)) or 1.0
    return np.clip(0.5 * gray + 0.5 * np.clip(texture / scale, 0, 1), 0, 1)


def _target_shape(shape: tuple[int, int], max_px: int) -> tuple[int, int]:
    height, width = shape
    longest = max(height, width)
    if longest <= max_px:
        return shape
    factor = max_px / longest
    return (max(2, int(height * factor)), max(2, int(width * factor)))
