"""Random Walker: Diffusion von der Saat aus.

Die Idee stammt aus der Physik. Von jedem unbekannten Pixel startet gedanklich ein
Zufallsweg; er läuft leichter dorthin, wo sich die Werte kaum ändern, und schwer über
Kanten hinweg. Das Pixel bekommt die Klasse, deren Saat der Weg am wahrscheinlichsten
zuerst erreicht. Gelöst wird das nicht durch Simulation, sondern als lineares
Gleichungssystem über alle Pixel - das Ergebnis ist ein globales Optimum, kein
Schwellwertschnitt.

Der Unterschied zum Schwellverfahren ist grundsätzlich: dort entscheidet jedes Pixel für
sich, hier entscheidet der **Zusammenhang**. Ein dunkles Band mitten in der Probe wird
nicht deshalb Harz, weil es dunkel ist, sondern nur dann, wenn ein Weg von ihm zur
Harzsaat führt, ohne eine Kante zu überqueren. Genau daran scheitert der Fehlerfall des
Schwellverfahrens.

Gerechnet wird auf Helligkeit **und** Textur zugleich: der Unterschied zwischen Harz und
dunklem Gefüge steckt gerade in der Textur, und ein Verfahren, das nur Grauwerte sieht,
kann ihn nicht finden.
"""

from __future__ import annotations

import logging

import numpy as np

from ..einstellungen import Abschnitt
from . import postprocess, seeds
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals

log = logging.getLogger(__name__)


class RandomWalkerSegmenter:
    """Trennt Probe und Einbettmittel durch Diffusion von sicheren Saatpunkten aus."""

    name = "random_walker"

    def segment(self, gray: np.ndarray, cfg: Abschnitt) -> SpecimenMask:
        return self.from_signals(sig.compute(gray, cfg), cfg)

    def from_signals(self, signals: Signals, cfg: Abschnitt) -> SpecimenMask:
        from skimage.segmentation import random_walker

        seed = seeds.from_signals(signals, cfg.seeds)
        if not seed.usable:
            return postprocess.empty(
                signals, self.name,
                ("keine brauchbare Saat - am Bildrand ist nichts, was nach Harz aussieht",),
            )

        walker_cfg = cfg.random_walker
        # Das Verfahren löst ein Gleichungssystem über alle Pixel und wächst
        # überproportional; es rechnet deshalb auf einer eigenen, kleineren Größe.
        data = _feature_image(signals, walker_cfg.use_texture)
        small = sig.downscale_values(data, _target_shape(signals.shape, walker_cfg.max_px))
        labels = sig.upscale(seed.resin, small.shape).astype(np.int32)
        labels[sig.upscale(seed.specimen, small.shape)] = 2
        labels[sig.upscale(seed.resin, small.shape)] = 1

        try:
            result = random_walker(small, labels, beta=walker_cfg.beta,
                                   mode="cg_j", tol=1e-4, prob_tol=1e-2)
        except Exception as exc:  # pragma: no cover - numerisch entartete Bilder
            log.warning("Random Walker ist gescheitert: %s", exc)
            return postprocess.empty(
                signals, self.name, (f"Random Walker ist gescheitert: {exc}",)
            )

        resin = sig.upscale(result == 1, signals.shape)
        return postprocess.finish(resin, signals, cfg, self.name,
                                  debug={"seed_resin": seed.resin,
                                         "seed_specimen": seed.specimen})


def _feature_image(signals: Signals, use_texture: bool) -> np.ndarray:
    """Helligkeit und Textur zu einem Bild verrechnen, auf 0…1 normiert.

    Die Textur wird **abgezogen**: strukturierte Stellen werden hell, glatte dunkel.
    Damit zeigen beide Kanäle in dieselbe Richtung - klein heißt "eher Harz" -, und die
    Kantenstärke, an der sich der Zufallsweg bricht, addiert sich, statt sich aufzuheben.
    """
    gray = signals.smoothed.astype(np.float32) / 255.0
    if not use_texture:
        return gray
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
