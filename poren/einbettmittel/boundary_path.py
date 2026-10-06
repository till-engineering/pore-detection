"""Grenzverfolgung: die Trennlinie als billigster Weg quer durchs Bild.

Dieses Verfahren sucht nicht nach Gebieten, sondern nach **einer Linie**. Es baut ein
Kostenbild, in dem der Übergang Harz/Probe billig ist und alles andere teuer, und sucht
darin den günstigsten durchgehenden Weg von einer Bildkante zur gegenüberliegenden. Der
Weg ist die Probengrenze; was auf der einen Seite liegt, ist Harz.

Der Reiz liegt darin, dass **global** optimiert wird. Ein Schwellverfahren entscheidet
jedes Pixel für sich und kann deshalb an einer schwachen Stelle des Übergangs durchbrechen.
Ein Weg kann das nicht: er muss durchgehend sein. Eine kurze Störung wird umgangen, weil
der Umweg billiger ist als das Aufgeben - die Linie bleibt zusammenhängend, ohne dass man
ihr das eigens beibringen müsste.

Der Preis ist eine harte Annahme: **es gibt genau eine Grenze, und sie durchquert das
Bild.** Das trifft auf die meisten Schliffaufnahmen mit Randbereich zu - Harz links,
Probe rechts. Es trifft nicht zu, wenn das Harz die Probe als Rahmen umschließt oder in
zwei getrennten Bändern auftritt. Dort liefert das Verfahren zwangsläufig Unsinn, und
genau deshalb steht es hier neben den anderen und nicht an ihrer Stelle.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..einstellungen import Abschnitt
from . import postprocess, seeds
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals


class BoundaryPathSegmenter:
    """Trennt Probe und Einbettmittel entlang des billigsten Wegs quer durchs Bild."""

    name = "boundary_path"

    def segment(self, gray: np.ndarray, cfg: Abschnitt) -> SpecimenMask:
        return self.from_signals(sig.compute(gray, cfg), cfg)

    def from_signals(self, signals: Signals, cfg: Abschnitt) -> SpecimenMask:
        path_cfg = cfg.boundary_path
        seed = seeds.from_signals(signals, cfg.seeds)
        if not seed.resin.any():
            return postprocess.empty(
                signals, self.name,
                (("keine Harzsaat - ohne sie ist nicht entscheidbar, welche Seite "
                  "der Linie das Einbettmittel ist"),),
            )

        small_shape = _target_shape(signals.shape, path_cfg.max_px)
        costs = sig.downscale_values(_cost_image(signals, path_cfg), small_shape)
        resin_seed = sig.upscale(seed.resin, small_shape)

        # Beide Durchquerungsrichtungen probieren. Die Wahl fällt **nicht** über den Preis:
        # eine Linie quer zur Probengrenze ist oft die billigste, trennt aber nichts
        # Sinnvolles. Bewertet wird, wie gut die Aufteilung die beiden Saaten voneinander
        # trennt - Harzsaat auf die eine Seite, Probensaat auf die andere.
        #
        # Beide Anteile werden gebraucht. Nur nach der Harzsaat zu gehen belohnt die
        # entartete Lösung "eine Seite ist das ganze Bild": die enthält die Harzsaat
        # vollständig, aber eben auch alles andere.
        specimen_seed = sig.upscale(seed.specimen, small_shape)
        best: tuple[float, np.ndarray] | None = None
        for vertical in (True, False):
            found = _cheapest_crossing(costs, vertical)
            if found is None:
                continue
            _price, side = found
            for candidate in (side, ~side):
                quality = _separation(candidate, resin_seed, specimen_seed)
                if best is None or quality > best[0]:
                    best = (quality, candidate)

        if best is None:
            return postprocess.empty(
                signals, self.name, ("kein durchgehender Weg quer durchs Bild gefunden",)
            )

        purity, side = best
        notes: tuple[str, ...] = ()
        if purity < 0.9:
            notes += (
                (f"die Linie trennt die Saaten nur zu {purity:.0%} - die Annahme "
                 f"'genau eine Grenze quer durchs Bild' passt hier nicht"),
            )

        resin = sig.upscale(side, signals.shape)
        return postprocess.finish(resin, signals, cfg, self.name, notes,
                                  debug={"seed_resin": seed.resin, "cost": costs})


def _separation(side: np.ndarray, resin_seed: np.ndarray, specimen_seed: np.ndarray) -> float:
    """Wie gut die Aufteilung die beiden Saaten voneinander trennt (0…1).

    Der Mittelwert aus "wie viel Harzsaat liegt drinnen" und "wie viel Probensaat liegt
    draußen". Eine Seite, die das ganze Bild umfasst, erreicht im ersten Teil 1,0 und im
    zweiten 0,0 - und fällt damit durch.
    """
    resin_total = float(resin_seed.sum()) or 1.0
    specimen_total = float(specimen_seed.sum()) or 1.0
    inside = float((resin_seed & side).sum()) / resin_total
    outside = float((specimen_seed & ~side).sum()) / specimen_total
    return 0.5 * (inside + outside)


def _cost_image(signals: Signals, cfg) -> np.ndarray:
    """Billig dort, wo sich "sieht nach Harz aus" schnell ändert.

    Der Gradient wird **nicht** auf dem Grauwert gebildet, sondern auf einem weichen Maß
    für Harzähnlichkeit aus Helligkeit und Textur. Auf dem Grauwert wäre jede Korngrenze
    im geätzten Gefüge genauso billig wie der gesuchte Übergang, und der Weg liefe
    irgendwo durchs Metall.
    """
    gray = signals.smoothed.astype(np.float32) / 255.0
    texture = signals.texture
    scale = float(np.percentile(texture, 99)) or 1.0
    resinness = 1.0 - np.clip(0.5 * gray + 0.5 * np.clip(texture / scale, 0, 1), 0, 1)

    blurred = cv2.GaussianBlur(resinness, (0, 0), 2.0)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(dx, dy)
    high = float(np.percentile(edge, 99)) or 1.0
    edge = np.clip(edge / high, 0, 1)

    # Grundkosten je Schritt halten den Weg gerade: Umwege müssen sich lohnen.
    return (cfg.smoothness + cfg.gradient_weight * (1.0 - edge)).astype(np.float64)


def _cheapest_crossing(
    costs: np.ndarray, vertical: bool
) -> tuple[float, np.ndarray] | None:
    """Billigster Weg von einer Kante zur gegenüberliegenden, plus die Seitenaufteilung.

    ``vertical`` sucht von oben nach unten - die Linie trennt dann links von rechts.
    """
    from skimage.graph import MCP_Geometric

    work = costs if vertical else costs.T
    height, width = work.shape

    mcp = MCP_Geometric(work, fully_connected=True)
    starts = [(0, column) for column in range(width)]
    cumulative, _traceback = mcp.find_costs(starts)

    bottom = cumulative[-1, :]
    if not np.isfinite(bottom).any():
        return None
    end_column = int(np.nanargmin(np.where(np.isfinite(bottom), bottom, np.inf)))
    price = float(bottom[end_column])

    barrier = np.zeros(work.shape, dtype=np.uint8)
    for row, column in mcp.traceback((height - 1, end_column)):
        barrier[row, column] = 1

    # Die Linie ist 8-verbunden, die Flutung 4-verbunden - so kann sie nicht umgangen
    # werden, und die beiden Hälften bleiben sauber getrennt.
    count, labels = cv2.connectedComponents((1 - barrier).astype(np.uint8), connectivity=4)
    if count <= 1:
        return None
    left_label = labels[height // 2, 0]
    side = labels == left_label if left_label > 0 else labels == 1
    side |= barrier.astype(bool)

    return price, (side if vertical else side.T)


def _target_shape(shape: tuple[int, int], max_px: int) -> tuple[int, int]:
    height, width = shape
    longest = max(height, width)
    if longest <= max_px:
        return shape
    factor = max_px / longest
    return (max(2, int(height * factor)), max(2, int(width * factor)))
