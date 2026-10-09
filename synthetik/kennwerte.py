"""Kennwerte einer Pixelmaske - bewusst ohne den Messcode des Programms.

Die Prüfung soll den Messcode prüfen, nicht sich selbst. Deshalb steht hier eine eigene,
möglichst einfache Rechnung für jede Größe, die verglichen wird. Die Definitionen sind
dieselben wie im Programm (``poren/messung.py``):

* Fläche = Anzahl Pixel
* Schwerpunkt = Mittel der Pixelmitten (x, y)
* Bounding Box = kleinster Kasten, ``(x, y, breite, hoehe)``
* Äquivalentdurchmesser = Durchmesser des flächengleichen Kreises, sqrt(4A/pi)
* Feret max = größter Abstand zweier Punkte des Umrisses. Der Umriss läuft wie bei
  skimage durch die Kantenmitten zwischen Pore und Umgebung.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def kennwerte(maske: np.ndarray) -> dict | None:
    """Kennwerte der True-Pixel von ``maske``; ``None`` für eine leere Maske."""
    ys, xs = np.nonzero(maske)
    if len(xs) == 0:
        return None
    flaeche = len(xs)
    x0, y0 = int(xs.min()), int(ys.min())
    return {
        "flaeche_px": flaeche,
        "schwerpunkt_px": [float(xs.mean()), float(ys.mean())],
        "bbox_px": [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1],
        "aequivalentdurchmesser_px": math.sqrt(4.0 * flaeche / math.pi),
        "feret_max_px": _feret(xs, ys),
    }


def _feret(xs: np.ndarray, ys: np.ndarray) -> float:
    # Die vier Kantenmitten jedes Pixels; die konvexe Hülle aller Mitten ist dieselbe wie
    # die der Randmitten, und auf ihr liegt der größte Abstand.
    punkte = np.concatenate([
        np.stack([xs - 0.5, ys], axis=1), np.stack([xs + 0.5, ys], axis=1),
        np.stack([xs, ys - 0.5], axis=1), np.stack([xs, ys + 0.5], axis=1),
    ]).astype(np.float32)
    huelle = cv2.convexHull(punkte).reshape(-1, 2).astype(np.float64)
    if len(huelle) < 2:
        return 0.0
    abstand = np.sqrt(((huelle[:, None, :] - huelle[None, :, :]) ** 2).sum(axis=2))
    return float(abstand.max())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Überdeckung zweier Masken: Schnitt / Vereinigung (1 = deckungsgleich)."""
    vereinigung = int((a | b).sum())
    return 1.0 if vereinigung == 0 else int((a & b).sum()) / vereinigung
