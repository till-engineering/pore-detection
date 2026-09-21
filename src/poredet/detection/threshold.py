"""Schlichte Schwelle auf dem Grauwert, gerechnet nur innerhalb der Probe.

Das einfachste Verfahren und der Vergleichsmaßstab für alle anderen: schnell,
vorhersagbar, ohne Parameter, die man erst verstehen muss. Seine Grenze ist bekannt und
liegt in der Sache - es entscheidet jedes Pixel am absoluten Grauwert und ist damit
gegenüber ungleichmäßiger Ausleuchtung wehrlos. Wo die Ausleuchtung stimmt, reicht es.

``multiotsu_masked`` ist die brauchbarere der beiden Varianten. Bei geätztem Gefüge
trennt der Zweiklassen-Otsu nämlich die dunkle Gefügesprenkelung vom hellen Grundgefüge -
und nicht die Poren vom Rest. Echte Hohlräume sind die *dunkelste* Klasse; mit drei
Klassen und Stufe 0 werden genau sie herausgelöst.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config.schema import PoreConfig, ThresholdConfig
from .base import DetectionInput, DetectionResult


class ThresholdDetector:
    """Otsu, Multi-Otsu oder feste Schwelle - maskiert auf die Probe."""

    name = "threshold"

    def detect(self, data: DetectionInput, cfg: PoreConfig) -> DetectionResult:
        inside = data.inside(data.gray)
        if inside.size == 0:
            return DetectionResult(
                mask=np.zeros(data.shape, dtype=bool), method=self.name,
                notes=("keine Probenfläche - es wurde nichts ausgewertet",),
            )

        level = _level(inside, cfg.threshold)
        mask = (data.gray <= level) & data.specimen
        return DetectionResult(
            mask=mask, method=self.name, thresholds={"grauwert": float(level)}
        )


def _level(inside: np.ndarray, cfg: ThresholdConfig) -> float:
    if cfg.method == "fixed":
        return float(cfg.fixed) + cfg.offset

    values = inside.astype(np.uint8)
    if cfg.method == "multiotsu_masked":
        from skimage.filters import threshold_multiotsu

        try:
            levels = threshold_multiotsu(values, classes=cfg.multiotsu_classes)
        except ValueError:
            level, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            level = float(levels[min(cfg.multiotsu_level, len(levels) - 1)])
    else:
        level, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(level) + cfg.offset
