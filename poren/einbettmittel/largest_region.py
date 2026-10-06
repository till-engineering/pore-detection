"""Größte zusammenhängende helle Region als Probe.

Der einfache Weg: hell = Probe, dunkel = alles andere. Er kommt ohne Texturmessung und
ohne Geometrie aus und ist damit schnell und vorhersagbar - aber er kann Harz und Poren
nicht unterscheiden und schneidet randnahe Poren aus der Probe heraus.

Sinnvoll als Vergleichsmaßstab und für Bilder, deren Probe klar heller ist als alles
andere. Für den Regelfall ist ``lasso`` zuständig.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..einstellungen import Abschnitt
from .base import SpecimenMask


class LargestRegionSegmenter:
    """Otsu, dann das größte helle Gebiet behalten."""

    name = "largest_region"

    def segment(self, gray: np.ndarray, cfg: Abschnitt) -> SpecimenMask:
        _level, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
        if count <= 1:
            shape = gray.shape[:2]
            return SpecimenMask(
                specimen=np.ones(shape, dtype=bool),
                resin=np.zeros(shape, dtype=bool),
                method=self.name,
                warnings=("kein helles Gebiet gefunden - das ganze Bild gilt als Probe",),
            )

        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(np.argmax(areas)) + 1
        specimen = labels == biggest
        return SpecimenMask(
            specimen=specimen,
            resin=~specimen,
            method=self.name,
            components=1,
            warnings=(
                ("largest_region kann Harz und Poren nicht unterscheiden - "
                 "randnahe Poren fallen aus der Probenmaske"),
            ),
        )
