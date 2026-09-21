"""Probenbereich als Polygon aus der GUI, für Problemfälle.

Dieselbe Rolle wie der manuell gesetzte Maßstab: irgendein Bild wird immer einen
Harzübergang haben, den kein Verfahren findet - ein Ausbruch, ein Schleifgrat, eine
zweite Phase, die aussieht wie Einbettmittel. Dann muss der Prüfer die Probe umfahren
können, und das Ergebnis muss denselben Weg durchs Programm nehmen.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .base import SpecimenMask

Point = tuple[float, float]


def from_polygon(
    shape: tuple[int, int],
    polygon: Sequence[Point],
    holes: Sequence[Sequence[Point]] = (),
) -> SpecimenMask:
    """Probenmaske aus einem gezeichneten Polygon, optional mit ausgesparten Bereichen."""
    if len(polygon) < 3:
        raise ValueError("Ein Polygon braucht mindestens drei Punkte")

    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    for hole in holes:
        if len(hole) >= 3:
            cv2.fillPoly(mask, [np.asarray(hole, dtype=np.int32)], 0)

    specimen = mask.astype(bool)
    return SpecimenMask(
        specimen=specimen,
        resin=~specimen,
        method="manual_roi",
        components=1,
    )
