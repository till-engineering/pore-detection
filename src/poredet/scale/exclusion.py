"""Maske des Overlay-Bereichs - sonst zählt der schwarze Balken als riesige Pore.

Das ist die zweite Hälfte dessen, was die Maßstabserkennung den späteren Modulen
liefert: nicht nur den Umrechnungsfaktor, sondern auch die Auskunft, *welcher Teil des
Bildes kein Gefüge ist*. Der Kasten ist eine perfekt rechteckige, tiefschwarz-weiße
Fläche - ohne Ausschluss wird er zur größten "Pore" des Bildes und verdirbt jede
Porositätszahl.
"""

from __future__ import annotations

import numpy as np

from ..core.models import BBox, ScaleInfo


def overlay_mask(
    shape: tuple[int, int], scale: ScaleInfo | None, pad: int = 0
) -> np.ndarray:
    """Boolesche Maske in Bildgröße: ``True`` = gehört zum Overlay, nicht auswerten.

    Ohne Maßstab oder ohne bekannten Kasten ist die Maske überall ``False`` - dann gibt
    es nichts auszuschließen.
    """
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    if scale is None:
        return mask
    box = scale.exclusion_box(pad=pad, width=width, height=height)
    if box is None or box.w <= 0 or box.h <= 0:
        return mask
    mask[box.slices()] = True
    return mask


def exclude(
    mask: np.ndarray, scale: ScaleInfo | None, pad: int = 0
) -> np.ndarray:
    """Overlay aus einer bestehenden Maske herausschneiden."""
    return mask & ~overlay_mask(mask.shape, scale, pad)


def excluded_area_px(
    shape: tuple[int, int], scale: ScaleInfo | None, pad: int = 0
) -> int:
    """Wie viele Pixel das Overlay der Auswertung entzieht.

    Die Zahl gehört in den Bericht: sie verkleinert die Bezugsfläche der Porosität.
    """
    box = None if scale is None else scale.exclusion_box(pad, shape[1], shape[0])
    return 0 if box is None else int(box.area)


def clip_to_image(box: BBox, shape: tuple[int, int]) -> BBox:
    """Box auf die Bildgrenzen stutzen."""
    height, width = shape[:2]
    x0 = max(0, min(box.x, width))
    y0 = max(0, min(box.y, height))
    x1 = max(x0, min(box.x2, width))
    y1 = max(y0, min(box.y2, height))
    return BBox.from_corners(x0, y0, x1, y1)
