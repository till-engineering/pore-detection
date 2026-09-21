"""Manuell gesetzter Maßstab (Linie ziehen, Länge eintragen).

Kernfeature, keine Notlösung: eine Erkennung ohne Korrekturweg wird in der Praxis nicht
akzeptiert. Irgendein Bild wird immer ein Overlay haben, das kein Detektor findet - ein
abfotografierter Ausdruck, ein Zuschnitt ohne Kasten, ein Gerät mit eigener Bauform.
Dann muss der Prüfer eine Linie ziehen und die Länge eintippen können, und das Ergebnis
muss denselben Weg durchs Programm nehmen wie ein erkannter Maßstab.

Genau deshalb liefern beide Wege dasselbe :class:`~poredet.core.models.ScaleInfo`; nur
``source`` unterscheidet sie, und das steht später im Bericht.
"""

from __future__ import annotations

import math

from ..core.models import BBox, ScaleInfo, ScaleSource
from ..core.units import parse_length


def from_line(
    p0: tuple[float, float],
    p1: tuple[float, float],
    length_text: str,
) -> ScaleInfo:
    """Maßstab aus einer gezogenen Linie und einer Längenangabe wie ``"200 µm"``.

    Wirft :class:`ValueError`, wenn die Angabe unverständlich oder die Linie zu kurz ist -
    hier wird nichts geraten, der Benutzer steht ja daneben und kann es richtigstellen.
    """
    parsed = parse_length(length_text)
    if parsed is None:
        raise ValueError(
            f"{length_text!r} ist keine verständliche Längenangabe - erwartet wird "
            f"etwas wie '200 µm', '0.5 mm' oder '2 mm'"
        )

    length_px = math.dist(p0, p1)
    if length_px < 1.0:
        raise ValueError("Die gezogene Linie ist kürzer als ein Pixel")

    x0, y0 = min(p0[0], p1[0]), min(p0[1], p1[1])
    x1, y1 = max(p0[0], p1[0]), max(p0[1], p1[1])

    return ScaleInfo.from_bar(
        value_um=parsed.value_um,
        bar_length_px=length_px,
        source=ScaleSource.MANUAL,
        confidence=1.0,
        label_text=length_text.strip(),
        unit_text=parsed.unit,
        bar_box=BBox.from_corners(int(x0), int(y0), math.ceil(x1), math.ceil(y1)),
    )


def from_um_per_px(um_per_px: float, note: str | None = None) -> ScaleInfo:
    """Maßstab direkt als Zahl - etwa aus einer Kalibriertabelle je Objektiv."""
    return ScaleInfo(
        um_per_px=float(um_per_px),
        source=ScaleSource.MANUAL,
        confidence=1.0,
        warnings=(note,) if note else (),
    )
