"""Kontrollbilder zur Maßstabserkennung.

Eine erkannte Zahl allein ist wertlos - man sieht ihr nicht an, ob der richtige Balken
vermessen wurde. Deshalb zeigt jedes Kontrollbild drei Dinge getrennt:

* den **weißen Kasten** (grün) - der Bereich, der aus der Analyse fällt,
* den **schwarzen Balken** (rot) - das ist die Referenzlänge, nicht die Kastenbreite,
* die **Beschriftung** (blau) - der Ausschnitt, den die OCR gelesen hat.

Ohne die Trennung von Kasten und Balken sieht es auf dem Kontrollbild so aus, als wäre
die Kastenbreite der Maßstab - und ein Fehler von 30 % fällt niemandem auf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.models import BBox, ScaleInfo
from ..core.units import format_um
from .resolver import ScaleOutcome

# BGR
COLOR_BOX = (60, 190, 60)
COLOR_BAR = (40, 40, 235)
COLOR_LABEL = (230, 150, 40)
COLOR_REJECTED = (120, 120, 120)


@dataclass(frozen=True)
class SheetEntry:
    """Eine Zeile der Übersicht."""

    name: str
    image: np.ndarray            # BGR
    outcome: ScaleOutcome
    reference_um_per_px: float | None = None   # Ground Truth, falls bekannt

    @property
    def scale(self) -> ScaleInfo | None:
        return self.outcome.scale

    @property
    def deviation_pct(self) -> float | None:
        if self.scale is None or not self.reference_um_per_px:
            return None
        return 100.0 * (self.scale.um_per_px - self.reference_um_per_px) / self.reference_um_per_px


# --------------------------------------------------------------------------------------
# Einzelbild
# --------------------------------------------------------------------------------------


def annotate(image: np.ndarray, outcome: ScaleOutcome, thickness: int = 1) -> np.ndarray:
    """Erkanntes Overlay in eine Kopie des Bildes zeichnen."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Verworfene Kandidaten blass, damit sichtbar bleibt, was geprüft wurde.
    chosen_bar = outcome.scale.bar_box if outcome.scale else None
    for candidate in outcome.candidates:
        if chosen_bar is not None and candidate.bar == chosen_bar:
            continue
        _rect(canvas, candidate.box, COLOR_REJECTED, thickness)

    scale = outcome.scale
    if scale is None:
        return canvas

    if scale.box is not None:
        _rect(canvas, scale.box, COLOR_BOX, thickness)
    if scale.label_box is not None:
        _rect(canvas, scale.label_box, COLOR_LABEL, thickness)
    if scale.bar_box is not None:
        _bar_marker(canvas, scale.bar_box, thickness)
    return canvas


def _rect(canvas: np.ndarray, box: BBox | None, color: tuple[int, int, int], t: int) -> None:
    if box is None:
        return
    cv2.rectangle(canvas, (box.x, box.y), (box.x2 - 1, box.y2 - 1), color, t)


def _bar_marker(canvas: np.ndarray, bar: BBox, t: int) -> None:
    """Balken einrahmen und die vermessene Länge mit Endmarken kenntlich machen."""
    _rect(canvas, bar, COLOR_BAR, t)
    y = bar.y + bar.h // 2
    tick = max(3, bar.h * 2)
    for x in (bar.x, bar.x2 - 1):
        cv2.line(canvas, (x, y - tick), (x, y + tick), COLOR_BAR, t)


def mark_location(image: np.ndarray, box: BBox | None) -> np.ndarray:
    """Fundort für die verkleinerte Gesamtansicht auffällig umranden.

    Die feinen Markierungen aus :func:`annotate` verschwinden, sobald das Bild auf
    Daumennagelgröße skaliert wird - dann ist nicht mehr erkennbar, *wo* der Maßstab saß.
    Diese Umrandung ist bewusst grob: sie soll den Blick führen, nicht vermessen.
    """
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if box is None:
        return canvas
    height, width = canvas.shape[:2]
    pad = max(6, int(0.015 * max(width, height)))
    outer = box.padded(pad, width, height)
    cv2.rectangle(canvas, (outer.x, outer.y), (outer.x2 - 1, outer.y2 - 1),
                  COLOR_BOX, max(2, int(0.004 * max(width, height))))
    return canvas


def crop_around(image: np.ndarray, box: BBox | None, margin: float = 1.0,
                min_size: int = 90) -> np.ndarray:
    """Ausschnitt um das Overlay, großzügig gerandet - für die Übersicht."""
    height, width = image.shape[:2]
    if box is None:
        return image
    pad_x = max(int(margin * box.w), min_size // 2)
    pad_y = max(int(margin * box.h), min_size // 2)
    x0 = max(0, box.x - pad_x)
    y0 = max(0, box.y - pad_y)
    x1 = min(width, box.x2 + pad_x)
    y1 = min(height, box.y2 + pad_y)
    return image[y0:y1, x0:x1]


# --------------------------------------------------------------------------------------
# Übersicht über viele Bilder
# --------------------------------------------------------------------------------------


def contact_sheet(entries: list[SheetEntry], out_path: str | Path,
                  dpi: int = 110) -> Path:
    """Übersichtsblatt: je Bild das Gesamtbild und der Ausschnitt mit dem Overlay.

    Links steht, wo im Bild der Maßstab gefunden wurde, rechts, was genau vermessen
    wurde. Die Zahl daneben ist ohne diese beiden Bilder nicht prüfbar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
        raise ValueError("Keine Einträge für die Übersicht")

    rows = len(entries)
    fig, axes = plt.subplots(
        rows, 3,
        figsize=(13.0, 1.8 * rows), dpi=dpi,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]},
        layout="constrained",
    )
    if rows == 1:
        axes = np.array([axes])
    fig.get_layout_engine().set(hspace=0.04, wspace=0.01, h_pad=0.02, w_pad=0.02)

    fig.suptitle(
        "Maßstabserkennung     grün: Kasten = Ausschlussbereich      "
        "rot: vermessener Balken      blau: gelesene Beschriftung",
        fontsize=11,
    )

    for row, entry in enumerate(entries):
        annotated = annotate(entry.image, entry.outcome,
                             thickness=max(1, entry.image.shape[1] // 600))
        box = entry.scale.box if entry.scale else (
            entry.outcome.candidates[0].box if entry.outcome.candidates else None
        )

        ax = axes[row, 0]
        ax.imshow(cv2.cvtColor(mark_location(annotated, box), cv2.COLOR_BGR2RGB))
        ax.set_title(entry.name, fontsize=8, loc="left", pad=3)
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(cv2.cvtColor(crop_around(annotated, box, margin=0.6), cv2.COLOR_BGR2RGB),
                  interpolation="nearest")
        ax.set_title("Overlay im Detail", fontsize=8, loc="left", pad=3, color="#555555")
        ax.axis("off")

        ax = axes[row, 2]
        ax.axis("off")
        _write_verdict(ax, entry)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _write_verdict(ax, entry: SheetEntry) -> None:
    """Das Ergebnis als Textblock neben die Bilder.

    Kopfzeile und Körper sind zwei Textblöcke mit fester Reihenfolge, nicht mehrere frei
    platzierte - sonst überlagern sich lange Warnungen und Kennwerte.
    """
    scale = entry.scale
    if scale is None:
        reason = entry.outcome.rejections[0] if entry.outcome.rejections else "kein Overlay"
        ax.text(0.0, 1.0, "NICHT ERKANNT", fontsize=11, family="monospace",
                color="#b00000", weight="bold", va="top", transform=ax.transAxes)
        ax.text(0.0, 0.70, _wrap(reason, 58), fontsize=7.5, family="monospace",
                color="#b00000", va="top", transform=ax.transAxes, linespacing=1.5)
        return

    ax.text(0.0, 1.0, f"{scale.um_per_px:.6g} µm/px", fontsize=13,
            family="monospace", weight="bold", color="#14691f", va="top",
            transform=ax.transAxes)

    lines = [
        f'gelesen : "{scale.label_text}"  →  {format_um(scale.value_um or 0.0)}',
        f"Balken  : {scale.bar_length_px:g} px",
        (f"Quelle  : {scale.engine or scale.source}    "
         f"Konfidenz {scale.confidence:.0%}"),
    ]
    deviation = entry.deviation_pct
    if deviation is not None:
        lines.append(
            f"Referenz: {entry.reference_um_per_px:.6g} µm/px  ({deviation:+.3f} %)"
        )
    ax.text(0.0, 0.74, "\n".join(lines), fontsize=7.5, family="monospace",
            color="#222222", va="top", transform=ax.transAxes, linespacing=1.6)

    if scale.warnings:
        text = "\n".join(_wrap("! " + w, 74) for w in scale.warnings[:3])
        ax.text(0.0, 0.30, text, fontsize=6.5, family="monospace", color="#a86400",
                va="top", transform=ax.transAxes, linespacing=1.45)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text
