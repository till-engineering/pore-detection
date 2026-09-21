"""Kontrollbilder zur Trennung Probe / Einbettmittel.

Eine Maske lässt sich nicht an einer Zahl beurteilen. "92 % Probe" kann heißen, dass der
Saum sauber abgetrennt wurde - oder dass die Hälfte des Harzes stehenblieb und ein
Probenstück fehlt. Deshalb zeigt jede Zeile vier Dinge:

* das **Originalbild** - was da eigentlich liegt,
* die **Maske im Bild** (rot überlagert = Harz), an der sich die Grenze beurteilen lässt,
* die **freigestellte Probe** - das, womit die Porendetektion später arbeitet,
* die **Zwischenschritte** getrennt: dunkel, strukturlos, und was beides erfüllt.

Die letzte Spalte ist die wichtigste, wenn etwas schiefgeht: sie zeigt, *welches* der
beiden Kriterien versagt hat. "Es funktioniert nicht" ist keine verwertbare Auskunft,
"die Texturschwelle greift zu tief" schon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .base import SpecimenMask

#: BGR
COLOR_RESIN = (60, 60, 235)
COLOR_EDGE = (70, 220, 70)


@dataclass(frozen=True)
class SheetEntry:
    """Eine Zeile der Übersicht."""

    name: str
    image: np.ndarray            # BGR, volle Auflösung
    mask: SpecimenMask


def overlay(image: np.ndarray, mask: SpecimenMask, alpha: float = 0.45) -> np.ndarray:
    """Harz rot einfärben und die Probengrenze nachziehen."""
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if mask.has_resin:
        tint = np.zeros_like(canvas)
        tint[:] = COLOR_RESIN
        resin = mask.resin
        canvas[resin] = cv2.addWeighted(
            canvas, 1.0 - alpha, tint, alpha, 0.0
        )[resin]

    contours, _hierarchy = cv2.findContours(
        mask.specimen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    thickness = max(1, int(0.002 * max(canvas.shape[:2])))
    cv2.drawContours(canvas, contours, -1, COLOR_EDGE, thickness)
    return canvas


def cutout(image: np.ndarray, mask: SpecimenMask, background: int = 255) -> np.ndarray:
    """Die freigestellte Probe - alles andere weiß.

    Weiß und nicht schwarz: Poren sind schwarz, und ein schwarzer Hintergrund ließe
    Randporen optisch mit dem weggeschnittenen Harz verschmelzen. Genau das soll das
    Kontrollbild ja zeigen.
    """
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas[~mask.specimen] = background
    return canvas


def steps_panel(mask: SpecimenMask) -> np.ndarray:
    """Die drei Zwischenmasken in einem Bild: dunkel, strukturlos, beides.

    Farbkodiert statt nebeneinander, damit die Spalte schmal bleibt:
    blau = nur dunkel, grün = nur strukturlos, weiß = beides (der Kandidat),
    rot umrandet = was davon als Harz übrig blieb.
    """
    dark = mask.debug.get("dark")
    smooth = mask.debug.get("smooth")
    resin = mask.debug.get("resin_work")
    if dark is None or smooth is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    canvas = np.zeros((*dark.shape, 3), dtype=np.uint8)
    canvas[dark & ~smooth] = (170, 90, 40)     # nur dunkel: dunkles Gefüge
    canvas[smooth & ~dark] = (40, 150, 70)     # nur glatt: helle, glatte Probe
    canvas[dark & smooth] = (235, 235, 235)    # Kandidat

    if resin is not None and resin.any():
        contours, _hierarchy = cv2.findContours(
            resin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, COLOR_RESIN, 2)
    return canvas


def contact_sheet(entries: list[SheetEntry], out_path: str | Path, dpi: int = 110) -> Path:
    """Übersichtsblatt über viele Bilder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
        raise ValueError("Keine Einträge für die Übersicht")

    rows = len(entries)
    fig, axes = plt.subplots(
        rows, 5,
        figsize=(16.0, 2.5 * rows), dpi=dpi,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.75, 1.25]},
        layout="constrained",
    )
    if rows == 1:
        axes = np.array([axes])
    fig.get_layout_engine().set(hspace=0.03, wspace=0.01, h_pad=0.02, w_pad=0.02)

    fig.suptitle(
        "Einbettmittel entfernen     rot = als Harz erkannt     grün = Probengrenze"
        "          Zwischenschritte: weiß = dunkel UND strukturlos,  "
        "braun = nur dunkel,  grün = nur strukturlos",
        fontsize=10,
    )

    for row, entry in enumerate(entries):
        panels = [
            ("Original", _thumb(entry.image)),
            ("erkannt", _thumb(overlay(entry.image, entry.mask))),
            ("freigestellte Probe", _thumb(cutout(entry.image, entry.mask))),
            ("Zwischenschritte", steps_panel(entry.mask)),
        ]
        for column, (title, panel) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_title(title if row == 0 else "", fontsize=8, loc="left", pad=3,
                         color="#555555")
            ax.axis("off")

        ax = axes[row, 4]
        ax.axis("off")
        _write_verdict(ax, entry)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _thumb(image: np.ndarray, max_px: int = 700) -> np.ndarray:
    """Für die Übersicht verkleinern - sonst wird das Blatt hunderte Megabyte groß."""
    longest = max(image.shape[:2])
    if longest <= max_px:
        return image
    factor = max_px / longest
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def _write_verdict(ax, entry: SheetEntry) -> None:
    mask = entry.mask
    ax.text(0.0, 1.0, entry.name, fontsize=9, family="monospace", weight="bold",
            va="top", transform=ax.transAxes)

    color = "#14691f" if mask.has_resin else "#8a6d00"
    headline = (
        f"Harz {mask.resin_frac:6.1%}   Probe {mask.specimen_frac:6.1%}"
        if mask.has_resin
        else "kein Einbettmittel gefunden"
    )
    ax.text(0.0, 0.84, headline, fontsize=11, family="monospace", weight="bold",
            color=color, va="top", transform=ax.transAxes)

    lines = [
        f"Verfahren     : {mask.method}",
        f"Probenstücke  : {mask.components}",
        f"Bildgröße     : {mask.shape[1]} x {mask.shape[0]} px",
    ]
    ax.text(0.0, 0.62, "\n".join(lines), fontsize=8, family="monospace", color="#222222",
            va="top", transform=ax.transAxes, linespacing=1.6)

    if mask.warnings:
        text = "\n".join(_wrap("! " + w, 46) for w in mask.warnings[:3])
        ax.text(0.0, 0.28, text, fontsize=7, family="monospace", color="#a86400",
                va="top", transform=ax.transAxes, linespacing=1.45)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text


# --------------------------------------------------------------------------------------
# Verfahrensvergleich
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """Ein Bild, von mehreren Verfahren bearbeitet."""

    name: str
    image: np.ndarray                      # BGR
    results: dict[str, SpecimenMask]       # Verfahren -> Ergebnis


def comparison_sheet(
    rows: list[ComparisonRow], methods: list[str], out_path: str | Path, dpi: int = 100
) -> Path:
    """Alle Verfahren nebeneinander, ein Bild je Zeile.

    Nebeneinander und nicht untereinander, weil sich Masken nur im direkten Vergleich
    beurteilen lassen: ob 15 % oder 20 % Harz richtig sind, sieht man keiner Zahl an -
    wohl aber, welche von sechs Grenzen dem Übergang im Bild folgt und welche nicht.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("Keine Einträge für den Vergleich")

    columns = 1 + len(methods)
    fig, axes = plt.subplots(
        len(rows), columns,
        figsize=(2.9 * columns, 2.15 * len(rows)), dpi=dpi, layout="constrained",
    )
    axes = np.atleast_2d(axes)
    fig.get_layout_engine().set(hspace=0.02, wspace=0.01, h_pad=0.01, w_pad=0.01)
    fig.suptitle(
        "Verfahrensvergleich: Einbettmittel entfernen     "
        "rot = als Harz erkannt,  grün = Probengrenze     "
        "! = das Verfahren meldet einen Vorbehalt",
        fontsize=11,
    )

    for row_index, row in enumerate(rows):
        ax = axes[row_index, 0]
        ax.imshow(cv2.cvtColor(_thumb(row.image, 420), cv2.COLOR_BGR2RGB))
        ax.set_title("Original" if row_index == 0 else "", fontsize=9, color="#555555")
        ax.set_ylabel(row.name[:26], fontsize=7, rotation=0, ha="right", va="center",
                      labelpad=6, family="monospace")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        for column, method in enumerate(methods, start=1):
            ax = axes[row_index, column]
            ax.axis("off")
            mask = row.results.get(method)
            if mask is None:
                ax.set_title(method if row_index == 0 else "", fontsize=9)
                continue
            ax.imshow(cv2.cvtColor(_thumb(overlay(row.image, mask), 420), cv2.COLOR_BGR2RGB))
            ax.set_title(method if row_index == 0 else "", fontsize=9, color="#555555")

            flag = " !" if mask.warnings else ""
            colour = "#a86400" if mask.warnings else ("#14691f" if mask.has_resin else "#999999")
            ax.text(0.02, 0.03, f"{mask.resin_frac:.1%}{flag}", transform=ax.transAxes,
                    fontsize=9, family="monospace", weight="bold", color=colour,
                    va="bottom", ha="left",
                    bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.5,
                          "edgecolor": "none"})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
