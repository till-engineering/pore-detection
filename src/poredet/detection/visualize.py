"""Kontrollbilder zur Porendetektion.

Eine Porositätszahl ist ohne Bild nicht prüfbar. "2,4 %" kann heißen, dass die Poren
sauber gefunden wurden - oder dass die Hälfte fehlt und dafür Gefügesprenkel mitgezählt
wurden. Deshalb zeigt jede Zeile vier Dinge:

* das **Originalbild**,
* die **gefundenen Poren** im Bild (rot), dazu die Probengrenze (grün) und der
  ausgeschlossene Maßstabskasten (blau),
* die **verworfenen** Kandidaten (grau) neben den behaltenen - so wird sichtbar, ob ein
  Filter zu scharf steht,
* das **Kontrastbild**, auf dem entschieden wurde.

Die letzte Spalte ist die wichtigste, wenn etwas fehlt: hebt sich eine Pore dort nicht
ab, hilft keine Schwelle der Welt, und das Problem liegt eine Stufe früher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.context import PipelineContext
from ..core.models import ImageResult

# BGR
COLOR_PORE = (60, 60, 235)
COLOR_REJECTED = (150, 150, 150)

#: Verworfene Poren werden nach dem Filter eingefaerbt, der sie getroffen hat. Eine
#: einheitliche Farbe wuerde die Frage offenlassen, die das Kontrollbild beantworten soll:
#: nicht "was wurde verworfen", sondern "warum".
COLOR_BY_FILTER = {
    "min_diameter": (200, 200, 90),    # tuerkis  - zu klein
    "scratch":      (200, 90, 200),    # magenta  - Schleifriefe
    "compactness":  (90, 170, 200),    # ocker    - zu fransig
    "roundness":    (110, 200, 200),   # gelb     - zu unrund
    "extent":       (160, 120, 200),
    "min_area":     (150, 150, 150),
    "edge":         (200, 140, 90),
}
COLOR_SPECIMEN = (70, 220, 70)
COLOR_EXCLUDED = (230, 160, 40)


@dataclass(frozen=True)
class SheetEntry:
    """Eine Zeile der Übersicht."""

    name: str
    context: PipelineContext
    result: ImageResult


def annotate(ctx: PipelineContext, show_rejected: bool = True) -> np.ndarray:
    """Gefundene Poren, verworfene Kandidaten und die Masken ins Bild zeichnen."""
    canvas = (ctx.color.copy() if ctx.color is not None
              else cv2.cvtColor(ctx.gray, cv2.COLOR_GRAY2BGR))
    dicke = max(1, int(0.0015 * max(canvas.shape[:2])))

    if show_rejected and ctx.rejected and ctx.labels is not None:
        nach_filter: dict[str, list[int]] = {}
        for eintrag in ctx.rejected:
            nach_filter.setdefault(eintrag.filter_name, []).append(eintrag.pore.label)
        for name, labels in nach_filter.items():
            farbe = COLOR_BY_FILTER.get(name, COLOR_REJECTED)
            _draw_regions(canvas, np.isin(ctx.labels, labels), farbe, dicke)

    if ctx.labels is not None and ctx.pores:
        gefunden = np.isin(ctx.labels, [p.label for p in ctx.pores])
        _draw_regions(canvas, gefunden, COLOR_PORE, dicke, fill_alpha=0.45)

    if ctx.specimen is not None:
        _draw_outline(canvas, ctx.specimen, COLOR_SPECIMEN, dicke)
    if ctx.excluded is not None and ctx.excluded.any():
        _draw_outline(canvas, ctx.excluded, COLOR_EXCLUDED, dicke + 1)
    return canvas


def contrast_view(ctx: PipelineContext) -> np.ndarray:
    """Das Kontrastbild als Graustufen - hell heißt "hebt sich dunkel ab"."""
    if ctx.contrast is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    werte = np.clip(ctx.contrast, 0, None)
    high = float(np.percentile(werte, 99.5)) or 1.0
    bild = np.clip(werte / high * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR)


def contact_sheet(entries: list[SheetEntry], out_path: str | Path, dpi: int = 100) -> Path:
    """Übersichtsblatt über viele Bilder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
        raise ValueError("Keine Einträge für die Übersicht")

    rows = len(entries)
    fig, axes = plt.subplots(
        rows, 4,
        figsize=(15.0, 2.6 * rows), dpi=dpi,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.35]},
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    fig.get_layout_engine().set(hspace=0.03, wspace=0.01, h_pad=0.02, w_pad=0.02)
    fig.suptitle(
        "Porendetektion     rot = gefundene Pore     verworfen nach Grund: "
        "türkis = zu klein,  magenta = Schleifriefe,  ocker = zu fransig,  "
        "gelb = zu unrund     grün = Probengrenze,  blau = Maßstab",
        fontsize=10,
    )

    for row, entry in enumerate(entries):
        ctx = entry.context
        panels = [
            ("Original", _thumb(ctx.color if ctx.color is not None
                                else cv2.cvtColor(ctx.gray, cv2.COLOR_GRAY2BGR))),
            ("gefunden", _thumb(annotate(ctx))),
            ("Kontrastbild", _thumb(contrast_view(ctx))),
        ]
        for column, (title, panel) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_title(title if row == 0 else "", fontsize=8, loc="left", pad=3,
                         color="#555555")
            ax.axis("off")

        ax = axes[row, 3]
        ax.axis("off")
        _write_verdict(ax, entry)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------


def _draw_regions(canvas, mask, color, dicke, fill_alpha: float = 0.0) -> None:
    if not mask.any():
        return
    if fill_alpha > 0:
        tint = np.zeros_like(canvas)
        tint[:] = color
        canvas[mask] = cv2.addWeighted(canvas, 1.0 - fill_alpha, tint, fill_alpha, 0.0)[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, dicke)


def _draw_outline(canvas, mask, color, dicke) -> None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, dicke)


def _thumb(image: np.ndarray, max_px: int = 620) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= max_px:
        return image
    factor = max_px / longest
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def _write_verdict(ax, entry: SheetEntry) -> None:
    result = entry.result
    ax.text(0.0, 1.0, entry.name[:42], fontsize=9, family="monospace", weight="bold",
            va="top", transform=ax.transAxes)

    porositaet = result.porosity_pct
    kopf = (f"{result.pore_count} Poren    {porositaet:.2f} % Porosität"
            if porositaet is not None else f"{result.pore_count} Poren")
    ax.text(0.0, 0.88, kopf, fontsize=11, family="monospace", weight="bold",
            color="#14691f" if result.pore_count else "#8a6d00",
            va="top", transform=ax.transAxes)

    groesste = result.largest_pore
    zeilen = []
    for name, text in result.stages.items():
        zeilen.append(f"{name:<10}: {text}")
    if groesste is not None:
        if groesste.equivalent_diameter_um is not None:
            zeilen.append(f"{'groesste':<10}: {groesste.equivalent_diameter_um:.1f} µm")
        else:
            zeilen.append(f"{'groesste':<10}: {groesste.equivalent_diameter_px:.0f} px")
    if result.pore_density_per_mm2 is not None:
        zeilen.append(f"{'Dichte':<10}: {result.pore_density_per_mm2:.1f} /mm²")

    ax.text(0.0, 0.76, "\n".join(_wrap(z, 58) for z in zeilen), fontsize=7,
            family="monospace", color="#222222", va="top", transform=ax.transAxes,
            linespacing=1.5)

    if result.warnings:
        text = "\n".join(_wrap("! " + w, 58) for w in result.warnings[:3])
        ax.text(0.0, 0.22, text, fontsize=6.5, family="monospace", color="#a86400",
                va="top", transform=ax.transAxes, linespacing=1.4)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width, subsequent_indent="            ")) or text
