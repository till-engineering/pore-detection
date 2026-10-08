"""Datenmodell: BBox, ScaleInfo, Pore, Masks, ImageResult, BatchResult.

Der Vertrag zwischen Analyse und jeder Ausgabeform. Geometrie wird immer in Pixeln
gespeichert; physikalische Werte sind abgeleitet und ``None``, solange kein Maßstab
bekannt ist - so wird nie mit einem geratenen Maßstab gerechnet.

Die Objekte hier sind der Vertrag zwischen Analyse und jeder Ausgabeform: CSV, JSON,
Kontrollbild, Bericht und später die GUI konsumieren sie, ohne das Innenleben der
Pipeline zu kennen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import formeln
from .einheiten import format_um

# --------------------------------------------------------------------------------------
# Geometrie
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """Achsenparallele Bounding Box in Pixelkoordinaten (Ursprung oben links)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else math.inf

    def slices(self) -> tuple[slice, slice]:
        """Slices zum Ausschneiden aus einem numpy-Array: ``img[bbox.slices()]``."""
        return (slice(self.y, self.y2), slice(self.x, self.x2))

    def padded(self, pad: int, width: int | None = None, height: int | None = None) -> BBox:
        """Um ``pad`` Pixel vergrößert, optional auf die Bildgrenzen begrenzt."""
        x = self.x - pad
        y = self.y - pad
        x2 = self.x2 + pad
        y2 = self.y2 + pad
        if width is not None:
            x, x2 = max(0, x), min(width, x2)
        if height is not None:
            y, y2 = max(0, y), min(height, y2)
        return BBox(x, y, max(0, x2 - x), max(0, y2 - y))

    def contains(self, other: BBox) -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.x2 <= self.x2
            and other.y2 <= self.y2
        )

    def union(self, other: BBox) -> BBox:
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        return BBox(x, y, max(self.x2, other.x2) - x, max(self.y2, other.y2) - y)

    @classmethod
    def from_corners(cls, x0: int, y0: int, x1: int, y1: int) -> BBox:
        return cls(int(x0), int(y0), int(x1 - x0), int(y1 - y0))


# --------------------------------------------------------------------------------------
# Maßstab
# --------------------------------------------------------------------------------------


class ScaleSource(StrEnum):
    """Woher der Maßstab stammt. Steht in jeder Ausgabe und in jedem Bericht."""

    OVERLAY_BAR = "overlay_bar"      # aus dem eingebrannten Balken gemessen
    MANUAL = "manual"                # vom Benutzer gesetzt (Linie + Länge)
    CONFIG_OVERRIDE = "config_override"  # in der Konfiguration fest vorgegeben


@dataclass(frozen=True)
class ScaleInfo:
    """Umrechnungsfaktor Pixel -> Mikrometer samt Herkunft und Rohwerten.

    Das ist die Schnittstelle, mit der **alle späteren Module** arbeiten. Sie messen in
    Pixeln und rufen am Ende :meth:`to_um` bzw. :meth:`to_um2` auf. Ist kein Maßstab
    bekannt, existiert kein ``ScaleInfo`` - dann bleiben die physikalischen Werte
    ``None``, statt dass irgendwo ein Faktor 1.0 unterstellt wird.

    Mitgeführt wird nicht nur die Zahl, sondern auch, wie sie zustande kam: Rohtext,
    gemessene Balkenlänge, Konfidenz und Warnungen. Ein stillschweigend falscher Maßstab
    verfälscht jede Messung, ohne aufzufallen - deshalb reist die Herkunft mit.
    """

    um_per_px: float
    source: ScaleSource

    # -- Belege ------------------------------------------------------------------------
    confidence: float = 1.0
    label_text: str | None = None          # was die OCR gelesen hat, unverändert
    value_um: float | None = None          # daraus geparster Wert in Mikrometer
    unit_text: str | None = None           # kanonische Einheit ("um", "mm", ...)
    bar_length_px: float | None = None     # DAS ist die gemessene Referenzlänge
    bar_box: BBox | None = None            # der schwarze Balken
    label_box: BBox | None = None          # die Beschriftung
    box: BBox | None = None                # der weiße Kasten, der beides umschließt
    engine: str | None = None              # welche OCR gelesen hat
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not math.isfinite(self.um_per_px) or self.um_per_px <= 0:
            raise ValueError(f"um_per_px muss positiv und endlich sein, ist {self.um_per_px!r}")

    # -- Umrechnung --------------------------------------------------------------------

    @property
    def px_per_um(self) -> float:
        return 1.0 / self.um_per_px

    def to_um(self, length_px: float) -> float:
        """Länge in Pixeln -> Mikrometer."""
        return length_px * self.um_per_px

    def to_um2(self, area_px: float) -> float:
        """Fläche in Pixeln -> Quadratmikrometer."""
        return area_px * self.um_per_px * self.um_per_px

    def to_mm(self, length_px: float) -> float:
        return self.to_um(length_px) / 1e3

    def to_mm2(self, area_px: float) -> float:
        return self.to_um2(area_px) / 1e6

    def to_px(self, length_um: float) -> float:
        """Mikrometer -> Pixel. Für Filtergrenzen, die physikalisch angegeben sind."""
        return length_um / self.um_per_px

    def area_to_px(self, area_um2: float) -> float:
        return area_um2 / (self.um_per_px * self.um_per_px)

    # -- Ausschlussbereich -------------------------------------------------------------

    def exclusion_box(self, pad: int = 0, width: int | None = None,
                      height: int | None = None) -> BBox | None:
        """Der Bildbereich, der nicht ausgewertet werden darf.

        Das eingebrannte Overlay ist kein Gefüge. Ohne diesen Ausschluss zählt der
        schwarze Balken als riesige Pore. ``None``, wenn der Maßstab nicht aus einem
        Overlay stammt.
        """
        if self.box is None:
            return None
        return self.box.padded(pad, width, height)

    # -- Darstellung -------------------------------------------------------------------

    @property
    def is_trustworthy(self) -> bool:
        """Ohne Warnungen und mit hoher Konfidenz gelesen."""
        return not self.warnings and self.confidence >= 0.8

    def describe(self) -> str:
        """Einzeiler für Log, Konsole und Kontrollbild."""
        parts = [f"{self.um_per_px:.6g} µm/px"]
        if self.label_text:
            parts.append(f'"{self.label_text}"')
        if self.bar_length_px:
            parts.append(f"Balken {self.bar_length_px:g} px")
        if self.value_um is not None:
            parts.append(f"= {format_um(self.value_um)}")
        parts.append(f"[{self.source}, {self.confidence:.0%}]")
        return "  ".join(parts)

    def with_warning(self, message: str, confidence_factor: float = 1.0) -> ScaleInfo:
        """Kopie mit einer zusätzlichen Warnung und ggf. gesenkter Konfidenz."""
        return replace(
            self,
            warnings=(*self.warnings, message),
            confidence=max(0.0, min(1.0, self.confidence * confidence_factor)),
        )

    # -- Konstruktoren -----------------------------------------------------------------

    @classmethod
    def from_bar(
        cls,
        *,
        value_um: float,
        bar_length_px: float,
        **kwargs: object,
    ) -> ScaleInfo:
        """Aus gemessener Balkenlänge und gelesener Beschriftung."""
        if bar_length_px <= 0:
            raise ValueError("Die Balkenlänge muss positiv sein")
        return cls(
            um_per_px=value_um / bar_length_px,
            value_um=value_um,
            bar_length_px=bar_length_px,
            **kwargs,  # type: ignore[arg-type]
        )

# --------------------------------------------------------------------------------------
# Pore
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pore:
    """Eine detektierte Pore mit allen abgeleiteten Kennwerten.

    Alle Geometriewerte stehen in Pixeln; die physikalischen sind abgeleitet und ``None``,
    solange kein Maßstab bekannt ist. ``um_per_px`` wird **pro Pore** mitgeführt, damit
    ein einzelnes Objekt für sich auswertbar bleibt - etwa wenn später Poren aus mehreren
    Bildern in einer gemeinsamen Statistik zusammenlaufen.
    """

    label: int
    area_px: float
    # Die Formwerte sind ``None``, solange die Pore nur grob vermessen ist - siehe
    # messung.measure_basis. Gezählte Poren haben sie immer.
    perimeter_px: float | None
    centroid_px: tuple[float, float]          # (x, y)
    bbox: BBox
    equivalent_diameter_px: float
    major_axis_px: float | None
    minor_axis_px: float | None
    feret_max_px: float | None
    solidity: float | None
    eccentricity: float | None
    orientation_deg: float | None
    mean_intensity: float
    min_intensity: float
    contrast: float                           # Abstand zum geschätzten Untergrund
    touches_image_edge: bool
    touches_specimen_edge: bool
    # Kleinster Abstand der Pore zum Probenrand in Pixeln (0 = berührt ihn). ``None``,
    # wenn die Probe keinen Rand hat - etwa bei full_frame.
    specimen_edge_distance_px: float | None = None
    um_per_px: float | None = None

    # -- Formkennwerte (maßstabsunabhängig) --------------------------------------------

    @property
    def vollstaendig(self) -> bool:
        """Sind auch die Formwerte gemessen?"""
        return self.feret_max_px is not None

    @property
    def circularity(self) -> float | None:
        """4πA / U² - siehe :func:`formeln.zirkularitaet`."""
        if self.perimeter_px is None:
            return None
        return formeln.zirkularitaet(self.area_px, self.perimeter_px)

    @property
    def aspect_ratio(self) -> float | None:
        """Haupt- durch Nebenachse - siehe :func:`formeln.seitenverhaeltnis`."""
        if self.major_axis_px is None or self.minor_axis_px is None:
            return None
        return formeln.seitenverhaeltnis(self.major_axis_px, self.minor_axis_px)

    @property
    def touches_any_edge(self) -> bool:
        return self.touches_image_edge or self.touches_specimen_edge

    # -- Physikalische Werte (None ohne Maßstab) ----------------------------------------

    def _um(self, value: float | None) -> float | None:
        return None if value is None else formeln.laenge_um(value, self.um_per_px)

    @property
    def area_um2(self) -> float | None:
        return formeln.flaeche_um2(self.area_px, self.um_per_px)

    @property
    def perimeter_um(self) -> float | None:
        return self._um(self.perimeter_px)

    @property
    def equivalent_diameter_um(self) -> float | None:
        return self._um(self.equivalent_diameter_px)

    @property
    def major_axis_um(self) -> float | None:
        return self._um(self.major_axis_px)

    @property
    def minor_axis_um(self) -> float | None:
        return self._um(self.minor_axis_px)

    @property
    def feret_max_um(self) -> float | None:
        return self._um(self.feret_max_px)

    @property
    def centroid_um(self) -> tuple[float, float] | None:
        if self.um_per_px is None:
            return None
        return (self.centroid_px[0] * self.um_per_px, self.centroid_px[1] * self.um_per_px)


@dataclass(frozen=True)
class RejectedPore:
    """Eine verworfene Pore samt Grund.

    Der Grund ist der eigentliche Wert dieses Objekts. "Warum fehlt diese Pore?" ist die
    häufigste Frage am Kontrollbild, und ohne den Grund lässt sie sich nur durch erneutes
    Durchrechnen beantworten.
    """

    pore: Pore
    filter_name: str
    reason: str


# --------------------------------------------------------------------------------------
# Ergebnisse
# --------------------------------------------------------------------------------------


@dataclass
class ImageResult:
    """Auswertung eines einzelnen Bildes.

    ``extras`` nimmt die Ergebnisse zusätzlicher Auswertungen auf, ohne dass dieses
    Modell dafür erweitert werden muss.
    """

    path: Path
    width: int
    height: int
    scale: ScaleInfo | None
    specimen_area_px: int
    excluded_area_px: int = 0
    pores: list[Pore] = field(default_factory=list)
    rejected: list[RejectedPore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def pore_count(self) -> int:
        return len(self.pores)

    @property
    def pore_area_px(self) -> float:
        return float(sum(p.area_px for p in self.pores))

    # -- Porosität ----------------------------------------------------------------------

    @property
    def porosity_pct(self) -> float | None:
        """Flächenanteil der Poren an der **Probenfläche**, nicht an der Bildfläche.

        Der Nenner ist der Punkt: bezöge man die Porosität auf das ganze Bild, wäre sie
        vom Anteil des Einbettmittels abhängig und damit von der Bildausschnittswahl -
        also keine Werkstoffkennzahl mehr.
        """
        return formeln.porositaet_pct(self.pore_area_px, self.specimen_area_px)

    @property
    def porosity_pct_excl_edge(self) -> float | None:
        """Wie :attr:`porosity_pct`, aber ohne angeschnittene Poren.

        Die Fläche der angeschnittenen Poren fällt aus Zähler **und** Nenner. Nur aus dem
        Zähler genommen, käme eine systematisch zu kleine Porosität heraus - ihre Fläche
        stünde weiter als porenfreie Probe im Nenner.
        """
        rand = sum(p.area_px for p in self.pores if p.touches_any_edge)
        return formeln.porositaet_pct(self.pore_area_px - rand, self.specimen_area_px - rand)

    # -- Kennwerte ----------------------------------------------------------------------

    @property
    def largest_pore(self) -> Pore | None:
        """Oft das eigentliche Abnahmekriterium - nicht die mittlere Porosität."""
        return max(self.pores, key=lambda p: p.area_px, default=None)

    @property
    def specimen_area_mm2(self) -> float | None:
        if self.scale is None:
            return None
        return self.scale.to_mm2(self.specimen_area_px)

    @property
    def pore_density_per_mm2(self) -> float | None:
        return formeln.porendichte(self.pore_count, self.specimen_area_mm2)

    @property
    def specimen_frac(self) -> float:
        total = self.width * self.height
        return self.specimen_area_px / total if total else 0.0

    def pores_sorted_by_area(self, descending: bool = True) -> list[Pore]:
        return sorted(self.pores, key=lambda p: p.area_px, reverse=descending)

    def describe(self) -> str:
        parts = [f"{self.pore_count} Poren"]
        if self.porosity_pct is not None:
            parts.append(f"{self.porosity_pct:.2f} % Porosität")
        largest = self.largest_pore
        if largest is not None:
            if largest.equivalent_diameter_um is not None:
                parts.append(f"größte {largest.equivalent_diameter_um:.1f} µm")
            else:
                parts.append(f"größte {largest.equivalent_diameter_px:.0f} px")
        if self.rejected:
            parts.append(f"{len(self.rejected)} verworfen")
        return ", ".join(parts)


@dataclass
class BatchResult:
    """Ergebnis eines kompletten Ordnerdurchlaufs."""

    input_dir: Path
    results: list[ImageResult] = field(default_factory=list)
    failures: list[tuple[Path, str]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    config_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def total_pores(self) -> int:
        return sum(r.pore_count for r in self.results)

    @property
    def images_without_scale(self) -> list[ImageResult]:
        return [r for r in self.results if r.scale is None]

    @property
    def mean_porosity_pct(self) -> float | None:
        """Flächengewichtete Gesamtporosität über alle Bilder.

        Flächengewichtet und nicht als Mittel der Einzelwerte: ein Bild mit wenig
        Probenfläche darf die Kennzahl der Probe nicht genauso stark ziehen wie ein
        vollflächiges.
        """
        specimen = sum(r.specimen_area_px for r in self.results)
        if specimen <= 0:
            return None
        return 100.0 * sum(r.pore_area_px for r in self.results) / specimen

    @property
    def largest_pore(self) -> tuple[ImageResult, Pore] | None:
        best: tuple[ImageResult, Pore] | None = None
        for result in self.results:
            pore = result.largest_pore
            if pore is not None and (best is None or pore.area_px > best[1].area_px):
                best = (result, pore)
        return best
