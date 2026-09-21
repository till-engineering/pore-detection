"""CSV-Export: eine Zeile je Pore, eine Zeile je Bild.

Zwei getrennte Dateien statt einer breiten: die Porentabelle ist die Rohmessung, die
Bildtabelle die Auswertung. Wer eine eigene Statistik rechnen will, braucht die erste;
wer Proben vergleichen will, die zweite.

Geschrieben wird mit Semikolon und Dezimalkomma - so öffnet Excel auf einem deutschen
System die Datei ohne Nachfrage und ohne dass aus "1.5" ein Datum wird.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ...core.models import BatchResult, ImageResult

PORE_COLUMNS = [
    "bild", "label", "flaeche_px", "flaeche_um2", "aequivalentdurchmesser_px",
    "aequivalentdurchmesser_um", "umfang_px", "feret_max_px", "feret_max_um",
    "hauptachse_px", "nebenachse_px", "rundheit", "seitenverhaeltnis", "soliditaet",
    "exzentrizitaet", "orientierung_grad", "schwerpunkt_x_px", "schwerpunkt_y_px",
    "grauwert_mittel", "grauwert_min", "kontrast", "am_bildrand", "am_probenrand",
]

IMAGE_COLUMNS = [
    "bild", "breite_px", "hoehe_px", "um_per_px", "massstab_text",
    "probenflaeche_px", "probenflaeche_mm2", "ausgeschlossen_px",
    "porenzahl", "porenflaeche_px", "porositaet_pct", "porositaet_pct_ohne_randporen",
    "porendichte_pro_mm2", "groesste_pore_um", "verworfen", "warnungen",
]


def write_pores(batch: BatchResult, path: str | Path) -> Path:
    """Eine Zeile je gefundener Pore."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(PORE_COLUMNS)
        for result in batch.results:
            for pore in result.pores:
                writer.writerow([
                    result.name, pore.label,
                    _num(pore.area_px), _num(pore.area_um2),
                    _num(pore.equivalent_diameter_px), _num(pore.equivalent_diameter_um),
                    _num(pore.perimeter_px), _num(pore.feret_max_px), _num(pore.feret_max_um),
                    _num(pore.major_axis_px), _num(pore.minor_axis_px),
                    _num(pore.circularity), _num(pore.aspect_ratio), _num(pore.solidity),
                    _num(pore.eccentricity), _num(pore.orientation_deg),
                    _num(pore.centroid_px[0]), _num(pore.centroid_px[1]),
                    _num(pore.mean_intensity), _num(pore.min_intensity), _num(pore.contrast),
                    _bool(pore.touches_image_edge), _bool(pore.touches_specimen_edge),
                ])
    return path


def write_images(batch: BatchResult, path: str | Path) -> Path:
    """Eine Zeile je Bild."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(IMAGE_COLUMNS)
        for result in batch.results:
            groesste = result.largest_pore
            writer.writerow([
                result.name, result.width, result.height,
                _num(result.scale.um_per_px if result.scale else None),
                result.scale.label_text if result.scale else "",
                result.specimen_area_px, _num(result.specimen_area_mm2),
                result.excluded_area_px,
                result.pore_count, _num(result.pore_area_px),
                _num(result.porosity_pct), _num(result.porosity_pct_excl_edge),
                _num(result.pore_density_per_mm2),
                _num(groesste.equivalent_diameter_um if groesste else None),
                len(result.rejected),
                " | ".join(result.warnings),
            ])
    return path


def _num(value: float | None) -> str:
    """Zahl mit Dezimalkomma, damit Excel sie ohne Umweg als Zahl liest."""
    if value is None:
        return ""
    return f"{value:.6g}".replace(".", ",")


def _bool(value: bool) -> str:
    return "ja" if value else "nein"


REJECTED_COLUMNS = [
    "bild", "label", "filter", "grund", "flaeche_px", "durchmesser_px",
    "seitenverhaeltnis", "soliditaet", "rundheit", "nebenachse_px",
]


def write_rejected(batch: BatchResult, path: str | Path) -> Path:
    """Eine Zeile je verworfener Pore - mit dem Grund.

    Die wichtigste der drei Tabellen beim Einstellen der Filter: sie beantwortet
    "warum fehlt diese Pore?" ohne erneutes Durchrechnen.
    """
    from ...analysis.geometry import roundness

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(REJECTED_COLUMNS)
        for result in batch.results:
            for eintrag in result.rejected:
                pore = eintrag.pore
                writer.writerow([
                    result.name, pore.label, eintrag.filter_name, eintrag.reason,
                    _num(pore.area_px), _num(pore.equivalent_diameter_px),
                    _num(pore.aspect_ratio if pore.aspect_ratio != float("inf") else None),
                    _num(pore.solidity), _num(roundness(pore)), _num(pore.minor_axis_px),
                ])
    return path


def rejection_summary(batch: BatchResult) -> dict[str, int]:
    """Wie oft jeder Filter gegriffen hat - über den ganzen Lauf."""
    zaehler: dict[str, int] = {}
    for result in batch.results:
        for eintrag in result.rejected:
            zaehler[eintrag.filter_name] = zaehler.get(eintrag.filter_name, 0) + 1
    return dict(sorted(zaehler.items(), key=lambda item: -item[1]))


def write_all(batch: BatchResult, directory: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(directory)
    return (
        write_pores(batch, directory / "poren.csv"),
        write_images(batch, directory / "bilder.csv"),
        write_rejected(batch, directory / "verworfen.csv"),
    )


def summary_line(result: ImageResult) -> str:
    """Eine Zeile für die Konsole."""
    return f"{result.name}: {result.describe()}"
