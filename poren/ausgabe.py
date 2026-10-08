"""Was im Zielordner landet: CSV-Tabellen und je Bild ein Ergebnisbild.

Im Zielordner, über alle Bilder:

* ``poren.csv``      eine Zeile je gezählter Pore (Rohmessung)
* ``bilder.csv``     eine Zeile je Bild (Porosität, Porenzahl, Maßstab, ...)
* ``verworfen.csv``  eine Zeile je verworfenem Objekt, mit Grund

Je Bild im Unterordner ``<bild>/`` (:func:`write_bild`):

* ``<bild>_ergebnis.png``    links das Original, rechts dasselbe mit den gezählten Poren rot
* ``<bild>_maske.png``       schwarz: Einbettmittel, Maßstab und gezählte Poren; weiß: Probe
* ``<bild>_histogramm.png``  Größenverteilung der gezählten Poren, wie im Viewer
* ``poren.csv``, ``kennzahlen.csv``, ``verworfen.csv``  dieselben Tabellen nur für dieses Bild

Geschrieben wird mit Semikolon und Dezimalkomma - so öffnet Excel auf einem deutschen
System die Datei ohne Nachfrage und ohne dass aus "1.5" ein Datum wird.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from . import verteilung as vert
from .filter import roundness
from .modelle import BatchResult, ImageResult, Pore

PORE_COLUMNS = [
    "bild", "label", "flaeche_px", "flaeche_um2", "aequivalentdurchmesser_px",
    "aequivalentdurchmesser_um", "umfang_px", "feret_max_px", "feret_max_um",
    "hauptachse_px", "nebenachse_px", "zirkularitaet", "rundheit_feret",
    "seitenverhaeltnis", "soliditaet",
    "exzentrizitaet", "orientierung_grad", "schwerpunkt_x_px", "schwerpunkt_y_px",
    "grauwert_mittel", "grauwert_min", "kontrast", "am_bildrand", "am_probenrand",
    "abstand_probenrand_px",
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
                    _num(pore.circularity), _num(roundness(pore)),
                    _num(pore.aspect_ratio), _num(pore.solidity),
                    _num(pore.eccentricity), _num(pore.orientation_deg),
                    _num(pore.centroid_px[0]), _num(pore.centroid_px[1]),
                    _num(pore.mean_intensity), _num(pore.min_intensity), _num(pore.contrast),
                    _bool(pore.touches_image_edge), _bool(pore.touches_specimen_edge),
                    _num(pore.specimen_edge_distance_px),
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
    "seitenverhaeltnis", "soliditaet", "rundheit_feret", "nebenachse_px",
]


def write_rejected(batch: BatchResult, path: str | Path) -> Path:
    """Eine Zeile je verworfener Pore - mit dem Grund.

    Die wichtigste der drei Tabellen beim Einstellen der Filter: sie beantwortet
    "warum fehlt diese Pore?" ohne erneutes Durchrechnen. Seitenverhältnis, Solidität,
    Rundheit und Nebenachse bleiben leer, wenn die Pore vor dem ersten Formfilter
    verworfen wurde - sie werden dann gar nicht erst gemessen (siehe messung.py).
    """
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


def write_all(batch: BatchResult, directory: str | Path) -> tuple[Path, Path, Path]:
    directory = Path(directory)
    return (
        write_pores(batch, directory / "poren.csv"),
        write_images(batch, directory / "bilder.csv"),
        write_rejected(batch, directory / "verworfen.csv"),
    )


def write_bild(ordner: str | Path, result: ImageResult) -> None:
    """Die Tabellen eines einzelnen Bildes in seinen Ordner."""
    ordner = Path(ordner)
    einzeln = BatchResult(input_dir=result.path.parent, results=[result])
    write_pores(einzeln, ordner / "poren.csv")
    write_images(einzeln, ordner / "kennzahlen.csv")
    write_rejected(einzeln, ordner / "verworfen.csv")


#: Farbe der Poren im Ergebnisbild (BGR).
ROT = (0, 0, 255)
#: Weißer Steg zwischen den beiden Bildhälften, in Pixeln.
LUECKE_PX = 10
#: Farben des Histogramms - dieselben wie im Viewer.
ROT_HEX, TINTE, AKZENT = "#ff1a1a", "#141820", "#0f6d8c"


def ergebnisbild_pfad(ziel: str | Path, bildpfad: str | Path) -> Path:
    return Path(ziel) / f"{Path(bildpfad).stem}_ergebnis.png"


def write_ergebnisbild(ziel: str | Path, bildpfad: str | Path, bild: np.ndarray,
                       labels: np.ndarray, poren_labels: list[int]) -> Path:
    """Original und Original mit roten Porenmasken nebeneinander als eine PNG.

    ``bild`` ist das Original (Graustufen oder BGR), ``labels`` das Label-Bild,
    ``poren_labels`` die Nummern der gezählten Poren.
    """
    links = cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR) if bild.ndim == 2 else bild
    rechts = links.copy()
    gezaehlt = np.zeros(int(labels.max()) + 1, dtype=bool)   # Nachschlagetabelle Label -> rot?
    gezaehlt[[label for label in poren_labels if label > 0]] = True
    rechts[gezaehlt[labels]] = ROT
    luecke = np.full((links.shape[0], LUECKE_PX, 3), 255, dtype=np.uint8)
    gesamt = np.hstack([links, luecke, rechts])

    return _png_schreiben(ergebnisbild_pfad(ziel, bildpfad), gesamt)


def write_maske(ziel: str | Path, bildpfad: str | Path, specimen: np.ndarray,
                labels: np.ndarray, poren_labels: list[int]) -> Path:
    """Schwarz-Weiß-Maske: alles außerhalb der Probe (Einbettmittel, Maßstab) und die
    gezählten Poren schwarz, die übrige Probe weiß. Verworfene Poren bleiben weiß."""
    maske = np.where(specimen, 255, 0).astype(np.uint8)
    maske[np.isin(labels, [label for label in poren_labels if label > 0])] = 0
    return _png_schreiben(Path(ziel) / f"{Path(bildpfad).stem}_maske.png", maske)


def write_histogramm(ziel: str | Path, bildpfad: str | Path, pores: list[Pore],
                     um_per_px: float | None) -> Path:
    """Größenverteilung der gezählten Poren als PNG - dasselbe Diagramm wie im Viewer."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if um_per_px:
        H = vert.flaechenhistogramm([p.area_um2 for p in pores], "µm²")
    else:
        H = vert.flaechenhistogramm([p.area_px for p in pores], "px")
    einheit = H["einheit"]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    if H["leer"]:
        ax.text(0.5, 0.5, "Keine Poren zum Auswerten", ha="center", va="center",
                transform=ax.transAxes, color="#5d6775")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        kanten, kw = np.asarray(H["kanten"]), H["kennwerte"]
        ax.bar(kanten[:-1], H["anzahlen"], width=np.diff(kanten), align="edge",
               color=ROT_HEX, edgecolor="white", linewidth=0.6, label="gezählt", zorder=2)
        if H["kurve"]:
            x, y = zip(*H["kurve"], strict=True)
            ax.plot(x, y, color=TINTE, alpha=0.72, linewidth=1.6, label="Dichte", zorder=3)
        ax.vlines(H["werte"], 0, H["maximum"] * 0.03, color=TINTE, alpha=0.38,
                  linewidth=0.7, zorder=3)
        for name, wert, stil, farbe in (("D50", kw["d50"], (0, (3, 3)), TINTE),
                                        ("D90", kw["d90"], (0, (3, 3)), TINTE),
                                        ("Mittel", kw["mittel"], "-", AKZENT)):
            ax.axvline(wert, color=farbe, linestyle=stil, linewidth=1.2, alpha=0.8, zorder=4)
            ax.annotate(name, (wert, 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points", fontsize=8, color=farbe)
        ax.set_xscale("log")
        ax.set_xlim(kanten[0], kanten[-1])
        ax.set_ylim(0, H["maximum"] * 1.08 or 1)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.grid(axis="y", color="#d5dae1", linewidth=0.8, zorder=0)
        ax.set_xlabel(f"Porenfläche A [{einheit}]")
        ax.set_ylabel("Anzahl n")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
        fig.text(0.5, 0.015,
                 f"n = {kw['n']}   Mittel {_num(kw['mittel'])}   D10 {_num(kw['d10'])}   "
                 f"D50 {_num(kw['d50'])}   D90 {_num(kw['d90'])} {einheit}",
                 ha="center", fontsize=8.5, color="#5d6775")
    ax.set_title(f"Größenverteilung der Poren - {Path(bildpfad).name}", fontsize=10)
    for seite in ("top", "right"):
        ax.spines[seite].set_visible(False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    pfad = Path(ziel) / f"{Path(bildpfad).stem}_histogramm.png"
    pfad.parent.mkdir(parents=True, exist_ok=True)
    # Über eine offene Datei statt Pfad - so wie bei cv2 keine Probleme mit Umlauten.
    with pfad.open("wb") as datei:
        fig.savefig(datei, format="png")
    plt.close(fig)
    return pfad


def _png_schreiben(pfad: Path, bild: np.ndarray) -> Path:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    # Über imencode + tofile statt imwrite: cv2.imwrite scheitert an Umlauten im Pfad.
    ok, puffer = cv2.imencode(".png", bild, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"{pfad.name} ließ sich nicht kodieren")
    puffer.tofile(str(pfad))
    return pfad
