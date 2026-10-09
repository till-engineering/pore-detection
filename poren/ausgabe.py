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
* ``daten/poren.json``, ``daten/einbettmittel.json``  für die maschinelle Weiterverarbeitung,
  mit Umrissen (:func:`write_daten`)

Geschrieben wird mit Semikolon und Dezimalkomma - so öffnet Excel auf einem deutschen
System die Datei ohne Nachfrage und ohne dass aus "1.5" ein Datum wird. Die JSON-Dateien
dagegen haben Punkt als Dezimaltrenner, wie JSON es verlangt.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import verteilung as vert
from .filter import roundness
from .modelle import BatchResult, ImageResult, Pore

if TYPE_CHECKING:
    from .pipeline import PipelineContext

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


# --------------------------------------------------------------------------------------
# JSON für die maschinelle Weiterverarbeitung
# --------------------------------------------------------------------------------------

#: Unterordner im Ordner eines Bildes.
DATEN_ORDNER = "daten"
#: Steigt, wenn sich der Aufbau der JSON-Dateien so ändert, dass Leser angepasst werden müssen.
DATEN_FORMAT = 1

_KOORDINATEN = (
    "Pixel, Ursprung oben links, x nach rechts, y nach unten. Umrisse sind geschlossene "
    "Polygone durch die Mitten der Randpixel ('aussen' durch die der Fläche, jedes Loch "
    "durch die des Lochs). Pixelmaske eines Bereichs: cv2.fillPoly(aussen, 1), danach "
    "cv2.fillPoly(loch, 0) für jedes Loch; die Gesamtmaske ist die Vereinigung aller "
    "Bereiche. Werte in µm = Pixel * um_pro_px."
)


def write_daten(ordner: str | Path, ctx: PipelineContext, result: ImageResult,
                einbettmittel: bool = True) -> Path:
    """``daten/poren.json`` und ``daten/einbettmittel.json`` in den Ordner eines Bildes.

    ``ctx`` liefert das Label-Bild und die Einbettmittelmaske - mit angewandten
    Korrekturen, also so, wie das Bild im Viewer steht. Das Einbettmittel ändert sich
    durch eine Korrektur nicht; ``einbettmittel=False`` spart dann das Nachschreiben.
    """
    ziel = Path(ordner) / DATEN_ORDNER
    ziel.mkdir(parents=True, exist_ok=True)
    kopf = _kopf(result)

    # Nur die gezählten Poren - verworfene stehen in verworfen.csv.
    korrektur = ctx.extras.get("korrekturen", {})
    manuell = {label: art for art, schluessel in (("aufgenommen", "aufgenommen"),
                                                  ("eingezeichnet", "gezeichnet"))
               for label in korrektur.get(schluessel, [])}
    gezaehlt = [p.label for p in result.pores]
    umrisse = _umrisse_je_label(np.where(np.isin(ctx.labels, gezaehlt), ctx.labels, 0))
    _json_schreiben(ziel / "poren.json", {
        **kopf,
        "porenzahl": len(result.pores),
        "porositaet_pct": _wert(result.porosity_pct),
        "poren": [_pore_json(p, manuell, umrisse) for p in result.pores],
    })

    if einbettmittel:
        maske = ctx.resin if ctx.resin is not None else np.zeros(ctx.shape, dtype=bool)
        flaeche = int(maske.sum())
        um = result.scale.um_per_px if result.scale else None
        _json_schreiben(ziel / "einbettmittel.json", {
            **kopf,
            "flaeche_px": flaeche,
            "flaeche_um2": _wert(flaeche * um * um if um else None),
            "anteil_bild": _wert(flaeche / (result.width * result.height)),
            "probenflaeche_px": result.specimen_area_px,
            "bereiche": _bereiche(maske),
        })
    return ziel


def _kopf(result: ImageResult) -> dict:
    """Was jede Datei für sich lesbar macht: Bild, Größe, Maßstab, Koordinatensystem."""
    scale = result.scale
    return {
        "format": DATEN_FORMAT,
        "bild": result.name,
        "erzeugt": datetime.now().isoformat(timespec="seconds"),
        "breite_px": result.width,
        "hoehe_px": result.height,
        "um_pro_px": _wert(scale.um_per_px if scale else None),
        "massstab_text": scale.label_text if scale else None,
        "koordinaten": _KOORDINATEN,
    }


def _pore_json(pore: Pore, manuell: dict[int, str], umrisse: dict[int, list[dict]]) -> dict:
    um = pore.um_per_px
    x, y = pore.centroid_px
    return {
        "label": pore.label,
        # "aufgenommen" (von Hand gezählt), "eingezeichnet" oder null (automatisch).
        "manuell": manuell.get(pore.label),
        "schwerpunkt_px": [_wert(x), _wert(y)],
        "schwerpunkt_um": [_wert(x * um), _wert(y * um)] if um else None,
        "bbox_px": {"x": pore.bbox.x, "y": pore.bbox.y, "breite": pore.bbox.w,
                    "hoehe": pore.bbox.h},
        "flaeche_px": _wert(pore.area_px),
        "flaeche_um2": _wert(pore.area_um2),
        "aequivalentdurchmesser_px": _wert(pore.equivalent_diameter_px),
        "aequivalentdurchmesser_um": _wert(pore.equivalent_diameter_um),
        "umfang_px": _wert(pore.perimeter_px),
        "feret_max_px": _wert(pore.feret_max_px),
        "feret_max_um": _wert(pore.feret_max_um),
        "hauptachse_px": _wert(pore.major_axis_px),
        "nebenachse_px": _wert(pore.minor_axis_px),
        "zirkularitaet": _wert(pore.circularity),
        "rundheit_feret": _wert(roundness(pore)),
        "seitenverhaeltnis": _wert(pore.aspect_ratio),
        "soliditaet": _wert(pore.solidity),
        "exzentrizitaet": _wert(pore.eccentricity),
        "orientierung_grad": _wert(pore.orientation_deg),
        "grauwert_mittel": _wert(pore.mean_intensity),
        "grauwert_min": _wert(pore.min_intensity),
        "kontrast": _wert(pore.contrast),
        "am_bildrand": pore.touches_image_edge,
        "am_probenrand": pore.touches_specimen_edge,
        "abstand_probenrand_px": _wert(pore.specimen_edge_distance_px),
        "umriss": umrisse.get(pore.label, []),
    }


def _umrisse_je_label(labels: np.ndarray) -> dict[int, list[dict]]:
    """Umrisse aller Objekte des Label-Bildes, je Objekt nur im eigenen Ausschnitt
    gesucht - das ganze Bild je Pore abzusuchen wäre bei tausend Poren zu langsam."""
    ergebnis: dict[int, list[dict]] = {}
    for label, fenster in enumerate(ndi.find_objects(labels), start=1):
        if fenster is None:
            continue
        maske = labels[fenster] == label
        ergebnis[label] = _bereiche(maske, versatz=(fenster[1].start, fenster[0].start))
    return ergebnis


def _bereiche(maske: np.ndarray, versatz: tuple[int, int] = (0, 0)) -> list[dict]:
    """Zusammenhängende Flächen einer Maske als ``{"aussen": [[x, y], ...],
    "loecher": [[[x, y], ...], ...]}``.

    Beide Umrisse laufen durch die Mitten **ihrer eigenen** Randpixel - der äußere durch
    die Randpixel der Fläche, ein Loch durch die Randpixel des Lochs. So ergibt
    ``fillPoly(aussen, 1)`` und danach ``fillPoly(loch, 0)`` genau die Fläche. Eine Insel
    in einem Loch ist eine eigene Fläche; die Maske ist die Vereinigung aller Flächen.
    """
    if not maske.any():
        return []
    # Fläche 8er-, Hintergrund 4er-Nachbarschaft - so trennt findContours auch.
    flaechen, _ = ndi.label(maske, structure=np.ones((3, 3), dtype=bool))
    loecher, _ = ndi.label(~maske)
    hoehe, breite = maske.shape

    bereiche: dict[int, dict] = {}
    for nummer, fenster in enumerate(ndi.find_objects(flaechen), start=1):
        if fenster is not None:
            bereiche[nummer] = {"aussen": _umriss(flaechen[fenster] == nummer, fenster, versatz),
                                "loecher": []}

    for nummer, fenster in enumerate(ndi.find_objects(loecher), start=1):
        if fenster is None:
            continue
        zeilen, spalten = fenster
        # Hintergrund am Rand ist kein Loch - er ist nach außen offen.
        if (zeilen.start == 0 or spalten.start == 0
                or zeilen.stop == hoehe or spalten.stop == breite):
            continue
        # Zu welcher Fläche das Loch gehört: die Fläche, die es umgibt - ein Pixel
        # über seinen Ausschnitt hinaus liegt sie ringsum.
        gross = (slice(zeilen.start - 1, zeilen.stop + 1),
                 slice(spalten.start - 1, spalten.stop + 1))
        loch = loecher[gross] == nummer
        ring = ndi.binary_dilation(loch, structure=np.ones((3, 3), dtype=bool)) & ~loch
        umgebend = flaechen[gross][ring]
        umgebend = umgebend[umgebend > 0]
        if len(umgebend) == 0:
            continue
        besitzer = int(np.bincount(umgebend).argmax())
        bereiche[besitzer]["loecher"].append(_umriss(loecher[fenster] == nummer, fenster, versatz))
    return list(bereiche.values())


def _umriss(maske: np.ndarray, fenster: tuple[slice, slice],
            versatz: tuple[int, int]) -> list[list[int]]:
    """Der äußere Umriss einer zusammenhängenden Maske im Ausschnitt ``fenster``."""
    konturen, _ = cv2.findContours(
        maske.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        offset=(fenster[1].start + versatz[0], fenster[0].start + versatz[1]))
    return max(konturen, key=len).reshape(-1, 2).tolist()


def _wert(value: float | None) -> float | None:
    """Zahl für JSON: gerundet auf 6 gültige Stellen, ``None`` statt NaN/unendlich."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def _json_schreiben(pfad: Path, daten: dict) -> Path:
    # Erst in eine Nachbardatei, dann umbenennen: wer die Datei gerade liest, bekommt
    # nie eine halb geschriebene.
    zwischen = pfad.with_suffix(".json.tmp")
    zwischen.write_text(json.dumps(daten, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    zwischen.replace(pfad)
    return pfad
