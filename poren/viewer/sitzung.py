"""Ein Bild im Viewer: Grundlage (einmal gerechnet) und Sitzung (Korrekturen).

Warum der Viewer schnell ist:

* **Keine Bilder im JSON.** Die Seite bekommt nur Kennzahlen, Porenliste und Umrisse.
  Die Bildebenen holt sie einzeln als PNG (``Grundlage.ebene``) - erst wenn sie
  eingeschaltet werden, und jede wird nur einmal kodiert.
* **Poren zeichnet der Browser.** Die Umrisse gehen einmal als Polygone hinaus; eine
  Korrektur schickt nur die geänderte Liste zurück, kein neu kodiertes Bild.
* **Einmal rechnen, dann ablegen.** ``sichern``/``wiederherstellen`` legen das Ergebnis
  eines Bildes komprimiert ab. Beim Blättern läuft die Pipeline nicht noch einmal.
"""

from __future__ import annotations

import base64
import copy
import gzip
import math
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours

from .. import korrekturen as manuell
from .. import verteilung as vert
from ..bild import read_image
from ..einheiten import UNIT_FACTORS_UM, UNIT_SYMBOLS, format_um
from ..filter import roundness
from ..modelle import ImageResult
from ..pipeline import Pipeline, PipelineContext

#: Zielbreite der Anzeige. Kleine Bilder werden um einen ganzzahligen Faktor vergrößert
#: angezeigt (Pixel bleiben Kästchen), große bleiben wie sie sind.
ZIEL_BREITE = 900

#: Flächenfarben der Ebenen (BGR) und ihre Deckkraft (0-255).
FARBE_HARZ = ((214, 92, 124), 115)
FARBE_AUSSCHLUSS = ((208, 111, 47), 115)
FARBE_KANDIDATEN = ((23, 146, 168), 130)


# --------------------------------------------------------------------------------------
# Hilfen
# --------------------------------------------------------------------------------------


def _png(bild: np.ndarray) -> bytes:
    """PNG mit schwacher Kompression - schnell zu kodieren, lokal ist Größe egal."""
    ok, puffer = cv2.imencode(".png", bild, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG-Kodierung gescheitert")
    return puffer.tobytes()


def _flaeche(maske: np.ndarray | None, farbe: tuple, shape: tuple[int, int]) -> np.ndarray:
    """Eingefärbte Fläche als BGRA - alles außerhalb bleibt durchsichtig."""
    ebene = np.zeros((*shape, 4), dtype=np.uint8)
    if maske is not None:
        bgr, alpha = farbe
        ebene[maske] = (*bgr, alpha)
    return ebene


def _z(wert, stellen: int = 3):
    """Zahl fürs JSON: gerundet, NaN/unendlich als null (JSON.parse lehnt NaN ab)."""
    if wert is None:
        return None
    wert = float(wert)
    return round(wert, stellen) if math.isfinite(wert) else None


def _g(wert):
    """Zahl fürs JSON auf vier gültige Stellen - auch sehr kleine Flächen bleiben > 0."""
    if wert is None or not math.isfinite(float(wert)):
        return None
    return float(f"{float(wert):.4g}")


def _ohne_gerade(punkte: list[float]) -> list[float]:
    """Punkte auf einer Geraden weglassen (flache Liste x0,y0,x1,y1,...).

    Die Isolinie einer Binärmaske ist eine Treppe; auf jeder Stufe liegen mehrere Punkte
    auf derselben Geraden. Sie wegzulassen ändert den Umriss um keinen Pixel.
    """
    n = len(punkte) // 2
    if n < 3:
        return punkte
    knapp = punkte[0:2]
    for i in range(1, n - 1):
        ax, ay = punkte[2 * i] - punkte[2 * i - 2], punkte[2 * i + 1] - punkte[2 * i - 1]
        bx, by = punkte[2 * i + 2] - punkte[2 * i], punkte[2 * i + 3] - punkte[2 * i + 1]
        if abs(ax * by - ay * bx) > 1e-9:
            knapp += punkte[2 * i: 2 * i + 2]
    knapp += punkte[-2:]
    return knapp


def umrisse(labels: np.ndarray) -> dict[int, list[list[float]]]:
    """Kontur je Objekt als Polygonzüge in Bildkoordinaten: ``{label: [[x,y,x,y,...], ...]}``.

    ``find_contours`` liefert die 0,5-Isolinie, also die tatsächliche Pixelgrenze - in
    der Lupe ist das halbe Pixel genau die Frage.
    """
    ergebnis: dict[int, list[list[float]]] = {}
    for label, fenster in enumerate(ndi.find_objects(labels), start=1):
        if fenster is None:
            continue
        # Ein Pixel Rand, damit die Kontur auch am Fensterrand geschlossen ist.
        maske = np.pad((labels[fenster] == label).astype(np.float32), 1)
        x0, y0 = fenster[1].start - 1, fenster[0].start - 1
        zuege = []
        for zug in find_contours(maske, 0.5):
            flach = []
            for y, x in zug:
                flach += [round(float(x) + x0, 1), round(float(y) + y0, 1)]
            flach = _ohne_gerade(flach)
            if len(flach) >= 6:
                zuege.append(flach)
        if zuege:
            ergebnis[label] = zuege
    return ergebnis


# --------------------------------------------------------------------------------------
# Maßstab: Kontrollausschnitt und Beschriftung
# --------------------------------------------------------------------------------------


def _kontrollausschnitt(gray: np.ndarray, scale) -> tuple[str | None, dict | None]:
    """Stark vergrößerter Ausschnitt um Balken und Beschriftung, als data-URI.

    Die Endmarken zeichnet der Browser als Vektor darüber - so ist zu sehen, ob sie
    wirklich auf den Balkenenden sitzen.
    """
    balken = scale.bar_box or scale.box
    if balken is None:
        return None, None

    rand_x = max(int(0.04 * balken.w), 3)
    rand_y = max(int(2.0 * balken.h), 4)
    x0, y0 = balken.x - rand_x, balken.y - rand_y
    x1, y1 = balken.x2 + rand_x, balken.y2 + rand_y
    schrift = scale.label_box
    if schrift is not None:
        x0, y0 = min(x0, schrift.x - 3), min(y0, schrift.y - 3)
        x1, y1 = max(x1, schrift.x2 + 3), max(y1, schrift.y2 + 3)
    else:
        y1 = balken.y2 + max(int(6.0 * balken.h), 20)

    hoehe, breite = gray.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(breite, x1), min(hoehe, y1)
    ausschnitt = gray[y0:y1, x0:x1]
    faktor = max(3, min(10, int(500 / max(ausschnitt.shape[1], 1))))
    gross = cv2.resize(ausschnitt, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_NEAREST)
    uri = "data:image/png;base64," + base64.b64encode(_png(gross)).decode("ascii")
    return uri, {"x0": int(x0), "y0": int(y0), "crop_w": int(x1 - x0), "crop_h": int(y1 - y0)}


def _beschriftung(scale) -> str | None:
    """Die gelesene Beschriftung in ihrer Einheit, mit richtigem Symbol ("µm" statt "um")."""
    if scale.value_um is None or not scale.unit_text:
        return scale.label_text
    faktor = UNIT_FACTORS_UM.get(scale.unit_text)
    if faktor is None:
        return scale.label_text
    zahl = f"{scale.value_um / faktor:.6g}".replace(".", ",")
    return f"{zahl} {UNIT_SYMBOLS.get(scale.unit_text, scale.unit_text)}"


# --------------------------------------------------------------------------------------
# Grundlage: das rein automatische Ergebnis eines Bildes
# --------------------------------------------------------------------------------------


@dataclass
class Grundlage:
    """Einmal gerechnet, dann gehalten. Eine Korrektur rechnet nur noch
    :func:`darstellen` - die Pipeline läuft nicht noch einmal."""

    bildpfad: Path
    pipeline: Pipeline
    ctx: PipelineContext
    fest: dict
    #: Umrisse aller Objekte - erst gerechnet, wenn der Viewer sie braucht (teuer bei
    #: vielen Kandidaten, und im Stapellauf ohne Viewer ganz umsonst).
    _umrisse: dict[int, list] | None = None
    _pngs: dict[str, bytes] = field(default_factory=dict)

    @property
    def umrisse(self) -> dict[int, list]:
        if self._umrisse is None:
            self._umrisse = umrisse(self.ctx.labels)
        return self._umrisse

    def ebene(self, name: str) -> bytes:
        """Eine Bildebene als PNG - beim ersten Abruf kodiert, danach aus dem Speicher."""
        if name not in self._pngs:
            self._pngs[name] = _png(self._ebene_zeichnen(name))
        return self._pngs[name]

    def _ebene_zeichnen(self, name: str) -> np.ndarray:
        ctx = self.ctx
        if name == "original":
            return ctx.color if ctx.color is not None else ctx.gray
        if name == "untergrund":
            # Untergrund = Grauwert + Kontrast (so ist der Kontrast definiert).
            return np.clip(ctx.gray + ctx.contrast, 0, 255).astype(np.uint8)
        if name == "kontrast":
            # Sparsam gerechnet, damit auch riesige Bilder in den Speicher passen: der
            # Deckel (robustes Maximum) aus einer Stichprobe, die Skalierung in OpenCV.
            innen = ctx.contrast[ctx.specimen] if ctx.specimen.any() else ctx.contrast.ravel()
            schritt = max(1, innen.size // 2_000_000)
            deckel = max(float(np.percentile(innen[::schritt], 99.5)), 1e-6)
            norm = cv2.convertScaleAbs(np.maximum(ctx.contrast, 0), alpha=255.0 / deckel)
            return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        if name == "harz":
            return _flaeche(ctx.resin, FARBE_HARZ, ctx.shape)
        if name == "ausschluss":
            return _flaeche(ctx.excluded, FARBE_AUSSCHLUSS, ctx.shape)
        if name == "kandidaten":
            return _flaeche(ctx.candidates, FARBE_KANDIDATEN, ctx.shape)
        raise KeyError(f"Unbekannte Ebene {name!r}")


def _fest(bildpfad: Path, pipeline: Pipeline, ctx: PipelineContext) -> dict:
    """Alles, was sich durch eine Korrektur nicht ändert."""
    breite = ctx.width
    scale = ctx.scale
    if scale is not None:
        ausschnitt, masse = _kontrollausschnitt(ctx.gray, scale)
        bb = scale.bar_box
        massstab = {
            "um_pro_px": scale.um_per_px,
            "beschriftung": _beschriftung(scale),
            "aufloesung": format_um(scale.um_per_px).replace(".", ",") + "/px",
            "balken_px": _z(scale.bar_length_px, 2),
            "konfidenz": _z(scale.confidence),
            "balken_box": None if bb is None else {k: int(getattr(bb, k)) for k in "xywh"},
            "ausschnitt": ausschnitt,
            "ausschnitt_masse": masse,
        }
    else:
        massstab = None
    return {
        "bild": bildpfad.name,
        "erzeugt": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        # Für die Bildadressen: nach "Bild neu laden" holt der Browser neue Ebenen.
        "version": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "breite": breite, "hoehe": ctx.height,
        "zoom": max(1, min(6, round(ZIEL_BREITE / max(breite, 1)))),
        "um_pro_px": ctx.um_per_px,
        "massstab": massstab,
        "verfahren": {"probe": pipeline.config.specimen.method,
                      "poren": pipeline.config.pore.method},
        "kandidaten": int(ctx.labels.max()),
    }


def rechnen(bildpfad: Path, pipeline: Pipeline) -> Grundlage:
    """Der teure Teil: das Bild durch die Pipeline schicken, **ohne** Korrekturen."""
    ctx = pipeline.analyse(bildpfad, corrections=[])
    ctx.background = None          # steckt in gray + contrast, spart Speicher
    return Grundlage(bildpfad=bildpfad, pipeline=pipeline, ctx=ctx,
                     fest=_fest(bildpfad, pipeline, ctx))


#: Felder, die beim Ablegen wegfallen: die Bilddaten kommen wieder aus der Datei,
#: der Kontrast wird als Untergrund (float16, glatt, komprimiert gut) gespeichert.
_NICHT_ABLEGEN = ("gray", "color", "background", "contrast")


def sichern(g: Grundlage, datei: Path) -> None:
    """Die Grundlage komprimiert ablegen, damit sie später nicht neu gerechnet wird."""
    ctx = copy.copy(g.ctx)
    for name in _NICHT_ABLEGEN:
        setattr(ctx, name, None)
    untergrund = (g.ctx.gray + g.ctx.contrast).astype(np.float16)
    with gzip.open(datei, "wb", compresslevel=1) as f:
        pickle.dump({"ctx": ctx, "untergrund": untergrund, "umrisse": g._umrisse,
                     "fest": g.fest}, f, protocol=pickle.HIGHEST_PROTOCOL)


def wiederherstellen(datei: Path, bildpfad: Path, pipeline: Pipeline) -> Grundlage:
    """Gegenstück zu :func:`sichern`."""
    with gzip.open(datei, "rb") as f:
        daten = pickle.load(f)
    bild = read_image(bildpfad)
    ctx: PipelineContext = daten["ctx"]
    ctx.gray, ctx.color = bild.gray, bild.color
    ctx.contrast = daten["untergrund"].astype(np.float32) - bild.gray.astype(np.float32)
    return Grundlage(bildpfad=bildpfad, pipeline=pipeline, ctx=ctx,
                     fest=daten["fest"], _umrisse=daten["umrisse"])


# --------------------------------------------------------------------------------------
# Korrekturen anwenden und darstellen
# --------------------------------------------------------------------------------------


def korrigiert(g: Grundlage, korrekturen: list[manuell.PoreCorrection]) -> PipelineContext:
    """Die Grundlage mit angewandten Korrekturen - dieselbe Stufe wie im Stapellauf."""
    ctx = copy.copy(g.ctx)
    ctx.stages = dict(g.ctx.stages)
    ctx.warnings = list(g.ctx.warnings)
    ctx.extras = dict(g.ctx.extras)
    g.pipeline.stage_corrections(ctx, korrekturen)
    return ctx


def _flaechenhistogramm(poren: list, um_per_px: float | None) -> dict:
    """Häufigkeitsverteilung der Porenfläche der gezählten Poren - aus den ungerundeten
    Flächen wie im PNG-Histogramm. Erst das Ergebnis wird fürs JSON gerundet, und zwar
    auf gültige Stellen: bei hoher Vergrößerung wären kleine Poren auf drei
    Nachkommastellen in µm² sonst 0 und fielen aus dem Diagramm."""
    if um_per_px:
        H = vert.flaechenhistogramm([p.area_um2 for p in poren], "µm²")
    else:
        H = vert.flaechenhistogramm([p.area_px for p in poren], "px")
    if H["leer"]:
        return H
    H["kennwerte"] = {k: (_g(v) if k != "n" else v) for k, v in H["kennwerte"].items()}
    H["kurve"] = [[_g(x), _g(y)] for x, y in H["kurve"]]
    H["werte"] = [_g(w) for w in H["werte"]]
    return H


def darstellen(g: Grundlage, korrekturen: list[manuell.PoreCorrection]) -> dict:
    """Alles, was sich durch eine Korrektur ändert - als JSON-taugliches dict."""
    ctx = korrigiert(g, korrekturen)
    ergebnis = Pipeline.to_result(ctx)

    bericht = ctx.extras.get("korrekturen", {})
    entfernt = set(bericht.get("entfernt", []))
    aufgenommen = set(bericht.get("aufgenommen", []))
    gezeichnet = set(bericht.get("gezeichnet", []))

    def eintrag(p, status: str, grund: str | None) -> dict:
        return {
            "label": p.label, "status": status, "grund": grund,
            "manuell": ("entfernt" if p.label in entfernt
                        else "aufgenommen" if p.label in aufgenommen
                        else "gezeichnet" if p.label in gezeichnet else None),
            "box": [p.bbox.x, p.bbox.y, p.bbox.w, p.bbox.h],
            "flaeche_px": _z(p.area_px), "flaeche_um2": _z(p.area_um2),
            "d_px": _z(p.equivalent_diameter_px), "d_um": _z(p.equivalent_diameter_um),
            # Dieselbe Rundheit wie im Filter "roundness" und in den Verwerfungsgründen.
            "rundheit": _z(roundness(p)), "solidity": _z(p.solidity),
            "kontrast": _z(p.contrast),
            "randbild": p.touches_image_edge, "randprobe": p.touches_specimen_edge,
        }

    poren = [eintrag(p, "behalten", None) for p in ctx.pores]
    poren += [eintrag(r.pore, "verworfen", f"{r.filter_name} - {r.reason}") for r in ctx.rejected]
    poren.sort(key=lambda d: -(d["flaeche_px"] or 0))

    # Eingezeichnete Poren gibt es in der Grundlage nicht - ihre Umrisse kommen mit.
    neue_umrisse = {}
    if gezeichnet:
        neue_umrisse = umrisse(np.where(np.isin(ctx.labels, list(gezeichnet)), ctx.labels, 0))

    groesste = ergebnis.largest_pore
    return {
        "stufen": dict(ctx.stages),
        "warnungen": list(ctx.warnings),
        "kennzahlen": {
            "kandidaten": g.fest["kandidaten"],
            "behalten": len(ctx.pores),
            "verworfen": len(ctx.rejected),
            "porositaet": _z(ergebnis.porosity_pct, 4),
            "porositaet_ohne_rand": _z(ergebnis.porosity_pct_excl_edge, 4),
            "probe_px": ergebnis.specimen_area_px,
            "probe_mm2": _z(ergebnis.specimen_area_mm2, 4),
            "dichte": _z(ergebnis.pore_density_per_mm2),
            "groesste_um": None if groesste is None else _z(groesste.equivalent_diameter_um),
            "groesste_px": None if groesste is None else _z(groesste.equivalent_diameter_px),
        },
        "korrekturen": {
            "entfernt": len(entfernt), "aufgenommen": len(aufgenommen),
            "gezeichnet": len(gezeichnet),
            "ohne_treffer": len(bericht.get("ohne_treffer", [])),
        },
        "poren": poren,
        "umrisse_neu": neue_umrisse,
        "histogramm": _flaechenhistogramm(ctx.pores, ctx.um_per_px),
    }


class Sitzung:
    """Ein Bild in Bearbeitung: Grundlage, aktuelle Korrekturen, Verlauf, Ablage.

    Jede Änderung wird sofort gespeichert. ``bei_aenderung`` wird danach aufgerufen -
    so hält der Stapel seine Ergebnisdateien aktuell.
    """

    def __init__(self, grundlage: Grundlage, ablage: manuell.CorrectionStore,
                 bei_aenderung: Callable[[Sitzung], None] | None = None) -> None:
        self.g = grundlage
        self.ablage = ablage
        self.korrekturen = self.ablage.load(grundlage.bildpfad)
        self._verlauf: list[list[manuell.PoreCorrection]] = []
        self._bei_aenderung = bei_aenderung

    def ansicht(self) -> dict:
        """Die vollständigen Daten für die Seite: fester und veränderlicher Teil."""
        return {**self.g.fest, **darstellen(self.g, self.korrekturen), "umrisse": self.g.umrisse}

    def ergebnis(self) -> tuple[ImageResult, PipelineContext]:
        """Ergebnis mit den aktuellen Korrekturen - so, wie es exportiert wird."""
        ctx = korrigiert(self.g, self.korrekturen)
        return Pipeline.to_result(ctx), ctx

    def plus(self, label: int) -> dict:
        """Werkzeug "+": die Pore zählen, auch wenn ein Filter sie verworfen hat."""
        return self._zaehlen(label, True)

    def minus(self, label: int) -> dict:
        """Werkzeug "-": die Pore nicht zählen. Eine eingezeichnete wird gelöscht."""
        ctx = korrigiert(self.g, self.korrekturen)
        if label in ctx.extras.get("korrekturen_gezeichnet", {}):
            return self._setzen(self._ohne_zeichnung(ctx, label))
        return self._zaehlen(label, False)

    def _ohne_zeichnung(self, ctx: PipelineContext, label: int) -> list[manuell.PoreCorrection]:
        """Die Korrekturen ohne die eingezeichnete Pore ``label`` - und ohne, dass an ihrer
        Stelle wieder etwas gezählt wird.

        Eine Zeichnung schluckt die erkannten Poren, die sie überdeckt (siehe
        ``korrekturen.apply``). Fiele nur die Zeichnung weg, kämen diese zurück, und "-"
        hinterließe an derselben Stelle wieder eine gezählte Pore. Deshalb werden sie mit
        entfernt - eine früher eingezeichnete gelöscht, eine erkannte verworfen.
        """
        flaeche = ctx.labels == label
        index = ctx.extras["korrekturen_gezeichnet"][label]
        neu = self.korrekturen[:index] + self.korrekturen[index + 1:]
        # Jede Runde nimmt eine zurückgekehrte Pore weg. Lässt sich keine mehr wegnehmen,
        # ist Schluss - sonst liefe die Schleife auf derselben Pore im Kreis.
        basis = self.g.ctx
        while True:
            danach = korrigiert(self.g, neu)
            unter = set(np.unique(danach.labels[flaeche]).tolist()) - {0}
            gezeichnet = danach.extras.get("korrekturen_gezeichnet", {})
            for p in danach.pores:
                if p.label not in unter:
                    continue
                if p.label in gezeichnet:
                    i = gezeichnet[p.label]
                    naechste = neu[:i] + neu[i + 1:]
                else:
                    punkt = manuell.interior_point(danach.labels, p.label)
                    if punkt is None:
                        continue
                    naechste = manuell.set_counted(basis.pores, basis.rejected, basis.labels,
                                                   neu, *punkt, counted=False)
                if naechste != neu:
                    neu = naechste
                    break
            else:
                return neu

    def zeichnen(self, punkte: list[list[float]]) -> dict:
        """Werkzeug "Zeichnen": der Umriss wird als neue Pore gezählt."""
        try:
            neu = manuell.PoreCorrection.drawn(punkte)
        except ValueError:
            return darstellen(self.g, self.korrekturen)
        return self._setzen([*self.korrekturen, neu])

    def bereich(self, punkte: list[list[float]]) -> dict:
        """Werkzeug "Bereich": alle gezählten Poren, die ganz im Umriss liegen, entfernen."""
        try:
            neu = manuell.PoreCorrection.area(punkte)
        except ValueError:
            return darstellen(self.g, self.korrekturen)
        return self._setzen([*self.korrekturen, neu])

    def rueckgaengig(self) -> dict:
        if not self._verlauf:
            return darstellen(self.g, self.korrekturen)
        self.korrekturen = self._verlauf.pop()
        self._gespeichert()
        return darstellen(self.g, self.korrekturen)

    def zuruecksetzen(self) -> dict:
        return self._setzen([])

    def _zaehlen(self, label: int, gezaehlt: bool) -> dict:
        """Gespeichert wird ein Punkt tief im Inneren der Pore, nicht das Label - das
        gilt nur für diesen einen Rechenlauf."""
        basis = self.g.ctx
        punkt = manuell.interior_point(basis.labels, label)
        if punkt is None:
            return darstellen(self.g, self.korrekturen)
        neu = manuell.set_counted(basis.pores, basis.rejected, basis.labels,
                                  self.korrekturen, *punkt, counted=gezaehlt)
        return self._setzen(neu)

    def _setzen(self, neu: list[manuell.PoreCorrection]) -> dict:
        if neu != self.korrekturen:
            self._verlauf.append(self.korrekturen)
            self.korrekturen = neu
            self._gespeichert()
        return darstellen(self.g, self.korrekturen)

    def _gespeichert(self) -> None:
        self.ablage.save(self.g.bildpfad, self.korrekturen)
        if self._bei_aenderung is not None:
            self._bei_aenderung(self)
