"""Das Ergebnis des Programms gegen die Wahrheit prüfen - und die Ausgaben gegeneinander.

Zwei Arten von Prüfungen mit verschiedenem Gewicht:

**Gegen die Wahrheit** (Probenmaske, Formen, Porosität). Hier misst das Programm etwas,
und die Frage ist, wie gut. Bewertet wird mit Toleranzen (:data:`TOLERANZ`): innerhalb
OK, bis zum Doppelten WARNUNG, darüber FEHLER. Grenzfälle (Rand, Paar, winzig, Kratzer)
hängen an Einstellungen - sie werden höchstens WARNUNG, der Rand nur HINWEIS.

**Innere Stimmigkeit** (Kennwerte gegen die eigenen Pixel, Maske, Ergebnisbild, CSV und
JSON gegeneinander). Hier gibt es nichts zu schätzen - jede Abweichung ist ein
Programmfehler und damit FEHLER.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from .kennwerte import iou

OK, HINWEIS, WARNUNG, FEHLER = "OK", "HINWEIS", "WARNUNG", "FEHLER"
STUFEN = (OK, HINWEIS, WARNUNG, FEHLER)

#: Toleranz bis WARNUNG; bis zum Doppelten bleibt es WARNUNG, darüber FEHLER.
TOLERANZ = {
    "flaeche_abs_px": 4.0,          # Fläche: max(abs, rel * Fläche)
    "flaeche_rel": 0.04,
    "schwerpunkt_px": 1.0,          # Abstand der Schwerpunkte
    "bbox_px": 1.0,                 # größte Abweichung einer Kastenkante
    "durchmesser_abs_px": 0.5,      # Äquivalentdurchmesser: max(abs, rel * d)
    "durchmesser_rel": 0.03,
    "feret_abs_px": 1.0,            # Feret max: max(abs, rel * Feret)
    "feret_rel": 0.03,
    "probe_iou": 0.985,             # Probenmaske gegen Wahrheit: mindestens (Fehler ab 1 - 2*(1-x))
    "probe_flaeche_rel": 0.01,      # Probenfläche
    "porositaet_rel": 0.05,         # Porosität gegen Wahrheit
    "falschfunde": 0,               # gezählte Poren ohne Form; mehr -> WARNUNG, > 3 -> FEHLER
}


def _stufe(abweichung: float, grenze: float) -> str:
    if abweichung <= grenze:
        return OK
    return WARNUNG if abweichung <= 2 * grenze else FEHLER


def _schlimmste(stufen) -> str:
    return max(stufen, key=STUFEN.index, default=OK)


@dataclass
class Befund:
    stufe: str
    pruefung: str
    text: str


@dataclass
class Vergleich:
    groesse: str
    soll: float | str
    ist: float | str
    abweichung: str
    stufe: str


@dataclass
class FormErgebnis:
    id: int
    kategorie: str
    art: str
    erwartung: str
    status: str                   # gezaehlt | verworfen | nicht_gefunden
    label: int | None
    grund: str = ""
    iou: float | None = None
    zusammen_mit: list[int] = field(default_factory=list)
    zerteilt: list[int] = field(default_factory=list)    # Labels, wenn in mehrere zerlegt
    vergleiche: list[Vergleich] = field(default_factory=list)
    stufe: str = OK
    text: str = ""


@dataclass
class BildErgebnis:
    name: str
    stem: str
    befunde: list[Befund] = field(default_factory=list)
    formen: list[FormErgebnis] = field(default_factory=list)
    falschfunde: list[int] = field(default_factory=list)
    kennzahlen: dict = field(default_factory=dict)

    @property
    def stufe(self) -> str:
        return _schlimmste([b.stufe for b in self.befunde] + [f.stufe for f in self.formen])


# --------------------------------------------------------------------------------------
# Ein Bild
# --------------------------------------------------------------------------------------


def pruefe_bild(wahr: dict, formen_px: np.ndarray, probe_wahr: np.ndarray,
                ergebnis, ctx, bildordner: Path) -> BildErgebnis:
    stem = Path(wahr["bild"]).stem
    b = BildErgebnis(name=wahr["bild"], stem=stem)
    _probe(b, ctx, probe_wahr)
    _formen(b, wahr, formen_px, ergebnis, ctx)
    _falschfunde(b, formen_px, ergebnis, ctx)
    _porositaet(b, wahr, formen_px, probe_wahr, ergebnis)
    _stimmigkeit(b, ergebnis, ctx, bildordner)
    return b


def _probe(b: BildErgebnis, ctx, probe_wahr: np.ndarray) -> None:
    grenze = 1.0 - TOLERANZ["probe_iou"]
    wert = iou(ctx.specimen, probe_wahr)
    b.befunde.append(Befund(_stufe(1.0 - wert, grenze), "Probenmaske",
                            f"Überdeckung mit der Wahrheit (IoU) {wert:.4f}"))
    soll, ist = int(probe_wahr.sum()), int(ctx.specimen.sum())
    rel = abs(ist - soll) / soll
    b.befunde.append(Befund(_stufe(rel, TOLERANZ["probe_flaeche_rel"]), "Probenfläche",
                            f"{ist} px statt {soll} px ({(ist - soll) / soll:+.2%})"))
    harz = iou(ctx.resin, ~probe_wahr)
    b.befunde.append(Befund(_stufe(1.0 - harz, grenze), "Einbettmittel",
                            f"Überdeckung mit der Wahrheit (IoU) {harz:.4f}"))
    b.kennzahlen.update(probe_iou=wert, probe_soll=soll, probe_ist=ist, harz_iou=harz)


def _formen(b: BildErgebnis, wahr: dict, formen_px: np.ndarray, ergebnis, ctx) -> None:
    labels = ctx.labels
    gezaehlt = {p.label: p for p in ergebnis.pores}
    verworfen = {r.pore.label: r for r in ergebnis.rejected}

    # Welche Form liegt in welchem Objekt des Programms? Je Form das Objekt mit der
    # größten Überdeckung - und alle gezählten, die einen nennenswerten Teil von ihr
    # bedecken: mehr als eines heißt, das Programm hat die Form zerteilt.
    zuordnung: dict[int, int | None] = {}
    teile: dict[int, list[int]] = {}
    for f in wahr["formen"]:
        unter = labels[formen_px == f["id"]]
        unter = unter[unter > 0]
        zuordnung[f["id"]] = int(np.bincount(unter).argmax()) if unter.size else None
        mindestens = max(3, 0.05 * (f["wahr"] or {}).get("flaeche_px", 0))
        werte, anzahl = np.unique(unter, return_counts=True)
        teile[f["id"]] = [int(w) for w, n in zip(werte, anzahl) if n >= mindestens and w in gezaehlt]

    for f in wahr["formen"]:
        label = zuordnung[f["id"]]
        e = FormErgebnis(id=f["id"], kategorie=f["kategorie"], art=f["art"],
                         erwartung=f["erwartung"], status="nicht_gefunden", label=label)
        if label is not None:
            e.zusammen_mit = [i for i, l in zuordnung.items() if l == label and i != f["id"]]
            e.iou = iou(formen_px == f["id"], labels == label)
            e.zerteilt = teile[f["id"]] if len(teile[f["id"]]) > 1 else []
            if label in gezaehlt:
                e.status = "gezaehlt"
                e.vergleiche = _vergleiche(f["wahr"], gezaehlt[label])
            elif label in verworfen:
                e.status = "verworfen"
                e.grund = f"{verworfen[label].filter_name}: {verworfen[label].reason}"
        _bewerten(e, f)
        b.formen.append(e)


def _vergleiche(soll: dict, pore) -> list[Vergleich]:
    t = TOLERANZ
    a_soll, a_ist = soll["flaeche_px"], pore.area_px
    d_soll, d_ist = soll["aequivalentdurchmesser_px"], pore.equivalent_diameter_px
    f_soll, f_ist = soll["feret_max_px"], pore.feret_max_px
    sx, sy = soll["schwerpunkt_px"]
    px, py = pore.centroid_px
    kasten_soll = soll["bbox_px"]
    kasten_ist = [pore.bbox.x, pore.bbox.y, pore.bbox.w, pore.bbox.h]
    kasten = max(abs(a - b) for a, b in zip(kasten_soll, kasten_ist))
    return [
        Vergleich("Fläche [px]", a_soll, a_ist, f"{a_ist - a_soll:+.0f} ({(a_ist - a_soll) / a_soll:+.1%})",
                  _stufe(abs(a_ist - a_soll), max(t["flaeche_abs_px"], t["flaeche_rel"] * a_soll))),
        Vergleich("Schwerpunkt [px]", f"{sx:.2f}, {sy:.2f}", f"{px:.2f}, {py:.2f}",
                  f"{math.hypot(px - sx, py - sy):.2f} px",
                  _stufe(math.hypot(px - sx, py - sy), t["schwerpunkt_px"])),
        Vergleich("Bounding Box [px]", " ".join(map(str, kasten_soll)),
                  " ".join(map(str, kasten_ist)), f"max {kasten} px",
                  _stufe(kasten, t["bbox_px"])),
        Vergleich("Äquivalentdurchmesser [px]", round(d_soll, 3), round(d_ist, 3),
                  f"{d_ist - d_soll:+.2f}",
                  _stufe(abs(d_ist - d_soll), max(t["durchmesser_abs_px"], t["durchmesser_rel"] * d_soll))),
        Vergleich("Feret max [px]", round(f_soll, 3), round(f_ist, 3), f"{f_ist - f_soll:+.2f}",
                  _stufe(abs(f_ist - f_soll), max(t["feret_abs_px"], t["feret_rel"] * f_soll))),
    ]


def _bewerten(e: FormErgebnis, f: dict) -> None:
    """Stufe und Text je Form - je nach Kategorie."""
    gesehen = {"gezaehlt": "gezählt", "verworfen": f"verworfen ({e.grund})",
               "nicht_gefunden": "nicht gefunden"}[e.status]
    zusammen = f", mit Form {', '.join(map(str, e.zusammen_mit))} zu einer Pore verschmolzen" \
        if e.zusammen_mit else ""
    zerteilt = (f"in {len(e.zerteilt)} gezählte Poren zerteilt (Labels "
                f"{', '.join(map(str, e.zerteilt))}) - Größen gelten nur für das größte Stück") \
        if e.zerteilt else ""
    messung = _schlimmste(v.stufe for v in e.vergleiche)

    if e.kategorie == "sauber":
        if e.status != "gezaehlt":
            e.stufe, e.text = FEHLER, f"nicht gezählt: {gesehen}"
        elif e.zerteilt:
            e.stufe, e.text = FEHLER, zerteilt
        elif e.zusammen_mit:
            e.stufe, e.text = FEHLER, f"gezählt{zusammen}"
        else:
            e.stufe, e.text = messung, "gezählt" + ("" if messung == OK else ", Größe weicht ab")
    elif e.kategorie == "paar":
        if e.status != "gezaehlt":
            e.stufe, e.text = WARNUNG, f"Paar: {gesehen}"
        elif e.zusammen_mit:
            e.stufe, e.text = WARNUNG, f"Paar nicht getrennt{zusammen}"
        elif e.zerteilt:
            e.stufe, e.text = WARNUNG, f"Paar: Kreis {zerteilt}"
        else:
            # Die Trennlinie im Paar setzt das Programm selbst - Größen höchstens WARNUNG.
            e.stufe = OK if messung == OK else WARNUNG
            e.text = "Paar getrennt" + ("" if messung == OK else ", Größe weicht ab")
    elif e.kategorie in ("winzig", "kratzer"):
        if e.status == "gezaehlt":
            e.stufe, e.text = WARNUNG, f"{e.kategorie} wurde gezählt{zusammen}"
        else:
            e.stufe, e.text = OK, f"nicht gezählt ({gesehen})"
    else:  # rand
        e.stufe = HINWEIS
        anteil = f.get("anteil_in_probe")
        e.text = f"am Probenrand ({anteil:.0%} in der Probe): {gesehen}{zusammen}" \
            if anteil is not None else f"am Probenrand: {gesehen}{zusammen}"
        if e.status == "gezaehlt" and e.vergleiche:
            v = e.vergleiche[0]
            e.text += f", Fläche {v.ist:.0f} statt {v.soll} px"


def _falschfunde(b: BildErgebnis, formen_px: np.ndarray, ergebnis, ctx) -> None:
    form_da = formen_px > 0
    fenster = ndi.find_objects(ctx.labels)
    for p in ergebnis.pores:
        f = fenster[p.label - 1]
        if f is None or not form_da[f][ctx.labels[f] == p.label].any():
            b.falschfunde.append(p.label)
    n = len(b.falschfunde)
    flaeche = sum(p.area_px for p in ergebnis.pores if p.label in b.falschfunde)
    stufe = OK if n <= TOLERANZ["falschfunde"] else (WARNUNG if n <= 3 else FEHLER)
    b.befunde.append(Befund(stufe, "Falschfunde",
                            "keine" if not n else
                            f"{n} gezählte Pore(n) ohne Form, zusammen {flaeche:.0f} px "
                            f"(Labels {', '.join(map(str, b.falschfunde[:10]))})"))


def _porositaet(b: BildErgebnis, wahr: dict, formen_px, probe_wahr, ergebnis) -> None:
    # Wahr ist, was gezählt gehört: alles außer winzigen Formen und Kratzern, und nur
    # der Teil in der Probe (so steht es in formen_px).
    zaehlen = [f["id"] for f in wahr["formen"] if f["kategorie"] not in ("winzig", "kratzer")]
    soll = 100.0 * np.isin(formen_px, zaehlen).sum() / probe_wahr.sum()
    ist = ergebnis.porosity_pct
    b.kennzahlen.update(porositaet_soll=soll, porositaet_ist=ist)
    if ist is None:
        b.befunde.append(Befund(FEHLER, "Porosität", "keine Porosität berechnet"))
        return
    rel = abs(ist - soll) / soll if soll else abs(ist)
    b.befunde.append(Befund(_stufe(rel, TOLERANZ["porositaet_rel"]), "Porosität",
                            f"{ist:.4f} % statt {soll:.4f} % ({(ist - soll) / soll:+.1%})"
                            if soll else f"{ist:.4f} % statt 0 %"))


# --------------------------------------------------------------------------------------
# Innere Stimmigkeit - jede Abweichung ist ein Programmfehler
# --------------------------------------------------------------------------------------


def _stimmigkeit(b: BildErgebnis, ergebnis, ctx, ordner: Path) -> None:
    labels, probe = ctx.labels, ctx.specimen
    gezaehlt = [p.label for p in ergebnis.pores]
    gez_maske = np.isin(labels, gezaehlt)
    fehler: list[str] = []

    def pruefen(name: str, funktion) -> None:
        try:
            meldungen = funktion()
        except Exception as exc:  # noqa: BLE001 - eine fehlende Datei ist ein Befund
            meldungen = [f"{type(exc).__name__}: {exc}"]
        if meldungen:
            fehler.extend(meldungen)
            b.befunde.append(Befund(FEHLER, name, "; ".join(meldungen[:5])
                                    + (f" (+{len(meldungen) - 5} weitere)" if len(meldungen) > 5 else "")))
        else:
            b.befunde.append(Befund(OK, name, "stimmt"))

    def ausserhalb():
        n = int(((labels > 0) & ~probe).sum())
        return [f"{n} Porenpixel außerhalb der Probe"] if n else []

    def kennwerte_gegen_pixel():
        meldungen = []
        fenster = ndi.find_objects(labels)
        for p in [*ergebnis.pores, *(r.pore for r in ergebnis.rejected)]:
            f = fenster[p.label - 1] if p.label - 1 < len(fenster) else None
            if f is None:
                meldungen.append(f"Pore {p.label} hat keine Pixel")
                continue
            ys, xs = np.nonzero(labels[f] == p.label)
            xs, ys = xs + f[1].start, ys + f[0].start
            if p.area_px != len(xs):
                meldungen.append(f"Pore {p.label}: Fläche {p.area_px} ≠ {len(xs)} Pixel")
            if math.hypot(p.centroid_px[0] - xs.mean(), p.centroid_px[1] - ys.mean()) > 1e-6:
                meldungen.append(f"Pore {p.label}: Schwerpunkt passt nicht zu den Pixeln")
            kasten = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                      int(ys.max() - ys.min() + 1))
            if (p.bbox.x, p.bbox.y, p.bbox.w, p.bbox.h) != kasten:
                meldungen.append(f"Pore {p.label}: Bounding Box passt nicht zu den Pixeln")
            if abs(p.equivalent_diameter_px - math.sqrt(4 * len(xs) / math.pi)) > 1e-9:
                meldungen.append(f"Pore {p.label}: Äquivalentdurchmesser falsch")
        return meldungen

    def porositaet():
        soll = 100.0 * gez_maske.sum() / probe.sum()
        ist = ergebnis.porosity_pct
        return [] if ist is not None and abs(ist - soll) < 1e-9 else \
            [f"Porosität {ist} ≠ Porenpixel/Probenpixel {soll}"]

    def maske_png():
        mk = _png(ordner / f"{b.stem}_maske.png")
        soll = np.where(probe & ~gez_maske, 255, 0)
        n = int((mk != soll).sum())
        return [f"{n} Pixel falsch"] if n else []

    def ergebnisbild():
        eb = _png(ordner / f"{b.stem}_ergebnis.png")
        bild = ctx.color if ctx.color is not None else ctx.gray
        links_soll = cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR) if bild.ndim == 2 else bild
        breite = links_soll.shape[1]
        from poren.ausgabe import LUECKE_PX, ROT
        rechts_soll = links_soll.copy()
        rechts_soll[gez_maske] = ROT
        meldungen = []
        if eb.shape[1] != 2 * breite + LUECKE_PX:
            return [f"Breite {eb.shape[1]} statt {2 * breite + LUECKE_PX}"]
        if not np.array_equal(eb[:, :breite], links_soll):
            meldungen.append("linke Hälfte ist nicht das Original")
        n = int((eb[:, breite + LUECKE_PX:] != rechts_soll).any(axis=2).sum())
        if n:
            meldungen.append(f"rechte Hälfte: {n} Pixel falsch eingefärbt")
        return meldungen

    def poren_json():
        daten = json.loads((ordner / "daten" / "poren.json").read_text("utf-8"))
        meldungen = []
        if daten["porenzahl"] != len(gezaehlt) or len(daten["poren"]) != len(gezaehlt):
            meldungen.append(f"{len(daten['poren'])} Poren statt {len(gezaehlt)}")
        nach_label = {p.label: p for p in ergebnis.pores}
        for e in daten["poren"]:
            p = nach_label.get(e["label"])
            if p is None:
                meldungen.append(f"Label {e['label']} ist keine gezählte Pore")
                continue
            soll = labels == e["label"]
            ist = _fuellen(e["umriss"], labels.shape)
            n = int((ist != soll).sum())
            if n:
                meldungen.append(f"Pore {e['label']}: Umriss ergibt {n} Pixel falsch")
            if e["flaeche_px"] != p.area_px:
                meldungen.append(f"Pore {e['label']}: Fläche {e['flaeche_px']} ≠ {p.area_px}")
            if e["bbox_px"] != {"x": p.bbox.x, "y": p.bbox.y, "breite": p.bbox.w, "hoehe": p.bbox.h}:
                meldungen.append(f"Pore {e['label']}: BBox weicht ab")
        return meldungen

    def einbettmittel_json():
        daten = json.loads((ordner / "daten" / "einbettmittel.json").read_text("utf-8"))
        ist = _fuellen(daten["bereiche"], labels.shape)
        n = int((ist != ctx.resin).sum())
        meldungen = [f"Polygone ergeben {n} Pixel falsch"] if n else []
        if daten["flaeche_px"] != int(ctx.resin.sum()):
            meldungen.append("Fläche weicht ab")
        return meldungen

    def poren_csv():
        zeilen = _csv(ordner / "poren.csv")
        meldungen = []
        if len(zeilen) != len(gezaehlt):
            meldungen.append(f"{len(zeilen)} Zeilen statt {len(gezaehlt)}")
        nach_label = {p.label: p for p in ergebnis.pores}
        for z in zeilen:
            p = nach_label.get(int(z["label"]))
            if p is None:
                meldungen.append(f"Label {z['label']} ist keine gezählte Pore")
            elif not _gleich6(_zahl(z["flaeche_px"]), p.area_px):
                meldungen.append(f"Pore {z['label']}: Fläche {z['flaeche_px']} ≠ {p.area_px}")
        return meldungen

    def kennzahlen_csv():
        z = _csv(ordner / "kennzahlen.csv")
        if len(z) != 1:
            return [f"{len(z)} Zeilen statt 1"]
        meldungen = []
        if int(z[0]["porenzahl"]) != len(gezaehlt):
            meldungen.append("Porenzahl weicht ab")
        if not _gleich6(_zahl(z[0]["porositaet_pct"]), ergebnis.porosity_pct):
            meldungen.append(f"Porosität {z[0]['porositaet_pct']} ≠ {ergebnis.porosity_pct}")
        if int(z[0]["probenflaeche_px"]) != int(probe.sum()):
            meldungen.append("Probenfläche weicht ab")
        return meldungen

    pruefen("Poren nur in der Probe", ausserhalb)
    pruefen("Kennwerte gegen eigene Pixel", kennwerte_gegen_pixel)
    pruefen("Porosität = Porenpixel / Probenpixel", porositaet)
    pruefen("Maske (PNG)", maske_png)
    pruefen("Ergebnisbild (PNG)", ergebnisbild)
    pruefen("poren.json", poren_json)
    pruefen("einbettmittel.json", einbettmittel_json)
    pruefen("poren.csv", poren_csv)
    pruefen("kennzahlen.csv", kennzahlen_csv)


def pruefe_gesamt(ziel: Path, ergebnisse: list) -> list[Befund]:
    """Die Tabellen über alle Bilder im Zielordner."""
    befunde = []
    poren = sum(e.pore_count for e in ergebnisse)
    for datei, soll in (("poren.csv", poren), ("bilder.csv", len(ergebnisse)),
                        ("verworfen.csv", sum(len(e.rejected) for e in ergebnisse))):
        try:
            n = len(_csv(ziel / datei))
            befunde.append(Befund(OK if n == soll else FEHLER, datei,
                                  f"{n} Zeilen" + ("" if n == soll else f" statt {soll}")))
        except OSError as exc:
            befunde.append(Befund(FEHLER, datei, str(exc)))
    return befunde


# --------------------------------------------------------------------------------------
# Hilfen
# --------------------------------------------------------------------------------------


def _png(pfad: Path) -> np.ndarray:
    bild = cv2.imdecode(np.fromfile(str(pfad), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bild is None:
        raise OSError(f"{pfad.name} nicht lesbar")
    return bild


def _fuellen(bereiche: list[dict], form) -> np.ndarray:
    """Die Anleitung aus der JSON-Datei: je Bereich Außenumriss füllen, Löcher leeren,
    alle Bereiche vereinigen."""
    gesamt = np.zeros(form, np.uint8)
    for r in bereiche:
        einzeln = np.zeros(form, np.uint8)
        cv2.fillPoly(einzeln, [np.array(r["aussen"], np.int32)], 1)
        for loch in r["loecher"]:
            cv2.fillPoly(einzeln, [np.array(loch, np.int32)], 0)
        gesamt |= einzeln
    return gesamt.astype(bool)


def _csv(pfad: Path) -> list[dict]:
    with pfad.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _zahl(text: str) -> float | None:
    return None if text == "" else float(text.replace(",", "."))


def _gleich6(a: float | None, b: float | None) -> bool:
    """Gleich auf die 6 gültigen Stellen, mit denen die CSV schreibt."""
    if a is None or b is None:
        return a is b
    return float(f"{b:.6g}") == a
