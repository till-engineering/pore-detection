"""Erzeugt eine eigenstaendige HTML-Ansicht eines einzelnen Schliffbildes.

Das ist der Bildschirm, den ``docs/ARCHITECTURE.md`` als den wichtigsten der spaeteren
GUI bezeichnet: **ein** Bild mit umschaltbaren Ebenen. Nicht der Stapellauf, sondern die
Frage "was hat das Verfahren hier eigentlich getan, und warum".

Die Ausgabe ist eine einzige HTML-Datei mit allen Bildebenen als eingebettete PNGs. Sie
braucht keinen Server und kein Netz - Doppelklick genuegt. Das ist Absicht: solange die
FastAPI-Oberflaeche (P5) nicht steht, soll die Sichtpruefung trotzdem moeglich sein, und
eine einzelne Datei laesst sich weitergeben.

Ausgewertet wird das Bild aus ``test/GUI_Test`` - ein anderes probiert man aus, indem
man es dort hineinlegt. Liegen mehrere darin, gewinnt das zuletzt geaenderte; der Name
steht nirgends im Code.

Aufruf::

    .venv\\Scripts\\python.exe scripts/gui_vorschau.py
    .venv\\Scripts\\python.exe scripts/gui_vorschau.py --ordner test/real_pores
    .venv\\Scripts\\python.exe scripts/gui_vorschau.py --bild test/GUI_Test/gas-porosity.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours

WURZEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WURZEL / "src"))

from poredet.analysis import spatial  # noqa: E402
from poredet.config.schema import AppConfig  # noqa: E402
from poredet.core.pipeline import PoreDetectionPipeline  # noqa: E402
from poredet.io.image_reader import find_images  # noqa: E402
from poredet.measurement import distributions as vert  # noqa: E402

#: Zielbreite der Anzeige in Pixeln. Kleine Vorlagen werden dafuer vergroessert -
#: ohne das sind Porenkonturen nicht zu beurteilen -, grosse bleiben wie sie sind.
#: Ein fester Faktor waere fuer beides falsch: er macht ein 3000-px-Bild zu einer
#: Datei von hunderten Megabyte und ein 270-px-Bild trotzdem nicht lesbar.
ZIEL_BREITE = 900


def zoom_fuer(breite: int) -> int:
    """Ganzzahliger Vergroesserungsfaktor. Ganzzahlig, damit Pixel Kaestchen bleiben."""
    return max(1, min(6, round(ZIEL_BREITE / max(breite, 1))))


#: Obergrenze der Kantenliste: Steg kleiner als dieses Vielfache des groesseren
#: Durchmessers. Liegt ueber allen gebraeuchlichen Zusammenfassungsregeln, damit das
#: Kriterium spaeter frei gewaehlt werden kann, ohne neu rechnen zu muessen.
SCHRANKE = 3.0


# --------------------------------------------------------------------------------------
# Ebenen zeichnen
# --------------------------------------------------------------------------------------


def _gross(bild: np.ndarray, zoom: int) -> np.ndarray:
    """Auf Anzeigegroesse bringen - Nearest, damit Pixelgrenzen sichtbar bleiben."""
    if zoom == 1:
        return bild
    return cv2.resize(bild, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def _png(bild: np.ndarray) -> str:
    """Als data-URI. Eine lokale HTML darf keine Nachbardateien laden (file-Sperre)."""
    ok, puffer = cv2.imencode(".png", bild)
    if not ok:
        raise RuntimeError("PNG-Kodierung gescheitert")
    return "data:image/png;base64," + base64.b64encode(puffer.tobytes()).decode("ascii")


def _flaeche(maske: np.ndarray, farbe_bgr: tuple[int, int, int], alpha: int,
             zoom: int) -> np.ndarray:
    """Eingefaerbte Flaeche als BGRA - alles ausserhalb bleibt durchsichtig."""
    gross = _gross(maske.astype(np.uint8), zoom).astype(bool)
    hoehe, breite = gross.shape
    ebene = np.zeros((hoehe, breite, 4), dtype=np.uint8)
    ebene[gross] = (*farbe_bgr, alpha)
    return ebene


def _konturen(
    labels: np.ndarray, farbe_bgr: tuple[int, int, int], zoom: int,
    dicke: int = 1, fuellung: int = 0,
) -> np.ndarray:
    """Umrisse je Objekt als BGRA. Gezeichnet wird auf dem vergroesserten Gitter."""
    gross = _gross(labels.astype(np.int32), zoom)
    hoehe, breite = gross.shape
    ebene = np.zeros((hoehe, breite, 4), dtype=np.uint8)

    if fuellung:
        ebene[gross > 0] = (*farbe_bgr, fuellung)

    for wert in np.unique(gross):
        if wert == 0:
            continue
        einzeln = (gross == wert).astype(np.uint8)
        umrisse, _ = cv2.findContours(einzeln, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ebene, umrisse, -1, (*farbe_bgr, 255), dicke)
    return ebene


def _heatmap(werte: np.ndarray, deckel: float, zoom: int) -> np.ndarray:
    """Kontrastbild als Falschfarben. Der Deckel ist ein robustes Maximum."""
    norm = np.clip(werte / max(deckel, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap(_gross((norm * 255).astype(np.uint8), zoom), cv2.COLORMAP_INFERNO)


# --------------------------------------------------------------------------------------
# Porenkonturen als Polygonzug
# --------------------------------------------------------------------------------------


def _ohne_gerade(punkte: list[list[float]]) -> list[list[float]]:
    """Punkte auf einer Geraden weglassen.

    Die Isolinie einer Binaermaske ist eine Treppe, und auf jeder Stufe liegen mehrere
    Punkte hintereinander auf derselben Geraden. Sie wegzulassen aendert den Zug um
    keinen Pixel und halbiert die Datei.
    """
    if len(punkte) < 3:
        return punkte
    knapp = [punkte[0]]
    for vorher, hier, nachher in zip(punkte, punkte[1:], punkte[2:], strict=False):
        ax, ay = hier[0] - vorher[0], hier[1] - vorher[1]
        bx, by = nachher[0] - hier[0], nachher[1] - hier[1]
        if abs(ax * by - ay * bx) > 1e-9:          # Kreuzprodukt: nicht kollinear
            knapp.append(hier)
    knapp.append(punkte[-1])
    return knapp


def _umrisse(labels: np.ndarray) -> dict[int, list[list[list[float]]]]:
    """Die Aussenkontur je Objekt als Polygonzug in Bildkoordinaten.

    Als Vektor und nicht als Pixelebene, aus demselben Grund wie bei der Messstrecke:
    die Lupe vergroessert eine gerasterte Ein-Pixel-Linie mit. Bei zwanzigfacher
    Vergroesserung ist die Kontur dann ein zwanzig Pixel breites Band und verdeckt
    genau die Kante, die beurteilt werden soll. Als Vektor bleibt sie in jeder
    Zoomstufe haarduenn.

    ``find_contours`` liefert die 0,5-Isolinie, also die tatsaechliche Pixelgrenze -
    nicht die Mittelpunkte der Randpixel, die ``cv2.findContours`` zurueckgibt. Das
    halbe Pixel Unterschied ist in der Gesamtansicht nichts und in der Lupe die Frage.
    """
    ergebnis: dict[int, list[list[list[float]]]] = {}
    for label, fenster in enumerate(ndi.find_objects(labels), start=1):
        if fenster is None:
            continue
        # Ein Pixel Rand, damit ein Objekt, das das Fenster ausfuellt, eine
        # geschlossene Kontur bekommt statt einer offenen an der Fensterkante.
        maske = np.pad((labels[fenster] == label).astype(np.float32), 1)
        x0, y0 = fenster[1].start - 1, fenster[0].start - 1
        zuege = []
        for zug in find_contours(maske, 0.5):
            punkte = _ohne_gerade(
                [[round(float(x) + x0, 1), round(float(y) + y0, 1)] for y, x in zug])
            if len(punkte) >= 3:
                zuege.append(punkte)
        if zuege:
            ergebnis[label] = zuege
    return ergebnis


# --------------------------------------------------------------------------------------
# Massstab: die vermessene Strecke, nicht die ausgeschlossene Flaeche
# --------------------------------------------------------------------------------------

def _balkenkasten(balken) -> dict | None:
    """Die Balkengeometrie fuer die Anzeige.

    Die Messstrecke wird im Browser als SVG gezeichnet, nicht hier als Pixelbild: eine
    ein Pixel duenne Rasterlinie faellt beim Herunterskalieren der Anzeige schlicht
    heraus. Als Vektor bleibt sie in jeder Zoomstufe gleich dick und sichtbar.
    """
    if balken is None:
        return None
    return {"x": balken.x, "y": balken.y, "w": balken.w, "h": balken.h}


def _kontrollausschnitt(gray: np.ndarray, scale) -> tuple[str, dict] | tuple[None, None]:
    """Stark vergroesserter Ausschnitt des Overlays - **ohne** eingezeichnete Marken.

    Der Zweck ist die Sichtpruefung der einen Frage, die keine Kennzahl beantwortet:
    liegen die Endmarken wirklich auf den Balkenenden? Eine um zwei Pixel zu kurz
    gemessene Strecke ist rechnerisch unauffaellig und im Ergebnis ein Faktor.

    Die Marken selbst zeichnet der Browser als Vektor darueber. Ins Bild gerechnet
    waeren sie wieder nur wenige Pixel dick und fielen beim Herunterskalieren auf die
    Spaltenbreite teilweise heraus - genau die Marke, auf die es ankommt.

    Zurueck kommt neben dem Bild die Lage des Ausschnitts, damit die Anzeige in
    Bildkoordinaten rechnen kann.
    """
    kasten = scale.box or scale.bar_box
    if kasten is None:
        return None, None

    hoehe, breite = gray.shape
    rand_x = max(int(0.35 * kasten.w), 8)
    rand_y = max(int(0.8 * kasten.h), 8)
    x0 = max(0, kasten.x - rand_x)
    y0 = max(0, kasten.y - rand_y)
    x1 = min(breite, kasten.x2 + rand_x)
    y1 = min(hoehe, kasten.y2 + rand_y)

    ausschnitt = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    # So weit vergroessern, dass einzelne Pixel als Kaestchen sichtbar werden - nur dann
    # laesst sich die Kante des Balkens ueberhaupt beurteilen.
    faktor = max(3, min(10, int(500 / max(ausschnitt.shape[1], 1))))
    gross = cv2.resize(ausschnitt, None, fx=faktor, fy=faktor, interpolation=cv2.INTER_NEAREST)

    return _png(gross), {
        "faktor": faktor,
        "breite": gross.shape[1], "hoehe": gross.shape[0],
        # Lage und Groesse in Bildkoordinaten - die viewBox des Vektor-Overlays.
        "x0": x0, "y0": y0, "crop_w": x1 - x0, "crop_h": y1 - y0,
    }


# --------------------------------------------------------------------------------------
# Auswertungen
# --------------------------------------------------------------------------------------


def _flaechenhistogramm(poren: list[dict], um_per_px: float | None) -> dict:
    """Haeufigkeitsverteilung der Porenflaeche, gezaehlte und verworfene getrennt.

    Beide Gruppen bekommen **dieselben** Klassengrenzen, sonst liessen sich die Balken
    nicht uebereinanderlegen. Die Grenzen entstehen ueber alle Objekte zusammen - sonst
    verschoebe sich die Achse, sobald ein Filter anders eingestellt wird.
    """
    feld = "flaeche_um2" if um_per_px else "flaeche_px"
    einheit = "µm²" if um_per_px else "px"

    alle = [p[feld] for p in poren if p[feld]]
    behalten = [p[feld] for p in poren if p["status"] == "behalten" and p[feld]]
    verworfen = [p[feld] for p in poren if p["status"] == "verworfen" and p[feld]]
    if not alle:
        return {"leer": True, "einheit": einheit}

    # Klassenzahl aus den Daten statt fest: eine feste Zahl ist bei wenigen Poren zu
    # fein und bei vielen zu grob.
    grenzen = vert.kanten(alle, klassen=vert.klassenzahl(alle), logarithmisch=True)
    v_behalten = vert.verteilen(behalten, grenzen)
    v_verworfen = vert.verteilen(verworfen, grenzen)

    return {
        "leer": False,
        "einheit": einheit,
        "logarithmisch": True,
        "kanten": list(grenzen),
        "behalten": list(v_behalten.anzahlen),
        "verworfen": list(v_verworfen.anzahlen),
        "maximum": max(a + b for a, b in
                       zip(v_behalten.anzahlen, v_verworfen.anzahlen, strict=True)),
        # Die Quantile beziehen sich auf die gezaehlten Poren: die verworfenen sind
        # per Entscheidung keine, und eine Kennzahl ueber beide waere sinnlos.
        "kennwerte": vert.kennwerte(behalten),
        # Klassenfreie Zusaetze: die Kurve zeigt die Verteilung ohne Klassenraster,
        # die Einzelwerte zeigen, wo wirklich Poren liegen.
        "kurve": [[round(x, 3), round(y, 3)] for x, y in vert.dichtekurve(behalten, grenzen)],
        "werte": sorted(round(w, 3) for w in behalten),
    }


def _kantenliste(
    labels: np.ndarray, um_per_px: float | None, durchmesser_px: dict[int, float]
) -> dict:
    """Alle Nachbarabstaende als Kantenliste - die Grundlage fuer ein spaeteres Kriterium.

    Je Pore nur den naechsten Nachbarn zu fuehren reicht dafuer nicht: liegen drei
    Poren dicht beieinander, fehlt immer eine der drei Verbindungen. Deshalb hier
    jedes Paar unterhalb einer grosszuegigen Schranke, mit beiden Bezugsgroessen
    (kleinerer und groesserer Durchmesser), damit die Wahl des Kriteriums offen bleibt.

    Paare, deren Verbindung durch eine dritte Pore laeuft, fallen heraus: zwischen
    ihnen steht kein durchgehender Steg, und ihre Nachbarschaft ist ohnehin ueber die
    Pore dazwischen vermittelt.
    """
    kanten = spatial.alle_abstaende(labels, durchmesser_px, max_faktor=SCHRANKE)
    # Nur zum Ausweisen, wie viele Paare die Sichtpruefung verworfen hat.
    ohne_pruefung = spatial.alle_abstaende(
        labels, durchmesser_px, max_faktor=SCHRANKE, nur_freie_sicht=False)
    faktor = um_per_px or 1.0
    return {
        "einheit": "µm" if um_per_px else "px",
        "schranke": SCHRANKE,
        "anzahl": len(kanten),
        "verdeckt": len(ohne_pruefung) - len(kanten),
        "kanten": [
            {"a": k.a, "b": k.b,
             "steg": round(k.steg_px * faktor, 1),
             "rk": round(k.relativ_klein, 3),
             "rg": round(k.relativ_gross, 3),
             "von": [round(v, 1) for v in k.von],
             "nach": [round(v, 1) for v in k.nach]}
            for k in kanten
        ],
    }


def _stegverteilung(labels: np.ndarray, um_per_px: float | None) -> dict:
    """Haeufigkeitsverteilung des kuerzesten Abstands zur Nachbarpore.

    Gerechnet wird nur ueber die **gezaehlten** Poren: ein verworfenes Objekt ist keine
    Pore, und ein Steg zu etwas, das keine Pore ist, waere keine Aussage.

    """
    nachbarn = spatial.nachbarabstaende(labels, um_per_px=um_per_px)
    if len(nachbarn) < 2:
        return {"leer": True, "einheit": "µm" if um_per_px else "px"}

    if um_per_px:
        werte = [n.steg_um for n in nachbarn]
        einheit = "µm"
    else:
        werte = [n.steg_px for n in nachbarn]
        einheit = "px"

    grenzen = vert.kanten(werte, klassen=vert.klassenzahl(werte), logarithmisch=True)
    verteilung = vert.verteilen(werte, grenzen)

    return {
        "leer": False,
        "einheit": einheit,
        "logarithmisch": True,
        "kanten": list(grenzen),
        "anzahlen": list(verteilung.anzahlen),
        "maximum": verteilung.maximum,
        "kennwerte": vert.kennwerte(werte),
        "kurve": [[round(x, 3), round(y, 3)] for x, y in vert.dichtekurve(werte, grenzen)],
        "werte": sorted(round(w, 3) for w in werte),
        # Einzelne Aufloesungsstufe: ein Steg von einem Pixel ist an der Messgrenze.
        "ein_pixel": um_per_px if um_per_px else 1.0,
    }


# --------------------------------------------------------------------------------------
# Auswertung einsammeln
# --------------------------------------------------------------------------------------


def auswerten(bildpfad: Path) -> dict:
    """Ein Bild durch die Pipeline schicken und alles einsammeln, was die Ansicht zeigt."""
    cfg = AppConfig()
    pipeline = PoreDetectionPipeline(cfg)
    ctx = pipeline.analyse(bildpfad)
    ergebnis = pipeline._to_result(ctx, 0.0)

    gray = ctx.gray
    hoehe, breite = gray.shape
    behalten_ids = [p.label for p in ctx.pores]
    verworfen_ids = [r.pore.label for r in ctx.rejected]

    labels = ctx.labels
    lab_behalten = np.where(np.isin(labels, behalten_ids), labels, 0)
    lab_verworfen = np.where(np.isin(labels, verworfen_ids), labels, 0)

    harz = ctx.resin if ctx.resin is not None else np.zeros(gray.shape, dtype=bool)
    ausschluss = ctx.excluded if ctx.excluded is not None else np.zeros(gray.shape, dtype=bool)

    if ctx.specimen is not None and ctx.specimen.any():
        deckel = float(np.percentile(ctx.contrast[ctx.specimen], 99.5))
    else:
        deckel = 1.0

    if ctx.scale is not None:
        kontrollbild, kontrollmasse = _kontrollausschnitt(gray, ctx.scale)
    else:
        kontrollbild, kontrollmasse = None, None

    # Farben in BGR, weil cv2 sie so schreibt. Sie entsprechen den Tokens im Stylesheet.
    zoom = zoom_fuer(breite)
    ebenen = {
        "original": _png(_gross(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), zoom)),
        "untergrund": _png(_gross(cv2.cvtColor(
            np.clip(ctx.background, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), zoom)),
        "kontrast": _png(_heatmap(ctx.contrast, deckel, zoom)),
        "harz": _png(_flaeche(harz, (214, 92, 124), 115, zoom)),
        "ausschluss": _png(_flaeche(ausschluss, (208, 111, 47), 115, zoom)),
        "kandidaten": _png(_konturen(ctx.candidates.astype(np.int32), (37, 193, 201), zoom)),
        "poren": _png(_konturen(lab_behalten, (107, 154, 27), zoom, fuellung=70)),
        "verworfen": _png(_konturen(lab_verworfen, (43, 114, 184), zoom, fuellung=50)),
    }

    umrisse = _umrisse(labels)

    def _eintrag(p, status: str, grund: str | None) -> dict:
        return {
            "label": p.label, "status": status, "grund": grund,
            "x": p.centroid_px[0], "y": p.centroid_px[1],
            "r": max(p.equivalent_diameter_px / 2.0, 1.0),
            # Der umschliessende Kasten, nicht nur Schwerpunkt und Radius: die Lupe
            # muss auch eine langgezogene Pore ganz zeigen, und ein Kreis um den
            # Schwerpunkt schneidet sie ab. Vier Zahlen je Pore - das faellt neben
            # den Kennwerten nicht ins Gewicht.
            "box": [p.bbox.x, p.bbox.y, p.bbox.w, p.bbox.h],
            # Die gefundene Kontur selbst. Die Lupe zeichnet sie als Vektor - im
            # Gesamtbild reicht die gerasterte Ebene, in der Vergroesserung nicht.
            "umriss": umrisse.get(p.label, []),
            "flaeche_px": p.area_px, "d_px": p.equivalent_diameter_px,
            "d_um": p.equivalent_diameter_um, "flaeche_um2": p.area_um2,
            "rundheit": p.circularity, "solidity": p.solidity,
            "kontrast": p.contrast,
            "randbild": p.touches_image_edge, "randprobe": p.touches_specimen_edge,
        }

    poren = [_eintrag(p, "behalten", None) for p in ctx.pores]
    poren += [_eintrag(r.pore, "verworfen", f"{r.filter_name} - {r.reason}")
              for r in ctx.rejected]
    poren.sort(key=lambda d: -d["flaeche_px"])

    histogramm = _flaechenhistogramm(poren, ctx.um_per_px)
    durchmesser = {p.label: p.equivalent_diameter_px for p in ctx.pores}
    stege = _stegverteilung(lab_behalten, ctx.um_per_px)
    kanten = _kantenliste(lab_behalten, ctx.um_per_px, durchmesser)

    groesste = ergebnis.largest_pore
    return {
        "bild": bildpfad.name,
        # Steht im Kopf der Seite: so ist sofort zu sehen, ob der Browser noch eine
        # alte Fassung anzeigt, statt dass man einem Fehler nachjagt, der behoben ist.
        "erzeugt": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "breite": breite, "hoehe": hoehe, "zoom": zoom,
        "um_pro_px": ctx.um_per_px,
        "massstab": None if ctx.scale is None else {
            "um_pro_px": ctx.scale.um_per_px,
            # Die gelesene Beschriftung und das, was daraus geworden ist: beides gehoert
            # nebeneinander, sonst ist ein Faktor-zehn-Fehler der OCR nicht zu sehen.
            "text": ctx.scale.label_text,
            "wert_um": ctx.scale.value_um,
            "einheit": ctx.scale.unit_text,
            "balken_px": ctx.scale.bar_length_px,
            "konfidenz": ctx.scale.confidence,
            "engine": ctx.scale.engine,
            "quelle": str(getattr(ctx.scale.source, "value", ctx.scale.source)),
            "balken_box": _balkenkasten(ctx.scale.bar_box),
            "warnungen": list(ctx.scale.warnings),
            "ausschnitt": kontrollbild,
            "ausschnitt_masse": kontrollmasse,
        },
        "verfahren": {"probe": cfg.specimen.method, "poren": cfg.pore.method},
        "stufen": dict(ctx.stages),
        "warnungen": list(ctx.warnings),
        "kennzahlen": {
            "kandidaten": int(labels.max()),
            "behalten": len(ctx.pores),
            "verworfen": len(ctx.rejected),
            "porositaet": ergebnis.porosity_pct,
            "porositaet_ohne_rand": ergebnis.porosity_pct_excl_edge,
            "probe_px": ergebnis.specimen_area_px,
            "probe_mm2": ergebnis.specimen_area_mm2,
            "dichte": ergebnis.pore_density_per_mm2,
            "groesste_um": None if groesste is None else groesste.equivalent_diameter_um,
            "groesste_px": None if groesste is None else groesste.equivalent_diameter_px,
        },
        "ebenen": ebenen,
        "poren": poren,
        "histogramm": histogramm,
        "stege": stege,
        "kanten": kanten,
    }


# --------------------------------------------------------------------------------------
# Seite schreiben
# --------------------------------------------------------------------------------------


def schreiben(daten: dict, ziel: Path) -> Path:
    vorlage = (Path(__file__).parent / "gui_vorlage.html").read_text(encoding="utf-8")
    seite = vorlage.replace("__DATEN__", json.dumps(daten, ensure_ascii=False))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(seite, encoding="utf-8")
    return ziel


def bild_waehlen(ordner: Path) -> Path | None:
    """Das auszuwertende Bild aus dem Ordner.

    Liegt mehr als eines darin, gewinnt das **zuletzt geaenderte**: wer ein Bild zum
    Ausprobieren hineinlegt, meint dieses und nicht das, was alphabetisch vorn steht.
    Die uebrigen werden genannt, damit die Wahl nachvollziehbar bleibt.
    """
    try:
        bilder = find_images(ordner)
    except FileNotFoundError:
        print(f"Ordner nicht gefunden: {ordner}", file=sys.stderr)
        return None
    if not bilder:
        print(f"Keine Bilder in {ordner}", file=sys.stderr)
        return None

    gewaehlt = max(bilder, key=lambda p: p.stat().st_mtime)
    if len(bilder) > 1:
        andere = ", ".join(p.name for p in bilder if p != gewaehlt)
        print(f"{len(bilder)} Bilder im Ordner, genommen wird das neueste: {gewaehlt.name}")
        print(f"   uebersprungen: {andere}")
        print("   (ein bestimmtes Bild mit --bild waehlen)")
    return gewaehlt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ordner", type=Path, default=WURZEL / "test/GUI_Test",
                        help="Ordner mit dem auszuwertenden Bild (Standard: test/GUI_Test)")
    parser.add_argument("--bild", type=Path, default=None,
                        help="Einzelnes Bild statt des Ordners")
    parser.add_argument("-o", "--out", type=Path, default=WURZEL / "data/runs/gui/index.html")
    parser.add_argument("--nicht-oeffnen", action="store_true",
                        help="Die Seite nur schreiben, nicht im Browser anzeigen")
    args = parser.parse_args(argv)

    if args.bild is not None:
        if not args.bild.is_file():
            print(f"Bild nicht gefunden: {args.bild}", file=sys.stderr)
            return 2
        bild = args.bild
    else:
        bild = bild_waehlen(args.ordner)
        if bild is None:
            return 2

    daten = auswerten(bild)
    ziel = schreiben(daten, args.out)
    k = daten["kennzahlen"]
    print(f"{daten['bild']}: {k['behalten']} Poren, {k['verworfen']} verworfen, "
          f"{k['porositaet']:.2f} % Porositaet")
    print(f"Geschrieben: {ziel}  ({ziel.stat().st_size / 1e6:.1f} MB)")

    if not args.nicht_oeffnen:
        # as_uri() kodiert Leerzeichen und Umlaute im Pfad - der Projektpfad hat beides.
        webbrowser.open(ziel.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
