"""Zonen-CSV und Kontrollbild in den Ordner eines Bildes schreiben.

* ``<bild>_zonen.csv`` - oben die Kenngrößen (Dicke, Kantenwinkel, Fehler an der
  Unterkante), darunter die Zonentabelle. Semikolon, Dezimalkomma, UTF-8 mit BOM: Excel
  öffnet die Datei direkt richtig.
* ``<bild>_zonen.png`` - das Bild mit Einbettmittel (blau), Poren (rot), Kantenpunkten,
  allen Linien (gelb) und der Porosität je Zone; rechts daneben die Abstände zwischen
  den Linien und die Gesamtdicke als Maßketten (µm, ohne Maßstab px).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .rechnen import Gerade, ZonenErgebnis

# Farben BGR
_EINBETTMITTEL = (200, 120, 40)
_POREN = (0, 0, 255)
_GELB = (0, 230, 255)           # alle Linien und Maßketten
_KANTE = (40, 170, 40)          # Kantenpunkte oben (genutzt)
_UNTERKANTE = (200, 0, 200)     # Kantenpunkte unten (genutzt)
_AUSREISSER = (150, 150, 150)   # Kantenpunkte, die die Gerade nicht tragen
_LEISTE = (45, 45, 45)          # Hintergrund der Abstandsleiste


def csv_pfad(ordner: Path, stamm: str) -> Path:
    return Path(ordner) / f"{stamm}_zonen.csv"


def bild_pfad(ordner: Path, stamm: str) -> Path:
    return Path(ordner) / f"{stamm}_zonen.png"


def schreiben(ordner: Path, stamm: str, erg: ZonenErgebnis, hintergrund: np.ndarray,
              einbettmittel: np.ndarray, poren: np.ndarray) -> None:
    csv_schreiben(csv_pfad(ordner, stamm), erg)
    _png_schreiben(bild_pfad(ordner, stamm),
                   kontrollbild(erg, hintergrund, einbettmittel, poren))


def hinweis_schreiben(ordner: Path, stamm: str, bild: str, grund: str) -> None:
    """Keine Zonen möglich: CSV mit dem Grund, altes Kontrollbild weg."""
    with csv_pfad(ordner, stamm).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["kenngroesse", "wert", "einheit"])
        w.writerow(["bild", bild, ""])
        w.writerow(["keine Zonenauswertung", grund, ""])
    bild_pfad(ordner, stamm).unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------

def csv_schreiben(pfad: Path, erg: ZonenErgebnis) -> Path:
    um = erg.um_pro_px
    laenge = "um" if um else "px"

    def l(px: float | None) -> str:              # Länge in der Ausgabeeinheit
        return "" if px is None else _zahl(px * um if um else px)

    def rel(px: float) -> str:                   # Anteil der Dicke in %
        return _zahl(100.0 * px / erg.dicke_px)

    def mm2(px: int) -> str:
        return _zahl(px * um * um * 1e-6) if um else ""

    f = erg.fehler
    kenn = [
        ["bild", erg.bild, ""],
        ["massstab", _zahl(um) if um else "kein Maßstab", "um/px" if um else ""],
        ["zonengrenzen", " | ".join(_zahl(g) for g in erg.grenzen_pct), "% der Dicke"],
        ["dicke (0 bis 100 %)", l(erg.dicke_px), laenge],
        ["oberkante_winkel", _zahl(erg.oben.gerade.winkel_grad), "grad zur Bildwaagerechten"],
        ["unterkante_winkel", _zahl(erg.unten.gerade.winkel_grad), "grad zur Bildwaagerechten"],
        ["fehler_winkel", _zahl(f.winkel_grad), "grad Unterkante gegen Oberkante"],
        ["fehler_links", l(f.links_px), f"{laenge} bei x = {f.x_links:.0f} px"],
        ["fehler_links_rel", rel(f.links_px), "% der Dicke"],
        ["fehler_rechts", l(f.rechts_px), f"{laenge} bei x = {f.x_rechts:.0f} px"],
        ["fehler_rechts_rel", rel(f.rechts_px), "% der Dicke"],
        ["fehler_max", l(f.max_px), f"{laenge} bei x = {f.x_max:.0f} px"],
        ["fehler_max_rel", rel(f.max_px), "% der Dicke"],
        ["fehler_rms", l(f.rms_px), laenge],
        ["fehler_rms_rel", rel(f.rms_px), "% der Dicke"],
        ["kantenpunkte_oben", f"{int(erg.oben.genutzt.sum())} von {len(erg.oben.x)}",
         "genutzt von gefunden"],
        ["kantenpunkte_unten", f"{int(erg.unten.genutzt.sum())} von {len(erg.unten.x)}",
         "genutzt von gefunden"],
        *(["warnung", text, ""] for text in erg.warnungen),
    ]
    with pfad.open("w", newline="", encoding="utf-8-sig") as datei:
        w = csv.writer(datei, delimiter=";")
        w.writerow(["kenngroesse", "wert", "einheit"])
        w.writerows(kenn)
        w.writerow([])
        w.writerow(["zone", "von_pct", "bis_pct", f"von_{laenge}", f"bis_{laenge}",
                    "probenflaeche_px", "probenflaeche_mm2", "porenflaeche_px",
                    "porenflaeche_mm2", "porositaet_pct"])
        for z in [*erg.zonen, erg.gesamt]:
            w.writerow([z.name, _zahl(z.von_pct), _zahl(z.bis_pct), l(z.von_px),
                        l(z.bis_px), z.probe_px, mm2(z.probe_px), z.poren_px,
                        mm2(z.poren_px), _zahl(z.porositaet_pct)])
    return pfad


def _zahl(wert: float | None) -> str:
    return "" if wert is None else f"{wert:.6g}".replace(".", ",")


# --------------------------------------------------------------------------------------
# Kontrollbild
# --------------------------------------------------------------------------------------

def kontrollbild(erg: ZonenErgebnis, hintergrund: np.ndarray, einbettmittel: np.ndarray,
                 poren: np.ndarray) -> np.ndarray:
    """Das Bild mit allen Linien, rechts daneben eine Leiste mit den Abständen."""
    bild =(cv2.cvtColor(hintergrund, cv2.COLOR_GRAY2BGR) if hintergrund.ndim == 2
            else hintergrund.copy())
    bild[einbettmittel] = (0.55 * bild[einbettmittel]
                           + 0.45 * np.array(_EINBETTMITTEL)).astype(np.uint8)
    bild[poren] = _POREN

    groesse = max(erg.breite_px, erg.hoehe_px)
    dicke = max(1, round(groesse / 900))
    schrift = max(0.4, groesse / 2200)

    # Kantenpunkte: genutzte farbig, Ausreißer grau.
    for kante, farbe in ((erg.oben, _KANTE), (erg.unten, _UNTERKANTE)):
        for x, y, ok in zip(kante.x, kante.y, kante.genutzt):
            cv2.circle(bild, (int(x), int(round(y))), dicke, farbe if ok else _AUSREISSER, -1)

    # Linien, alle gelb: 0 und 100 % kräftig, Zonengrenzen dünn, Mittellinie (50 %)
    # gestrichelt, echte Unterkante (Regression) gepunktet.
    x0, x1 = erg.x_von, erg.x_bis
    for g in erg.grenzen_pct:
        if g != 50.0:
            _linie(bild, erg.grenzlinie(g), x0, x1, _GELB,
                   2 * dicke if g in (0.0, 100.0) else dicke)
    _linie(bild, erg.grenzlinie(50.0), x0, x1, _GELB, dicke, muster="strich")
    _linie(bild, erg.unten.gerade, x0, x1, _GELB, dicke, muster="punkt")

    # Beschriftung je Zone in der Zonenmitte, abwechselnd links und rechts.
    for k, zone in enumerate(z for z in erg.zonen if z.von_pct is not None
                             and z.bis_pct is not None):
        mitte = erg.grenzlinie((zone.von_pct + zone.bis_pct) / 2)
        x = x0 + (x1 - x0) * (0.12 if k % 2 == 0 else 0.62)
        p = zone.porositaet_pct
        text = (f"{zone.name}: {_zahl(zone.von_pct)}-{_zahl(zone.bis_pct)} %  "
                f"P = {'-' if p is None else f'{p:.3f}'.replace('.', ',')} %")
        _text(bild, text, (int(x), int(mitte.y(x))), schrift, dicke)

    f = erg.fehler
    einheit = _einheit(erg)
    _text(bild, f"Fehler Unterkante: {_zahl(round(f.winkel_grad, 2))} Grad, "
                f"links {_zahl(_laenge(erg, f.links_px))} {einheit}, "
                f"rechts {_zahl(_laenge(erg, f.rechts_px))} {einheit}",
          (int(x0), int(erg.hoehe_px - 8 * dicke)), schrift, dicke)
    return _mit_abstaenden(bild, erg, dicke, schrift)


def _mit_abstaenden(bild: np.ndarray, erg: ZonenErgebnis, dicke: int,
                    schrift: float) -> np.ndarray:
    """Leiste rechts neben dem Bild: Maßkette zwischen den Linien und eine über die ganze
    Dicke. Gemessen senkrecht zur Oberkante; die Linien werden am rechten Probenende
    (``x_bis``) waagerecht in die Leiste weitergeführt."""
    hoehe, breite = bild.shape[:2]
    einheit = _einheit(erg)
    grenzen = sorted(set(erg.grenzen_pct) | {0.0, 100.0})
    ys = [erg.grenzlinie(g).y(erg.x_bis) for g in grenzen]

    abschnitte = []                       # (y oben, y unten, Text)
    for k in range(1, len(grenzen)):
        laenge = (grenzen[k] - grenzen[k - 1]) / 100.0 * erg.dicke_px
        abschnitte.append((ys[k - 1], ys[k], f"{_zahl(_laenge(erg, laenge))} {einheit}"))
    gesamt = f"gesamt {_zahl(_laenge(erg, erg.dicke_px))} {einheit}"

    def textbreite(text: str) -> int:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, schrift, dicke)[0][0]

    texthoehe = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, schrift, dicke)[0][1]
    rand = 2 * texthoehe
    spalte = max(textbreite(t) for *_, t in abschnitte)
    # Zu schmale Abschnitte: jede zweite Beschriftung weicht in eine zweite Spalte aus.
    versetzt = [k % 2 == 1 and abs(yb - ya) < 1.6 * texthoehe
                for k, (ya, yb, _) in enumerate(abschnitte)]
    spalten = 2 if any(versetzt) else 1

    x_kette = breite + 2 * rand                       # Maßkette der Zonen
    x_text = x_kette + rand
    x_gesamt = x_text + spalten * (spalte + rand)     # Maßkette gesamt
    rechts = x_gesamt + rand + textbreite(gesamt) + rand

    # Ragen Linien über das Bild hinaus (schräge Probe), wird die Leinwand höher.
    oben = int(max(0, -min(ys) + rand))
    unten = int(max(0, max(ys) - hoehe + rand))
    leinwand = np.full((hoehe + oben + unten, rechts, 3), _LEISTE, dtype=np.uint8)
    leinwand[oben:oben + hoehe, :breite] = bild
    ys = [y + oben for y in ys]
    y0, y100 = ys[grenzen.index(0.0)], ys[grenzen.index(100.0)]
    fein = max(1, dicke // 2)

    for y in ys:                                      # Linien in die Leiste weiterführen
        y = int(round(y))
        cv2.line(leinwand, (erg.x_bis, y), (x_kette + rand // 2, y), _GELB, fein, cv2.LINE_AA)
    for k, (ya, yb, text) in enumerate(abschnitte):
        _masspfeil(leinwand, x_kette, ya + oben, yb + oben, dicke)
        x = x_text + (spalte + rand if versetzt[k] else 0)
        _text(leinwand, text, (x, int(round((ya + yb) / 2 + oben))), schrift, dicke,
              hintergrund=_LEISTE, farbe=(255, 255, 255))
    for y in (y0, y100):
        y = int(round(y))
        cv2.line(leinwand, (x_kette, y), (x_gesamt + rand // 2, y), _GELB, fein, cv2.LINE_AA)
    _masspfeil(leinwand, x_gesamt, y0, y100, dicke)
    _text(leinwand, gesamt, (x_gesamt + rand, int(round((y0 + y100) / 2))), schrift, dicke,
          hintergrund=_LEISTE, farbe=(255, 255, 255))
    return leinwand


def _masspfeil(bild: np.ndarray, x: int, ya: float, yb: float, dicke: int) -> None:
    """Senkrechte Maßlinie mit Pfeilspitzen an beiden Enden."""
    a, b = (x, int(round(ya))), (x, int(round(yb)))
    spitze = min(0.4, 6 * dicke / max(1.0, abs(yb - ya)))
    cv2.arrowedLine(bild, a, b, _GELB, dicke, cv2.LINE_AA, tipLength=spitze)
    cv2.arrowedLine(bild, b, a, _GELB, dicke, cv2.LINE_AA, tipLength=spitze)


def _einheit(erg: ZonenErgebnis) -> str:
    return "um" if erg.um_pro_px else "px"


def _laenge(erg: ZonenErgebnis, px: float) -> float:
    return round(px * erg.um_pro_px if erg.um_pro_px else px, 1)


def _linie(bild: np.ndarray, g: Gerade, x0: int, x1: int, farbe, dicke: int,
           muster: str | None = None) -> None:
    """Gerade von x0 bis x1: durchgezogen, ``"strich"`` oder ``"punkt"``."""
    if muster is None:
        cv2.line(bild, (x0, int(round(g.y(x0)))), (x1, int(round(g.y(x1)))), farbe, dicke,
                 cv2.LINE_AA)
        return
    strich = max(8, 8 * dicke) if muster == "strich" else max(2, dicke)
    luecke = strich if muster == "strich" else 3 * strich
    for xa in range(x0, x1, strich + luecke):
        xb = min(xa + strich, x1)
        cv2.line(bild, (xa, int(round(g.y(xa)))), (xb, int(round(g.y(xb)))), farbe, dicke,
                 cv2.LINE_AA)


def _text(bild: np.ndarray, text: str, ort: tuple[int, int], schrift: float,
          dicke: int, hintergrund=(255, 255, 255), farbe=(0, 0, 0)) -> None:
    """Text senkrecht mittig auf ``ort[1]``, auf einem Kasten in ``hintergrund``."""
    (b, h), unten = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, schrift, dicke)
    x, y = ort[0], ort[1] + h // 2
    cv2.rectangle(bild, (x - 3, y - h - 3), (x + b + 3, y + unten + 1), hintergrund, -1)
    cv2.putText(bild, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, schrift, farbe, dicke,
                cv2.LINE_AA)


def _png_schreiben(pfad: Path, bild: np.ndarray) -> Path:
    # imencode + tofile statt imwrite: cv2.imwrite scheitert an Umlauten im Pfad.
    ok, puffer = cv2.imencode(".png", bild, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"PNG nicht kodierbar: {pfad}")
    puffer.tofile(str(pfad))
    return pfad
