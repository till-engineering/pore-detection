"""Masken eines Bildes aus dem Zielordner eines Laufs zurückgewinnen.

Je Bildordner werden gelesen:

* ``daten/einbettmittel.json`` - Einbettmittel als Polygone -> Einbettmittelmaske
* ``daten/poren.json``         - gezählte Poren als Polygone (mit Korrekturen) -> Porenmaske
* ``<bild>_maske.png``         - weiß = Probe ohne Poren; daraus Probenfläche
                                 (weiß + Poren) und Maßstabskasten (schwarz, aber weder
                                 Einbettmittel noch Pore)
* ``<bild>_ergebnis.png``      - linke Hälfte = Original, nur als Hintergrund fürs Kontrollbild
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BildMasken:
    ordner: Path
    bild: str                  # Dateiname des Originalbilds
    stamm: str                 # Dateiname ohne Endung - Vorsilbe der Ausgabedateien
    um_pro_px: float | None
    probe: np.ndarray          # bool - auswertbare Probe inkl. Poren
    poren: np.ndarray          # bool - gezählte Poren
    einbettmittel: np.ndarray  # bool
    ausgeschlossen: np.ndarray  # bool - Maßstabskasten
    hintergrund: np.ndarray    # uint8 BGR oder grau


def bildordner(pfad: Path) -> list[Path]:
    """``pfad`` ist ein Bildordner oder der Zielordner eines Laufs."""
    pfad = Path(pfad)
    if _ist_bildordner(pfad):
        return [pfad]
    return sorted(o for o in pfad.iterdir() if o.is_dir() and _ist_bildordner(o))


def _ist_bildordner(ordner: Path) -> bool:
    return ((ordner / "daten" / "einbettmittel.json").is_file()
            and (ordner / "daten" / "poren.json").is_file())


def laden(ordner: Path) -> BildMasken:
    ordner = Path(ordner)
    harz = _json(ordner / "daten" / "einbettmittel.json")
    poren_daten = _json(ordner / "daten" / "poren.json")
    form = (int(harz["hoehe_px"]), int(harz["breite_px"]))
    stamm = Path(harz["bild"]).stem

    einbettmittel = _polygone(harz["bereiche"], form)
    poren = np.zeros(form, dtype=bool)
    for pore in poren_daten["poren"]:
        poren |= _polygone(pore["umriss"], form)

    maske = _png(ordner / f"{stamm}_maske.png", cv2.IMREAD_GRAYSCALE)
    if maske is not None and maske.shape == form:
        weiss = maske > 127
        probe = weiss | poren
        ausgeschlossen = ~weiss & ~einbettmittel & ~poren
    else:  # ohne Maske: alles außer Einbettmittel ist Probe
        probe = ~einbettmittel
        ausgeschlossen = np.zeros(form, dtype=bool)

    ergebnis = _png(ordner / f"{stamm}_ergebnis.png", cv2.IMREAD_COLOR)
    if ergebnis is not None and ergebnis.shape[0] == form[0] and ergebnis.shape[1] > form[1]:
        hintergrund = np.ascontiguousarray(ergebnis[:, :form[1]])
    else:
        hintergrund = np.where(probe, 230, 60).astype(np.uint8)

    return BildMasken(ordner, harz["bild"], stamm, harz.get("um_pro_px"), probe, poren,
                      einbettmittel, ausgeschlossen, hintergrund)


def _polygone(bereiche: list[dict], form: tuple[int, int]) -> np.ndarray:
    """Bereiche aus der JSON in eine Maske: je Bereich Außenumriss füllen, Löcher leeren,
    dann vereinigen. Je Bereich nur im eigenen umschließenden Kasten - schnell."""
    maske = np.zeros(form, dtype=bool)
    for bereich in bereiche:
        aussen = np.asarray(bereich["aussen"], dtype=np.int32).reshape(-1, 2)
        x0, y0 = aussen.min(axis=0)
        x1, y1 = aussen.max(axis=0)
        teil = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        versatz = np.array([x0, y0], dtype=np.int32)
        cv2.fillPoly(teil, [aussen - versatz], 1)
        for loch in bereich.get("loecher", []):
            loch = np.asarray(loch, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(teil, [loch - versatz], 0)
        maske[y0:y1 + 1, x0:x1 + 1] |= teil.astype(bool)
    return maske


def _json(pfad: Path) -> dict:
    with pfad.open(encoding="utf-8") as f:
        return json.load(f)


def _png(pfad: Path, art: int) -> np.ndarray | None:
    # np.fromfile + imdecode statt imread: cv2.imread scheitert an Umlauten im Pfad.
    if not pfad.is_file():
        return None
    return cv2.imdecode(np.fromfile(str(pfad), dtype=np.uint8), art)
