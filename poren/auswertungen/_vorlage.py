"""VORLAGE für ein Auswertungsmodul - wird nicht ausgeführt (Name beginnt mit "_").

Kopieren, umbenennen (z. B. ``groessenklassen.py``, ohne "_" vorn), anpassen.
Beide Funktionen sind optional - es reicht eine.
"""

from __future__ import annotations

import csv
from pathlib import Path

from poren.auswertungen import BildDaten
from poren.modelle import ImageResult

# AKTIV = False   # auskommentieren, um das Modul abzuschalten, ohne es zu löschen


def pro_bild(bild: BildDaten, ordner: Path) -> None:
    """Läuft je Bild. Beispiel: Poren je Durchmesserklasse als CSV."""
    grenzen = [0, 10, 25, 50, 100, float("inf")]           # µm (bzw. px ohne Maßstab)
    einheit = "um" if bild.um_pro_px else "px"
    anzahl = [0] * (len(grenzen) - 1)
    for pore in bild.poren:
        d = pore.equivalent_diameter_um if bild.um_pro_px else pore.equivalent_diameter_px
        for i in range(len(anzahl)):
            if grenzen[i] <= d < grenzen[i + 1]:
                anzahl[i] += 1
                break

    datei = ordner / f"{Path(bild.name).stem}.csv"
    with datei.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([f"von_{einheit}", f"bis_{einheit}", "anzahl"])
        for i, n in enumerate(anzahl):
            w.writerow([grenzen[i], grenzen[i + 1], n])


def gesamt(bilder: list[ImageResult], ordner: Path) -> None:
    """Läuft einmal über alle Bilder. Beispiel: Porosität je Bild in einer Tabelle."""
    with (ordner / "porositaet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["bild", "porositaet_pct", "poren"])
        for e in bilder:
            p = "" if e.porosity_pct is None else f"{e.porosity_pct:.4f}".replace(".", ",")
            w.writerow([e.name, p, e.pore_count])
