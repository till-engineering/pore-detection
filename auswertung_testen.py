"""Ein Auswertungsmodul ohne GUI an einem einzelnen Bild ausprobieren.

    .venv\Scripts\python.exe auswertung_testen.py <modulname> <bild> [zielordner]

Beispiel:
    .venv\Scripts\python.exe auswertung_testen.py groessenklassen test\real_pores\6_cropped_1_1.tif

Das Bild läuft durch dieselbe Pipeline wie im Startfenster (mit den Einstellungen aus
einstellungen.yaml), danach werden ``pro_bild`` und ``gesamt`` des Moduls aufgerufen.
Ausgabe landet in ``<zielordner>/auswertungen/<modulname>/`` (Standard: ``test_ausgabe``).
Fehler im Modul werden hier NICHT abgefangen - der Traceback zeigt direkt die Stelle.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from poren.auswertungen import ORDNERNAME, bilddaten
from poren.pipeline import Pipeline


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    name, bild = sys.argv[1], Path(sys.argv[2])
    ziel = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("test_ausgabe")
    modul = importlib.import_module(f"poren.auswertungen.{name}")

    pipeline = Pipeline()
    pipeline.config.corrections["directory"] = ziel / "korrekturen"
    ctx = pipeline.analyse(bild)
    ergebnis = Pipeline.to_result(ctx)
    print(f"{bild.name}: {ergebnis.describe()}")

    ordner = ziel / ORDNERNAME / name
    ordner.mkdir(parents=True, exist_ok=True)
    if hasattr(modul, "pro_bild"):
        modul.pro_bild(bilddaten(ctx, ergebnis), ordner)
        print("pro_bild: ok")
    if hasattr(modul, "gesamt"):
        modul.gesamt([ergebnis], ordner)
        print("gesamt: ok")
    print(f"Ausgabe in {ordner.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
