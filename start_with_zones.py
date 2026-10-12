"""Startfenster wie ``start.py`` - zusätzlich mit Zonenauswertung (``zonen/``).

Startet die bestehende Pipeline unverändert und hängt die Zonen an zwei Stellen ein:

* als Auswertungsmodul: nach jedem Bild und nach jeder Korrektur im Viewer entstehen im
  Ordner des Bildes ``<bild>_zonen.csv`` und ``<bild>_zonen.png``;
* im Viewer: Ebene "Zonen" (Linien im Bild) und die Karte "Porosität je Zone".

Die Zonen stehen in ``zonen/einstellungen_zonen.yaml`` (wird bei jeder Berechnung neu
gelesen). ``start.py`` bleibt davon unberührt und zeigt keine Zonen.

Aufruf::

    .venv\\Scripts\\python.exe start_with_zones.py
"""

from __future__ import annotations

import types

import start
from poren import auswertungen
from poren.viewer import zusaetze


# Die Zonen laden OpenCV - erst beim ersten Aufruf, damit das Fenster sofort erscheint.
def _zonen_pro_bild(bild, ordner):
    from zonen import bruecke
    bruecke.pro_bild(bild, ordner)


def _zonen_fuer_viewer(ctx):
    from zonen import bruecke
    return bruecke.fuer_viewer(ctx)


def main() -> int:
    # Name "zonen": Fehler landen unter diesem Namen in auswertungen/fehler.txt.
    modul = types.ModuleType("zonen")
    modul.pro_bild = _zonen_pro_bild
    auswertungen.module().append(modul)
    zusaetze.LIEFERANTEN["zonen"] = _zonen_fuer_viewer
    return start.main()


if __name__ == "__main__":
    raise SystemExit(main())
