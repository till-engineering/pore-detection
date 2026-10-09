"""Synthetische Prüfung: Bilder mit bekannten Poren erzeugen, auswerten, vergleichen.

    .venv\\Scripts\\python.exe synthetik_testen.py [anzahl] [--seed N] [--ohne-viewer]

* ``anzahl``        wie viele Bilder (Standard 6)
* ``--seed N``      dieselben Bilder wie ein früherer Lauf (der Seed steht im Bericht);
                    ohne Angabe jedes Mal neue Bilder
* ``--ohne-viewer`` am Ende nicht den Viewer öffnen - nur Bericht und Konsole

Ablauf:

1. ``test_synthetik/`` wird komplett geleert.
2. Bilder erzeugen: helle Probe, dunkles Einbettmittel hinter gewelltem Rand, dunkle Formen
   mit bekannter Größe, dazu Grenzfälle (siehe ``synthetik/erzeugen.py``).
3. Auswerten wie im Startfenster - derselbe Stapellauf, **immer mit den
   Standardeinstellungen** (``einstellungen.yaml``); deine ``einstellungen_eigene.yaml``
   wird nicht gelesen und nicht verändert.
4. Prüfen: Probenmaske, jede Form (gefunden? Größe?), Falschfunde, Porosität - und ob
   Maske, Ergebnisbild, CSV und JSON zueinander und zu den Pixeln passen.
5. Bericht ablegen: ``test_synthetik/bericht.html`` und ``bericht.txt``, die
   Vergleichsbilder in ``test_synthetik/pruefung/``. Danach öffnet sich der Viewer.

Was im Viewer an den Einstellungen verstellt wird, gilt nur für diesen Testordner (dort
``einstellungen_eigene.yaml``) und ist beim nächsten Lauf wieder weg.

Rückgabewert: 0 ohne FEHLER, 1 mit FEHLER, 2 bei falschem Aufruf.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

WURZEL = Path(__file__).resolve().parent
ORDNER = WURZEL / "test_synthetik"
#: Nur ein Ordner mit dieser Datei wird beim Start geleert - nie ein fremder.
MARKE = ".synthetik_testordner"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Synthetische Prüfung der Porenauswertung")
    parser.add_argument("anzahl", nargs="?", type=int, default=6, help="Anzahl Bilder")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ohne-viewer", action="store_true")
    args = parser.parse_args()
    if args.anzahl < 1:
        parser.error("anzahl muss mindestens 1 sein")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1_000_000)

    if not _aufraeumen(ORDNER):
        return 2

    # Immer der Standard: keine eigene Einstellungsdatei von außen, und was der Viewer
    # speichert, landet im Testordner statt in deiner einstellungen_eigene.yaml.
    os.environ.pop("POREN_EINSTELLUNGEN", None)
    from poren import einstellungen as einst
    einst.EIGENE = ORDNER / "einstellungen_eigene.yaml"

    from poren import stapel
    from synthetik import bericht, pruefen
    from synthetik.erzeugen import erzeugen

    bilder, wahrheit = ORDNER / "bilder", ORDNER / "wahrheit"
    ziel, pruefung = ORDNER / "ergebnis", ORDNER / "pruefung"
    pruefung.mkdir(parents=True)

    print(f"Erzeuge {args.anzahl} Bild(er), Seed {seed} …")
    erzeugen(args.anzahl, seed, bilder, wahrheit)

    print("Werte aus (Standardeinstellungen) …")

    def melden(nummer, gesamt, pfad, ergebnis, fehler):
        print(f"  {nummer:>3}/{gesamt}  {pfad.name}: "
              + (f"FEHLER - {fehler}" if fehler else f"{ergebnis.pore_count} Poren"))

    mappe = stapel.verarbeiten(bilder, ziel, melden=melden)

    print("Prüfe …")
    import cv2
    import numpy as np

    def png(pfad):
        return cv2.imdecode(np.fromfile(str(pfad), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    ergebnisse = []
    for i, pfad in enumerate(mappe.bilder):
        stem = pfad.stem
        wahr = json.loads((wahrheit / f"{stem}.json").read_text("utf-8"))
        formen_px = png(wahrheit / f"{stem}_formen.png").astype(np.int32)
        probe_wahr = png(wahrheit / f"{stem}_probe.png") > 0
        ergebnis, ctx = mappe.sitzung(i).ergebnis()
        b = pruefen.pruefe_bild(wahr, formen_px, probe_wahr, ergebnis, ctx, mappe.ordner[i])
        bericht.vergleichsbild(pruefung / f"{stem}_vergleich.png", ctx.gray, probe_wahr,
                               formen_px, ergebnis, ctx, b)
        ergebnisse.append(b)

    gesamt = pruefen.pruefe_gesamt(ziel, mappe.ergebnisse)
    for pfad, grund in mappe.fehler:
        gesamt.append(pruefen.Befund(pruefen.FEHLER, f"Auswertung {pfad.name}", grund))
    for text in mappe.schreibfehler:
        gesamt.append(pruefen.Befund(pruefen.FEHLER, "Datei nicht geschrieben", text))

    kopf = {"zeit": datetime.now().strftime("%d.%m.%Y %H:%M:%S"), "seed": seed,
            "einstellungen": "Standard (einstellungen.yaml)"}
    zusammenfassung = bericht.text(ergebnisse, gesamt, kopf)
    # Der Bericht liegt oben im Testordner, die Vergleichsbilder in pruefung/.
    (ORDNER / "bericht.txt").write_text(zusammenfassung, encoding="utf-8")
    (ORDNER / "bericht.html").write_text(
        bericht.html_bericht(ergebnisse, gesamt, kopf, bildordner=pruefung.name), encoding="utf-8")
    print()
    print(zusammenfassung)
    print(f"Bericht: {ORDNER / 'bericht.html'}")
    print(f"Wiederholen mit denselben Bildern: synthetik_testen.py {args.anzahl} --seed {seed}")

    fehler = bericht.zaehlen(ergebnisse, gesamt)[pruefen.FEHLER] > 0
    if not args.ohne_viewer and mappe.anzahl():
        print("Viewer läuft - Fenster schließen beendet das Skript.")
        from poren.viewer.fenster import Viewer

        viewer = Viewer(lambda: mappe, beenden_beim_schliessen=True)
        threading.Thread(target=viewer.zeigen, daemon=True).start()
        viewer.ausfuehren()
    return 1 if fehler else 0


def _aufraeumen(ordner: Path) -> bool:
    """Den Testordner komplett leeren und neu anlegen. Ein Ordner ohne Marke wird nicht
    angefasst - so kann nie versehentlich etwas anderes gelöscht werden."""
    if ordner.exists():
        if not (ordner / MARKE).is_file() and any(ordner.iterdir()):
            print(f"{ordner} enthält Dateien, ist aber kein Testordner (es fehlt {MARKE}) - "
                  "bitte selbst prüfen und leeren.")
            return False
        try:
            shutil.rmtree(ordner)
        except OSError as exc:
            print(f"{ordner} ließ sich nicht leeren ({exc}) - ist eine Datei daraus noch "
                  "geöffnet (Excel, Bildbetrachter, Viewer)?")
            return False
    ordner.mkdir(parents=True)
    (ordner / MARKE).write_text("Von synthetik_testen.py angelegt - wird bei jedem Lauf geleert.\n",
                                encoding="utf-8")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
