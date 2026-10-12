"""Zonenauswertung über die Konsole.

    .venv\\Scripts\\python.exe -m zonen <zielordner eines Laufs | bildordner> [--einstellungen datei.yaml]

Schreibt je Bildordner ``<bild>_zonen.csv`` und ``<bild>_zonen.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ausgabe, einlesen, einstellungen, rechnen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m zonen", description=__doc__.split("\n")[0])
    parser.add_argument("ordner", type=Path, help="Zielordner eines Laufs oder ein Bildordner")
    parser.add_argument("--einstellungen", type=Path, default=None,
                        help="andere YAML statt zonen/einstellungen_zonen.yaml")
    args = parser.parse_args(argv)

    einst = einstellungen.laden(args.einstellungen)
    ordner = einlesen.bildordner(args.ordner)
    if not ordner:
        print(f"Keine Bildordner (mit daten/einbettmittel.json) unter {args.ordner}")
        return 1

    fehler = 0
    for o in ordner:
        try:
            m = einlesen.laden(o)
            try:
                erg = rechnen.berechnen(m.bild, m.probe, m.poren, m.einbettmittel,
                                        m.ausgeschlossen, m.um_pro_px, einst)
            except rechnen.KeineKante as grund:
                ausgabe.hinweis_schreiben(o, m.stamm, m.bild, str(grund))
                print(f"{o.name}: keine Zonen - {grund}")
                continue
            ausgabe.schreiben(o, m.stamm, erg, m.hintergrund, m.einbettmittel, m.poren)
            print(f"{o.name}: " + _kurz(erg))
            for text in erg.warnungen:
                print(f"    Warnung: {text}")
        except Exception as exc:  # noqa: BLE001 - ein Bild darf den Lauf nicht abbrechen
            fehler += 1
            print(f"{o.name}: FEHLER - {type(exc).__name__}: {exc}")
    return 1 if fehler else 0


def _kurz(erg: rechnen.ZonenErgebnis) -> str:
    teile = []
    for z in erg.zonen:
        p = z.porositaet_pct
        teile.append(f"{z.name}: " + ("-" if p is None else f"{p:.3f} %".replace(".", ",")))
    f = erg.fehler
    return " | ".join(teile) + f"  (Unterkante {f.winkel_grad:+.2f}°)".replace(".", ",")


if __name__ == "__main__":
    sys.exit(main())
