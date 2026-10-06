"""Logs im Zielordner: welche Einstellungen wann galten und was mit jedem Bild geschah.

* ``<Zielordner>/einstellungen_log.txt`` - ein Eintrag je Laufstart und je Übernahme
  neuer Einstellungen im Viewer: was gegenüber vorher geändert wurde und was insgesamt
  vom Standard (``einstellungen.yaml``) abweicht.
* ``<Zielordner>/<bild>/log.txt`` - ein Eintrag je Auswertung und Korrektur dieses Bildes.
* ``<Zielordner>/<bild>/einstellungen.yaml`` - die vollständigen Einstellungen, mit denen
  das Ergebnis in diesem Ordner gerechnet wurde.

Die Logs werden nur ergänzt, nie überschrieben.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import einstellungen as einst

LAUF_LOG = "einstellungen_log.txt"
BILD_LOG = "log.txt"
BILD_EINSTELLUNGEN = "einstellungen.yaml"


def _zeit() -> str:
    return f"{datetime.now():%d.%m.%Y %H:%M:%S}"


def _anhaengen(datei: Path, text: str) -> None:
    datei.parent.mkdir(parents=True, exist_ok=True)
    with datei.open("a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n\n")


def _liste(titel: str, zeilen: list[str], leer: str) -> list[str]:
    if not zeilen:
        return [f"{titel}: {leer}"]
    return [f"{titel}:"] + [f"  {z}" for z in zeilen]


def abweichungen(werte: dict) -> list[str]:
    """Was in ``werte`` vom Standard abweicht, als Zeilen."""
    return einst.unterschiede(einst.standard(), werte)


def lauf(ziel: Path, ereignis: str, werte: dict, vorher: dict | None = None) -> None:
    """Einen Eintrag in ``einstellungen_log.txt`` schreiben."""
    zeilen = [f"==== {_zeit()}  {ereignis}"]
    if vorher is not None:
        zeilen += _liste("Geändert gegenüber vorher", einst.unterschiede(vorher, werte),
                         "nichts")
    zeilen += _liste("Abweichungen vom Standard", abweichungen(werte),
                     "keine (einstellungen.yaml)")
    _anhaengen(Path(ziel) / LAUF_LOG, "\n".join(zeilen))


def bild(ordner: Path, ereignis: str, zeilen: list[str] = ()) -> None:
    """Einen Eintrag in ``<bild>/log.txt`` schreiben."""
    _anhaengen(Path(ordner) / BILD_LOG, "\n".join([f"---- {_zeit()}  {ereignis}", *zeilen]))


def bild_einstellungen(ordner: Path, werte: dict) -> None:
    """Die vollständigen Einstellungen eines Ergebnisses neben das Ergebnis legen."""
    Path(ordner).mkdir(parents=True, exist_ok=True)
    kopf = (f"# Einstellungen, mit denen dieses Ergebnis gerechnet wurde ({_zeit()}).\n"
            "# Vollständig - als einstellungen.yaml verwendbar.\n\n")
    (Path(ordner) / BILD_EINSTELLUNGEN).write_text(kopf + einst.als_yaml(werte),
                                                   encoding="utf-8")
