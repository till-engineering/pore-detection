"""Einstellungen aus ``einstellungen.yaml`` laden - und eigene Abweichungen davon.

Kein Schema: die YAML wird gelesen und ist dann per Punkt erreichbar, z. B.
``cfg.pore.method`` oder ``cfg.specimen.grabcut.iterations``. Jeder Wert, den der Code
braucht, muss in der Datei stehen - fehlt einer, sagt die Fehlermeldung welcher.

**Zwei Dateien.** ``einstellungen.yaml`` ist der Standard und wird vom Programm nie
geschrieben - die Kommentare darin bleiben erhalten. Was im Viewer verstellt wird, landet
in ``einstellungen_eigene.yaml``, und zwar **nur die Abweichungen** vom Standard. Beim
Laden wird diese Datei über den Standard gelegt. "Auf Standard zurücksetzen" heißt also:
die eigene Datei löschen.

Listen (die Filter) werden nicht einzeln überlagert, sondern als Ganzes ersetzt.
"""

from __future__ import annotations

import copy
import os
import re
from datetime import datetime
from pathlib import Path

import yaml

PROJEKT = Path(__file__).resolve().parents[1]
DATEI = PROJEKT / "einstellungen.yaml"
EIGENE = PROJEKT / "einstellungen_eigene.yaml"


class Abschnitt(dict):
    """Ein Abschnitt der Einstellungen. Werte gehen per ``abschnitt.name`` oder ``abschnitt["name"]``."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Einstellung {name!r} fehlt in einstellungen.yaml") from None

    def kopie(self, **aenderungen) -> Abschnitt:
        """Kopie mit geänderten Werten - die Einstellungen selbst bleiben unberührt."""
        return Abschnitt({**self, **aenderungen})


def _wandeln(wert):
    if isinstance(wert, dict):
        return Abschnitt({k: _wandeln(v) for k, v in wert.items()})
    if isinstance(wert, list):
        return [_wandeln(v) for v in wert]
    return wert


def _lesen(pfad: Path) -> dict:
    return yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}


def laden(pfad: str | Path | None = None) -> Abschnitt:
    """Die Einstellungen lesen. Ohne Angabe: ``einstellungen.yaml`` mit den eigenen
    Abweichungen darüber (oder nur die Datei aus der Umgebungsvariable
    ``POREN_EINSTELLUNGEN``)."""
    pfad = pfad or os.environ.get("POREN_EINSTELLUNGEN")
    return _wandeln(_lesen(Path(pfad)) if pfad else roh())


# --------------------------------------------------------------------------------------
# Standard und eigene Abweichungen
# --------------------------------------------------------------------------------------


def standard() -> dict:
    """Der Standard aus ``einstellungen.yaml`` als einfaches dict."""
    return _lesen(DATEI)


def eigene() -> dict:
    """Nur die eigenen Abweichungen (leer, wenn es keine gibt)."""
    return _lesen(EIGENE) if EIGENE.is_file() else {}


def roh() -> dict:
    """Die geltenden Einstellungen als einfaches dict: Standard plus Abweichungen."""
    return _ueberlagern(standard(), eigene())


def speichern(werte: dict) -> dict:
    """Geltende Einstellungen ablegen. Gespeichert wird nur, was vom Standard abweicht;
    ohne Abweichung wird die eigene Datei gelöscht. Gibt die Abweichungen zurück."""
    basis = standard()
    abweichung = _abweichung(basis, angleichen(basis, werte))
    if not abweichung:
        EIGENE.unlink(missing_ok=True)
        return {}
    kopf = (
        "# Eigene Einstellungen - nur die Abweichungen von einstellungen.yaml.\n"
        "# Geschrieben vom Viewer (Menü Einstellungen). Löschen = Standard.\n"
        f"# Stand: {datetime.now():%d.%m.%Y %H:%M:%S}\n\n"
    )
    EIGENE.write_text(kopf + yaml.safe_dump(abweichung, allow_unicode=True, sort_keys=False,
                                            default_flow_style=None, width=100),
                      encoding="utf-8")
    return abweichung


def angleichen(vorlage, wert):
    """Typen an den Standard angleichen. Der Browser kennt nur "Zahl" - aus 4.0 wird
    dort 4, und ein Kernel aus der YAML soll eine ganze Zahl bleiben."""
    if isinstance(vorlage, dict) and isinstance(wert, dict):
        return {k: angleichen(vorlage.get(k), v) for k, v in wert.items()}
    if _ist_filterliste(vorlage) and _ist_filterliste(wert):
        nach_name = {f["name"]: f for f in vorlage}
        return [angleichen(nach_name.get(f["name"]), f) for f in wert]
    if isinstance(vorlage, bool) or isinstance(wert, bool):
        return wert
    if isinstance(vorlage, float) and isinstance(wert, (int, float)):
        return float(wert)
    if isinstance(vorlage, int) and isinstance(wert, float) and wert.is_integer():
        return int(wert)
    return wert


def _ueberlagern(basis: dict, oben: dict) -> dict:
    ergebnis = copy.deepcopy(basis)
    for schluessel, wert in oben.items():
        if isinstance(wert, dict) and isinstance(ergebnis.get(schluessel), dict):
            ergebnis[schluessel] = _ueberlagern(ergebnis[schluessel], wert)
        else:
            ergebnis[schluessel] = copy.deepcopy(wert)
    return ergebnis


def _abweichung(basis: dict, werte: dict) -> dict:
    """Was in ``werte`` anders ist als in ``basis`` - verschachtelt, Listen als Ganzes."""
    ergebnis = {}
    for schluessel, wert in werte.items():
        alt = basis.get(schluessel)
        if isinstance(wert, dict) and isinstance(alt, dict):
            unter = _abweichung(alt, wert)
            if unter:
                ergebnis[schluessel] = unter
        elif schluessel not in basis or not _gleich(alt, wert):
            ergebnis[schluessel] = wert
    return ergebnis


def _gleich(a, b) -> bool:
    """Gleichheit wie in der YAML gemeint: 4 und 4.0 sind derselbe Wert."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def unterschiede(vorher: dict, nachher: dict, pfad: str = "") -> list[str]:
    """Lesbare Liste der Änderungen, eine Zeile je Wert: ``pfad: alt -> neu``.

    Filterlisten werden über den Filternamen verglichen, nicht über die Position.
    """
    zeilen = []
    for schluessel in list(dict.fromkeys([*vorher, *nachher])):
        name = f"{pfad}.{schluessel}" if pfad else str(schluessel)
        alt, neu = vorher.get(schluessel), nachher.get(schluessel)
        if isinstance(alt, dict) and isinstance(neu, dict):
            zeilen += unterschiede(alt, neu, name)
        elif _ist_filterliste(alt) and _ist_filterliste(neu):
            alt_n = {f["name"]: f for f in alt}
            neu_n = {f["name"]: f for f in neu}
            for fname in dict.fromkeys([*alt_n, *neu_n]):
                zeilen += unterschiede(alt_n.get(fname, {}), neu_n.get(fname, {}),
                                       f"{name}.{fname}")
            if [f["name"] for f in alt] != [f["name"] for f in neu] and set(alt_n) == set(neu_n):
                zeilen.append(f"{name}: Reihenfolge geändert")
        elif not _gleich(alt, neu):
            zeilen.append(f"{name}: {_text(alt)} -> {_text(neu)}")
    return zeilen


def _ist_filterliste(wert) -> bool:
    return isinstance(wert, list) and all(isinstance(f, dict) and "name" in f for f in wert)


def _text(wert) -> str:
    if wert is None:
        return "-"
    return yaml.safe_dump(wert, default_flow_style=True, allow_unicode=True,
                          width=10_000).strip().removesuffix("...").strip()


def als_yaml(werte: dict) -> str:
    """Vollständige Einstellungen als YAML-Text (ohne Kommentare)."""
    return yaml.safe_dump(werte, allow_unicode=True, sort_keys=False,
                          default_flow_style=None, width=100)


# --------------------------------------------------------------------------------------
# Hilfetexte und Auswahllisten für das Menü - aus den Kommentaren der Standarddatei
# --------------------------------------------------------------------------------------

_SCHLUESSEL = re.compile(r"^(\s*)([A-Za-z_]\w*):(.*)$")
_FILTERZEILE = re.compile(r"^\s*-\s*\{\s*name:\s*(\w+)")
_AUSWAHL = re.compile(r"^\s*([\w-]+(?:\s*\|\s*[\w-]+)+)")
_AUSWAHL_ZEILE = re.compile(r"^([\w-]+)\s+=\s")
#: Ab dieser Spalte gilt eine reine Kommentarzeile als Fortsetzung des Kommentars davor.
_FORTSETZUNG_AB = 20


def _kommentar(rest: str) -> tuple[str, str | None]:
    """Wert und Kommentar einer Zeile trennen (ein ``#`` in Anführungszeichen zählt nicht)."""
    in_text = None
    for i, z in enumerate(rest):
        if z in "\"'" and in_text in (None, z):
            in_text = None if in_text else z
        elif z == "#" and in_text is None and (i == 0 or rest[i - 1].isspace()):
            return rest[:i].strip(), rest[i + 1:].strip()
    return rest.strip(), None


def hilfen() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Hilfetext je Einstellung (``"specimen.method"``) und Auswahllisten, gelesen aus
    den Kommentaren in ``einstellungen.yaml``. Filter stehen unter
    ``"analysis.filters.<name>"``."""
    texte: dict[str, list[str]] = {}
    stapel: list[tuple[int, str]] = []
    letzter: str | None = None
    for zeile in DATEI.read_text(encoding="utf-8").splitlines():
        if not zeile.strip():
            continue
        if zeile.lstrip().startswith("#"):
            spalte = len(zeile) - len(zeile.lstrip())
            if letzter and spalte >= _FORTSETZUNG_AB:
                texte[letzter].append(zeile.strip()[1:].strip())
            elif spalte < _FORTSETZUNG_AB:
                letzter = None
            continue
        if m := _FILTERZEILE.match(zeile):
            _, notiz = _kommentar(zeile)
            letzter = f"analysis.filters.{m.group(1)}"
            texte[letzter] = [notiz] if notiz else []
            continue
        m = _SCHLUESSEL.match(zeile)
        if not m:
            continue
        tiefe = len(m.group(1))
        while stapel and stapel[-1][0] >= tiefe:
            stapel.pop()
        stapel.append((tiefe, m.group(2)))
        letzter = ".".join(n for _, n in stapel)
        _, notiz = _kommentar(m.group(3))
        texte[letzter] = [notiz] if notiz else []

    hilfe = {k: "\n".join(v) for k, v in texte.items() if v}
    auswahl: dict[str, list[str]] = {}
    for k, zeilen in texte.items():
        if not zeilen:
            continue
        verbunden = " ".join(zeilen)
        if m := _AUSWAHL.match(verbunden):
            auswahl[k] = [o.strip() for o in m.group(1).split("|")]
        else:
            namen = [m.group(1) for z in zeilen if (m := _AUSWAHL_ZEILE.match(z))]
            if len(namen) >= 2:
                auswahl[k] = namen
    return hilfe, auswahl
