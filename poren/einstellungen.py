"""Einstellungen aus ``einstellungen.yaml`` laden.

Kein Schema: die YAML wird gelesen und ist dann per Punkt erreichbar, z. B.
``cfg.pore.method`` oder ``cfg.specimen.grabcut.iterations``. Jeder Wert, den der Code
braucht, muss in der Datei stehen - fehlt einer, sagt die Fehlermeldung welcher.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJEKT = Path(__file__).resolve().parents[1]
DATEI = PROJEKT / "einstellungen.yaml"


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


def laden(pfad: str | Path | None = None) -> Abschnitt:
    """Die Einstellungsdatei lesen. Ohne Angabe ``einstellungen.yaml`` im Projektordner
    (oder die Datei aus der Umgebungsvariable ``POREN_EINSTELLUNGEN``)."""
    pfad = Path(pfad or os.environ.get("POREN_EINSTELLUNGEN") or DATEI)
    return _wandeln(yaml.safe_load(pfad.read_text(encoding="utf-8")) or {})
