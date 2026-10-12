"""Zusatzdaten für den Viewer - von außen eingehängt, nicht Teil der Pipeline.

Ein Startskript (z. B. ``start_with_zones.py``) trägt hier eine Funktion ein::

    from poren.viewer import zusaetze
    zusaetze.LIEFERANTEN["zonen"] = funktion      # funktion(ctx) -> dict | None

Sie bekommt den korrigierten ``PipelineContext`` und liefert ein JSON-taugliches dict.
Es geht bei jedem Bild und nach jeder Korrektur unter ``zusaetze.<name>`` an die Seite.
Die Seite zeigt nur, was sie kennt und was geliefert wird. Ohne Eintrag sieht der Viewer
aus wie immer.

Dieses Modul ist bewusst leicht (keine Imports von OpenCV & Co.), damit ein Startskript
sich eintragen kann, ohne das Startfenster zu verzögern.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable

LIEFERANTEN: dict[str, Callable[[object], dict | None]] = {}


def sammeln(ctx) -> dict:
    """Alle eingehängten Zusatzdaten. Ein Fehler kostet nur diesen Zusatz - die Seite
    bekommt dann ``{"ausfall": ...}`` statt der Daten."""
    daten = {}
    for name, lieferant in LIEFERANTEN.items():
        try:
            daten[name] = lieferant(ctx)
        except Exception as exc:  # noqa: BLE001 - ein Zusatz darf den Viewer nicht stören
            traceback.print_exc()
            daten[name] = {"ausfall": f"{type(exc).__name__}: {exc}"}
    return daten
