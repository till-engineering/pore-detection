"""Zusätzliche Auswertungen - einfach als Datei in diesen Ordner legen.

Jede ``.py``-Datei hier (außer solchen, die mit ``_`` beginnen) ist ein Auswertungsmodul.
Es kann zwei Funktionen haben, beide optional:

    def pro_bild(bild: BildDaten, ordner: Path) -> None
        läuft nach jedem ausgewerteten Bild - und erneut nach jeder Korrektur im Viewer

    def gesamt(bilder: list[ImageResult], ordner: Path) -> None
        läuft am Ende des Laufs über alle Bilder - und erneut nach jeder Korrektur

``ordner`` ist ``<Zielordner>/auswertungen/<modulname>/`` und existiert bereits. Ein Modul
schreibt nur dorthin. Ein Fehler in einem Modul bricht den Lauf nicht ab; er landet in
``<Zielordner>/auswertungen/fehler.txt``.

Vorlage: ``_vorlage.py``. Anleitung: ``AUSWERTUNG_MODULE.md`` im Projektordner.
Ausprobieren ohne GUI: ``python auswertung_testen.py <modulname> <bild>``.
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

from ..modelle import ImageResult, Pore, RejectedPore, ScaleInfo


@dataclass(frozen=True)
class BildDaten:
    """Alles, was ein Auswertungsmodul über ein Bild bekommt. Nur lesen, nicht ändern."""

    name: str                       # Dateiname, z. B. "probe_01.tif"
    pfad: Path                      # Pfad des Originalbilds
    ergebnis: ImageResult           # Kennzahlen des Bildes (Porosität, ...), wie in bilder.csv
    poren: list[Pore]               # gezählte Poren (nach Filtern und Korrekturen)
    verworfen: list[RejectedPore]   # verworfene Objekte samt Grund
    grau: np.ndarray                # uint8 (H, W) - Graubild
    farbe: np.ndarray | None        # uint8 (H, W, 3) BGR - nur bei Farbbildern, sonst None
    labels: np.ndarray              # int32 (H, W) - 0 = keine Pore, sonst Pore.label
    probe: np.ndarray               # bool (H, W) - auswertbare Probenfläche
    einbettmittel: np.ndarray       # bool (H, W) - erkanntes Einbettmittel
    overlay: np.ndarray             # bool (H, W) - Maßstabskasten (ausgeschlossen)
    massstab: ScaleInfo | None      # None = kein Maßstab erkannt
    um_pro_px: float | None         # Mikrometer je Pixel, None ohne Maßstab


ORDNERNAME = "auswertungen"
_module: list[ModuleType] | None = None


def module() -> list[ModuleType]:
    """Alle Auswertungsmodule dieses Ordners (einmal geladen, dann gemerkt)."""
    global _module
    if _module is None:
        _module = []
        for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
            if info.name.startswith("_"):
                continue
            modul = importlib.import_module(f"{__name__}.{info.name}")
            if getattr(modul, "AKTIV", True):
                _module.append(modul)
    return _module


def bilddaten(ctx, ergebnis: ImageResult) -> BildDaten:
    """Aus dem Pipeline-Kontext die Daten für die Module zusammenstellen."""
    return BildDaten(
        name=ctx.path.name, pfad=ctx.path, ergebnis=ergebnis,
        poren=list(ctx.pores), verworfen=list(ctx.rejected),
        grau=ctx.gray, farbe=ctx.color, labels=ctx.labels,
        probe=ctx.specimen, einbettmittel=ctx.resin, overlay=ctx.excluded,
        massstab=ctx.scale, um_pro_px=ctx.um_per_px,
    )


def pro_bild(ctx, ergebnis: ImageResult, ziel: Path) -> None:
    """``pro_bild`` aller Module für ein Bild aufrufen."""
    daten = None
    for modul in module():
        if hasattr(modul, "pro_bild"):
            daten = daten or bilddaten(ctx, ergebnis)
            _ausfuehren(modul, "pro_bild", ziel, daten)


def gesamt(ergebnisse: list[ImageResult], ziel: Path) -> None:
    """``gesamt`` aller Module über alle Bilder aufrufen."""
    for modul in module():
        if hasattr(modul, "gesamt"):
            _ausfuehren(modul, "gesamt", ziel, list(ergebnisse))


def _ausfuehren(modul: ModuleType, funktion: str, ziel: Path, daten) -> None:
    name = modul.__name__.rsplit(".", 1)[-1]
    ordner = Path(ziel) / ORDNERNAME / name
    try:
        ordner.mkdir(parents=True, exist_ok=True)
        getattr(modul, funktion)(daten, ordner)
    except Exception:  # noqa: BLE001 - ein Modul darf den Lauf nicht abbrechen
        text = traceback.format_exc()
        print(f"Auswertung {name}.{funktion} gescheitert:\n{text}")
        datei = Path(ziel) / ORDNERNAME / "fehler.txt"
        datei.parent.mkdir(parents=True, exist_ok=True)
        with datei.open("a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now():%d.%m.%Y %H:%M:%S}  {name}.{funktion}\n{text}\n")
