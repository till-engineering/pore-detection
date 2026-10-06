"""Stapel: einen Ordner auswerten, die Ergebnisse ablegen, die Bilder für den Viewer halten.

* **Der Stapellauf** wertet alle Bilder eines Ordners aus und schreibt in den Zielordner:
  ``poren.csv``, ``bilder.csv``, ``verworfen.csv`` und je Bild ``<name>_ergebnis.png``
  (links Original, rechts mit den gezählten Poren rot). Manuelle Korrekturen liegen
  unter ``korrekturen/`` im Zielordner, zusätzliche Auswertungen (``poren/auswertungen/``)
  unter ``auswertungen/``.
* **Die Mappe** hält die ausgewerteten Bilder für den Viewer bereit. Wird dort eine Pore
  korrigiert, rechnet sie das Ergebnis dieses Bildes neu und schreibt CSVs und
  Ergebnisbild sofort nach. Was im Zielordner liegt, entspricht immer dem Viewer.

Speicher: Im Arbeitsspeicher stehen nur die zuletzt angesehenen Bilder (``IM_SPEICHER``).
Jedes ausgewertete Bild wird außerdem komprimiert in einem temporären Ordner abgelegt -
beim Blättern wird es von dort geladen statt neu gerechnet.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from . import ausgabe, auswertungen
from .bild import find_images
from .modelle import BatchResult, ImageResult
from .pipeline import Pipeline
from .viewer.sitzung import Grundlage, Sitzung, rechnen, sichern, wiederherstellen

#: So viele Bilder hält die Mappe fertig im Speicher.
IM_SPEICHER = 3

#: Rückmeldung je Bild: (laufende Nummer ab 1, Gesamtzahl, Pfad, Ergebnis, Fehlertext).
Meldung = Callable[[int, int, Path, ImageResult | None, str | None], None]


class Mappe:
    """Die ausgewerteten Bilder eines Laufs - für Export und Viewer."""

    def __init__(self, eingang: Path, ziel: Path) -> None:
        self.eingang = Path(eingang)
        self.ziel = Path(ziel)
        self.pipeline = Pipeline()
        # Korrekturen liegen im Zielordner - zusammen mit den Ergebnissen, die sie betreffen.
        self.pipeline.config.corrections["directory"] = self.ziel / "korrekturen"
        self.bilder: list[Path] = []
        self.ergebnisse: list[ImageResult] = []
        self.fehler: list[tuple[Path, str]] = []
        #: Kennung des Laufs - der Browser hängt sie an Bildadressen (Cache).
        self.kennung = datetime.now().strftime("%Y%m%d%H%M%S")

        self._ablage = Path(tempfile.mkdtemp(prefix="poren_"))
        weakref.finalize(self, shutil.rmtree, self._ablage, ignore_errors=True)
        self._sitzungen: OrderedDict[int, Sitzung] = OrderedDict()
        # Eine Sperre für alles Rechnen: Vorladen und Anfrage dürfen sich nicht überholen.
        self._sperre = threading.RLock()

    # -- Aufbauen ------------------------------------------------------------------------

    def aufnehmen(self, pfad: Path) -> ImageResult:
        """Ein Bild auswerten, ablegen und exportieren. Wirft, wenn es nicht geht."""
        with self._sperre:
            index = len(self.bilder)
            grundlage = rechnen(pfad, self.pipeline)
            sichern(grundlage, self._datei(index))
            self.bilder.append(pfad)
            sitzung = self._merken(index, grundlage)
            ergebnis, ctx = sitzung.ergebnis()
            self.ergebnisse.append(ergebnis)
            self._ergebnisbild(sitzung, ctx)
            auswertungen.pro_bild(ctx, ergebnis, self.ziel)
            return ergebnis

    def batch(self) -> BatchResult:
        return BatchResult(input_dir=self.eingang, results=list(self.ergebnisse),
                           failures=list(self.fehler))

    def tabellen_schreiben(self) -> None:
        ausgabe.write_all(self.batch(), self.ziel)
        auswertungen.gesamt(self.ergebnisse, self.ziel)

    # -- Für den Viewer ------------------------------------------------------------------

    def anzahl(self) -> int:
        return len(self.bilder)

    def sitzung(self, index: int) -> Sitzung:
        """Die Sitzung eines Bildes - aus dem Speicher oder aus der Ablage geladen."""
        with self._sperre:
            if index in self._sitzungen:
                self._sitzungen.move_to_end(index)
                return self._sitzungen[index]
            grundlage = wiederherstellen(self._datei(index), self.bilder[index], self.pipeline)
            return self._merken(index, grundlage)

    def vorladen(self, index: int) -> None:
        """Ein Bild im Hintergrund laden, damit das Blättern nicht wartet."""
        if 0 <= index < len(self.bilder) and index not in self._sitzungen:
            threading.Thread(target=self._still_vorladen, args=(index,), daemon=True).start()

    # -- Intern --------------------------------------------------------------------------

    def _datei(self, index: int) -> Path:
        return self._ablage / f"{index:05d}.pkl.gz"

    def _merken(self, index: int, grundlage: Grundlage) -> Sitzung:
        sitzung = Sitzung(grundlage, bei_aenderung=lambda s: self._geaendert(index, s))
        self._sitzungen[index] = sitzung
        while len(self._sitzungen) > IM_SPEICHER:
            self._sitzungen.popitem(last=False)
        return sitzung

    def _still_vorladen(self, index: int) -> None:
        try:
            self.sitzung(index)
        except Exception:  # noqa: BLE001 - beim echten Aufruf kommt der Fehler wieder
            pass

    def _ergebnisbild(self, sitzung: Sitzung, ctx) -> None:
        bild = ctx.color if ctx.color is not None else ctx.gray
        ausgabe.write_ergebnisbild(self.ziel, sitzung.g.bildpfad, bild, ctx.labels,
                                   [p.label for p in ctx.pores])

    def _geaendert(self, index: int, sitzung: Sitzung) -> None:
        """Nach einer Korrektur: Ergebnis dieses Bildes neu, Zielordner nachschreiben."""
        ergebnis, ctx = sitzung.ergebnis()
        self.ergebnisse[index] = ergebnis
        self._ergebnisbild(sitzung, ctx)
        auswertungen.pro_bild(ctx, ergebnis, self.ziel)
        self.tabellen_schreiben()


def verarbeiten(eingang: Path, ziel: Path, melden: Meldung | None = None) -> Mappe:
    """Alle Bilder unter ``eingang`` auswerten und die Ergebnisse nach ``ziel`` schreiben.

    Ein Bild, das scheitert, bricht den Lauf nicht ab - es landet mit Grund in
    ``Mappe.fehler``.
    """
    pfade = find_images(eingang)
    mappe = Mappe(eingang, ziel)
    for nummer, pfad in enumerate(pfade, start=1):
        ergebnis, fehler = None, None
        try:
            ergebnis = mappe.aufnehmen(pfad)
        except Exception as exc:  # noqa: BLE001 - ein Bild darf den Lauf nicht töten
            fehler = f"{type(exc).__name__}: {exc}"
            mappe.fehler.append((pfad, fehler))
        if melden is not None:
            melden(nummer, len(pfade), pfad, ergebnis, fehler)
    mappe.tabellen_schreiben()
    return mappe
