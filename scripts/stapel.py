"""Stapel: einen Ordner auswerten, die Ergebnisse ablegen, die Bilder fuer den Viewer halten.

Zwei Dinge gehoeren hier zusammen, weil sie dieselben Daten brauchen:

* **Der Stapellauf** wertet alle Bilder eines Ordners aus und schreibt die Ergebnisse
  in den Zielordner - ``poren.csv``, ``bilder.csv``, ``verworfen.csv`` und
  ``ergebnis.json`` ueber die vorhandenen Exporter aus ``poredet.io.exporters``.
* **Die Mappe** haelt die ausgewerteten Bilder fuer den Viewer bereit. Wird dort eine
  Pore korrigiert, rechnet die Mappe das Ergebnis dieses Bildes neu und schreibt die
  Dateien im Zielordner sofort nach. Was im Zielordner liegt, entspricht damit immer
  dem, was der Viewer zeigt.

Speicher: ein ausgewertetes Bild samt Ebenen belegt einige Megabyte. Die Mappe haelt
deshalb nur ``vorhalten`` Bilder im Speicher. Die uebrigen werden beim Blaettern neu
gerechnet, und das naechste Bild wird im Hintergrund schon vorbereitet, waehrend man
das aktuelle ansieht.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from gui_vorschau import Sitzung, rechnen, standard_konfiguration
from poredet.core.models import BatchResult, ImageResult
from poredet.core.pipeline import PoreDetectionPipeline
from poredet.io.exporters import csv_export, json_export
from poredet.io.image_reader import find_images

#: Rueckmeldung je Bild: (laufende Nummer ab 1, Gesamtzahl, Pfad, Ergebnis, Fehlertext).
Meldung = Callable[[int, int, Path, ImageResult | None, str | None], None]


def exportieren(batch: BatchResult, ziel: Path) -> list[Path]:
    """Die Ergebnisse eines Laufs in den Zielordner schreiben."""
    ziel.mkdir(parents=True, exist_ok=True)
    pfade = list(csv_export.write_all(batch, ziel))
    pfade.append(json_export.write(batch, ziel / "ergebnis.json"))
    return pfade


class Mappe:
    """Die ausgewerteten Bilder eines Laufs - fuer Export und Viewer."""

    def __init__(self, eingang: Path, ziel: Path | None = None, vorhalten: int = 12) -> None:
        self.eingang = Path(eingang)
        self.ziel = ziel
        self.pipeline = PoreDetectionPipeline(standard_konfiguration())
        self.bilder: list[Path] = []
        self.ergebnisse: list[ImageResult] = []
        self.fehler: list[tuple[Path, str]] = []
        self.begonnen = datetime.now()
        self.beendet: datetime | None = None

        self._vorhalten = max(1, vorhalten)
        self._sitzungen: OrderedDict[int, Sitzung] = OrderedDict()
        # Eine Sperre fuer alles Rechnen: die Pipeline ist nicht fuer paralleles
        # Arbeiten gebaut, und Vorladen und Anfrage duerfen sich nicht ueberholen.
        self._sperre = threading.RLock()
        self._exportsperre = threading.Lock()

    # -- Aufbauen ------------------------------------------------------------------------

    def aufnehmen(self, pfad: Path) -> ImageResult:
        """Ein Bild auswerten und aufnehmen. Wirft, wenn das Bild nicht auszuwerten ist."""
        with self._sperre:
            index = len(self.bilder)
            sitzung = self._sitzung_bauen(index, pfad)
            ergebnis = sitzung.ergebnis()
            self.bilder.append(pfad)
            self.ergebnisse.append(ergebnis)
            # Waehrend des Laufs werden nur die ersten Bilder gehalten, nicht die
            # letzten: der Viewer beginnt beim ersten.
            if len(self._sitzungen) < self._vorhalten:
                self._sitzungen[index] = sitzung
            return ergebnis

    def batch(self) -> BatchResult:
        cfg = self.pipeline.config
        return BatchResult(
            input_dir=self.eingang,
            results=list(self.ergebnisse),
            failures=list(self.fehler),
            started_at=self.begonnen,
            finished_at=self.beendet,
            config_summary={
                "specimen": cfg.specimen.method,
                "pore": cfg.pore.method,
                "filters": [f.name for f in cfg.analysis.filters if f.enabled],
            },
        )

    # -- Fuer den Viewer -----------------------------------------------------------------

    def anzahl(self) -> int:
        return len(self.bilder)

    def sitzung(self, index: int) -> Sitzung:
        """Die Sitzung eines Bildes - aus dem Speicher oder neu gerechnet."""
        with self._sperre:
            if index in self._sitzungen:
                self._sitzungen.move_to_end(index)
                return self._sitzungen[index]
            sitzung = self._sitzung_bauen(index, self.bilder[index])
            self._sitzungen[index] = sitzung
            while len(self._sitzungen) > self._vorhalten:
                self._sitzungen.popitem(last=False)
            return sitzung

    def vorladen(self, index: int) -> None:
        """Ein Bild im Hintergrund vorbereiten, damit das Blaettern nicht wartet."""
        if 0 <= index < len(self.bilder) and index not in self._sitzungen:
            threading.Thread(target=self._still_vorladen, args=(index,), daemon=True).start()

    def seitendaten(self, index: int) -> dict:
        """Die Daten der Viewer-Seite fuer ein Bild, samt Navigation."""
        daten = self.sitzung(index).ansicht()
        daten["navigation"] = {
            "index": index,
            "anzahl": len(self.bilder),
            "namen": [p.name for p in self.bilder],
        }
        return daten

    # -- Intern --------------------------------------------------------------------------

    def _sitzung_bauen(self, index: int, pfad: Path) -> Sitzung:
        return Sitzung(rechnen(pfad, pipeline=self.pipeline),
                       bei_aenderung=lambda s: self._geaendert(index, s))

    def _still_vorladen(self, index: int) -> None:
        try:
            self.sitzung(index)
        except Exception:  # noqa: BLE001 - beim echten Aufruf kommt der Fehler wieder
            pass

    def _geaendert(self, index: int, sitzung: Sitzung) -> None:
        """Nach einer Korrektur: Ergebnis dieses Bildes neu, Zielordner nachschreiben."""
        self.ergebnisse[index] = sitzung.ergebnis()
        if self.ziel is not None:
            with self._exportsperre:
                exportieren(self.batch(), self.ziel)


def verarbeiten(eingang: Path, ziel: Path, melden: Meldung | None = None,
                vorhalten: int = 12, rekursiv: bool = False) -> Mappe:
    """Alle Bilder unter ``eingang`` auswerten und die Ergebnisse nach ``ziel`` schreiben.

    Ein Bild, das scheitert, bricht den Lauf nicht ab - es landet mit Grund in
    ``Mappe.fehler`` und in den Exportdateien.
    """
    pfade = find_images(eingang, recursive=rekursiv)
    mappe = Mappe(eingang, ziel, vorhalten=vorhalten)
    for nummer, pfad in enumerate(pfade, start=1):
        ergebnis, fehler = None, None
        try:
            ergebnis = mappe.aufnehmen(pfad)
        except Exception as exc:  # noqa: BLE001 - ein Bild darf den Lauf nicht toeten
            fehler = f"{type(exc).__name__}: {exc}"
            mappe.fehler.append((pfad, fehler))
        if melden is not None:
            melden(nummer, len(pfade), pfad, ergebnis, fehler)
    mappe.beendet = datetime.now()
    exportieren(mappe.batch(), ziel)
    return mappe
