"""Stapel: einen Ordner auswerten, die Ergebnisse ablegen, die Bilder für den Viewer halten.

* **Der Stapellauf** wertet alle Bilder eines Ordners aus. Im Zielordner liegen danach
  die Tabellen über alle Bilder (``poren.csv``, ``bilder.csv``, ``verworfen.csv``),
  ``einstellungen_log.txt`` und je Bild ein Ordner ``<bild>/`` mit Ergebnisbild, Maske,
  Histogramm, den Tabellen dieses Bildes, Poren und Einbettmittel als JSON (``daten/``),
  den manuellen Korrekturen, den verwendeten
  Einstellungen und einem ``log.txt``. Zusätzliche Auswertungen (``poren/auswertungen/``) schreiben nach
  ``auswertungen/``.
* **Die Mappe** hält die ausgewerteten Bilder für den Viewer bereit. Wird dort eine Pore
  korrigiert, rechnet sie das Ergebnis dieses Bildes neu und schreibt CSVs und
  Ergebnisbild sofort nach. Was im Zielordner liegt, entspricht immer dem Viewer.
* **Neue Einstellungen** (:meth:`Mappe.neu_laden`) bauen die Pipeline neu auf und rechnen
  das angezeigte Bild sofort neu. Die übrigen Bilder gelten als veraltet und werden neu
  gerechnet, sobald sie im Viewer geöffnet werden; bis dahin stehen in ihren Ordnern die
  Ergebnisse mit den Einstellungen, die dort in ``einstellungen.yaml`` liegen.

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

from . import ausgabe, auswertungen, protokoll
from . import einstellungen as einst
from .bild import find_images
from .korrekturen import CorrectionStore
from .modelle import BatchResult, ImageResult
from .pipeline import Pipeline
from .viewer.sitzung import Grundlage, Sitzung, rechnen, sichern, wiederherstellen

#: So viele Bilder hält die Mappe fertig im Speicher.
IM_SPEICHER = 3

#: Warnung an Bildern, die noch mit älteren Einstellungen gerechnet sind.
VERALTET = ("mit älteren Einstellungen gerechnet - im Viewer öffnen, dann wird es mit den "
            "aktuellen neu gerechnet")

#: Rückmeldung je Bild: (laufende Nummer ab 1, Gesamtzahl, Pfad, Ergebnis, Fehlertext).
Meldung = Callable[[int, int, Path, ImageResult | None, str | None], None]


class Mappe:
    """Die ausgewerteten Bilder eines Laufs - für Export und Viewer."""

    def __init__(self, eingang: Path, ziel: Path) -> None:
        self.eingang = Path(eingang)
        self.ziel = Path(ziel)
        #: Die geltenden Einstellungen als einfaches dict - für Logs und Viewer-Menü.
        self.einstellungen = einst.roh()
        self.pipeline = Pipeline(einst.laden())
        self.bilder: list[Path] = []
        self.ordner: list[Path] = []
        self.ergebnisse: list[ImageResult] = []
        self.fehler: list[tuple[Path, str]] = []
        #: Dateien, die sich nicht schreiben ließen (meist: in Excel geöffnet). Das
        #: Rechnen ist davon nicht betroffen - nur die Ablage ist unvollständig.
        self.schreibfehler: list[str] = []
        #: Kennung des Laufs - daran erkennt der Viewer einen neuen Lauf.
        self.kennung = datetime.now().strftime("%Y%m%d%H%M%S")

        self._ablage = Path(tempfile.mkdtemp(prefix="poren_"))
        weakref.finalize(self, shutil.rmtree, self._ablage, ignore_errors=True)
        self._sitzungen: OrderedDict[int, Sitzung] = OrderedDict()
        #: Bilder, die noch mit älteren Einstellungen gerechnet sind.
        self._veraltet: set[int] = set()
        # Eine Sperre für alles Rechnen: Vorladen und Anfrage dürfen sich nicht überholen.
        self._sperre = threading.RLock()

        protokoll.lauf(self.ziel, f"Lauf gestartet - Einlesepfad {self.eingang}",
                       self.einstellungen)

    # -- Aufbauen ------------------------------------------------------------------------

    def aufnehmen(self, pfad: Path) -> ImageResult:
        """Ein Bild auswerten, ablegen und exportieren. Wirft, wenn es nicht geht.

        Ins Verzeichnis der Mappe kommt das Bild erst, wenn sein Ergebnis feststeht -
        ``bilder``, ``ordner`` und ``ergebnisse`` laufen sonst auseinander, und jeder
        spätere Index zeigt auf das falsche Bild. Scheitert danach nur das Schreiben der
        Dateien, ist das Bild trotzdem ausgewertet; der Grund steht in seinen Warnungen.
        """
        with self._sperre:
            index = len(self.bilder)
            grundlage = rechnen(pfad, self.pipeline)
            sichern(grundlage, self._datei(index))
            self.bilder.append(pfad)
            self.ordner.append(self._neuer_ordner(pfad))
            try:
                sitzung = self._merken(index, grundlage)
                ergebnis, ctx = sitzung.ergebnis()
            except Exception:
                del self.bilder[index:], self.ordner[index:]
                self._sitzungen.pop(index, None)
                raise
            self.ergebnisse.append(ergebnis)
            self._bild_schreiben(index, sitzung, ctx, ergebnis, "ausgewertet")
            return ergebnis

    def batch(self) -> BatchResult:
        return BatchResult(input_dir=self.eingang, results=list(self.ergebnisse),
                           failures=list(self.fehler))

    def tabellen_schreiben(self) -> None:
        batch = self.batch()
        for datei, schreiben in (("poren.csv", ausgabe.write_pores),
                                 ("bilder.csv", ausgabe.write_images),
                                 ("verworfen.csv", ausgabe.write_rejected)):
            self._schreiben(datei, schreiben, batch, self.ziel / datei)
        auswertungen.gesamt(self.ergebnisse, self.ziel)

    def _schreiben(self, was: str, funktion: Callable, *argumente) -> bool:
        """Eine Datei schreiben, ohne dass ein Fehler den Lauf abbricht. Eine in Excel
        geöffnete CSV ist unter Windows gesperrt - das darf kein ausgewertetes Bild und
        keine Korrektur kosten."""
        try:
            funktion(*argumente)
            return True
        except OSError as exc:
            text = f"{was} nicht geschrieben ({type(exc).__name__}: {exc}) - in Excel geöffnet?"
            self.schreibfehler.append(text)
            print(text)
            return False

    # -- Für den Viewer ------------------------------------------------------------------

    def anzahl(self) -> int:
        return len(self.bilder)

    def sitzung(self, index: int) -> Sitzung:
        """Die Sitzung eines Bildes - aus dem Speicher, aus der Ablage oder, wenn es mit
        älteren Einstellungen gerechnet ist, neu gerechnet."""
        with self._sperre:
            if index in self._veraltet:
                return self._neu_rechnen(index, "neu ausgewertet (geänderte Einstellungen)")
            if index in self._sitzungen:
                self._sitzungen.move_to_end(index)
                return self._sitzungen[index]
            grundlage = wiederherstellen(self._datei(index), self.bilder[index], self.pipeline)
            return self._merken(index, grundlage)

    def vorladen(self, index: int) -> None:
        """Ein Bild im Hintergrund laden, damit das Blättern nicht wartet."""
        if 0 <= index < len(self.bilder) and index not in self._sitzungen:
            threading.Thread(target=self._still_vorladen, args=(index,), daemon=True).start()

    def neu_laden(self, index: int, werte: dict | None = None) -> list[str]:
        """Einstellungen übernehmen: speichern, Pipeline neu aufbauen, Bild ``index`` neu
        rechnen. Gibt die Änderungen gegenüber vorher zurück.

        Ohne ``werte`` wird nur neu gelesen - so greifen auch Änderungen, die von Hand in
        ``einstellungen.yaml`` gemacht wurden.
        """
        with self._sperre:
            vorher = self.einstellungen
            if werte is not None:
                einst.speichern(werte)
            neu = einst.roh()
            # Erst aufbauen, dann übernehmen: scheitert der Aufbau (z. B. unbekanntes
            # Verfahren), bleibt die alte Pipeline in Betrieb.
            self.pipeline = Pipeline(einst.laden())
            self.einstellungen = neu
            aenderungen = einst.unterschiede(vorher, neu)
            protokoll.lauf(self.ziel, "Einstellungen übernommen im Viewer "
                           f"(Bild {self.bilder[index].name})", neu, vorher)
            self._sitzungen.clear()
            if aenderungen:
                self._veraltet = set(range(len(self.bilder)))
                # Die Tabellen über alle Bilder mischen jetzt alte und neue Einstellungen.
                # Das muss dort sichtbar sein, bis das Bild neu gerechnet ist - dann
                # ersetzt das neue Ergebnis das alte samt dieser Warnung.
                for i in self._veraltet - {index}:
                    if VERALTET not in self.ergebnisse[i].warnings:
                        self.ergebnisse[i].warnings.append(VERALTET)
            self._neu_rechnen(index, "neu ausgewertet (Bild neu laden)")
            return aenderungen

    # -- Intern --------------------------------------------------------------------------

    def _datei(self, index: int) -> Path:
        return self._ablage / f"{index:05d}.pkl.gz"

    def _neuer_ordner(self, pfad: Path) -> Path:
        """Ordner für ein Bild: sein Name ohne Endung. Gleichnamige Bilder (etwa aus
        Unterordnern) bekommen eine Nummer angehängt."""
        vergeben = {o.name.lower() for o in self.ordner}
        name, nummer = pfad.stem, 2
        while name.lower() in vergeben:
            name, nummer = f"{pfad.stem}_{nummer}", nummer + 1
        return self.ziel / name

    def _merken(self, index: int, grundlage: Grundlage) -> Sitzung:
        # Korrekturen liegen im Ordner des Bildes; der frühere gemeinsame Ordner
        # korrekturen/ wird noch gelesen, damit alte Korrekturen nicht verloren gehen.
        ablage = CorrectionStore(self.ordner[index], alt=self.ziel / "korrekturen")
        sitzung = Sitzung(grundlage, ablage, bei_aenderung=lambda s: self._geaendert(index, s))
        self._sitzungen[index] = sitzung
        while len(self._sitzungen) > IM_SPEICHER:
            self._sitzungen.popitem(last=False)
        return sitzung

    def _neu_rechnen(self, index: int, ereignis: str) -> Sitzung:
        grundlage = rechnen(self.bilder[index], self.pipeline)
        sichern(grundlage, self._datei(index))
        self._veraltet.discard(index)
        sitzung = self._merken(index, grundlage)
        ergebnis, ctx = sitzung.ergebnis()
        self.ergebnisse[index] = ergebnis
        self._bild_schreiben(index, sitzung, ctx, ergebnis, ereignis)
        self.tabellen_schreiben()
        return sitzung

    def _still_vorladen(self, index: int) -> None:
        try:
            # Die Umrisse gleich mit - der Stapellauf rechnet sie nicht vor.
            self.sitzung(index).g.umrisse
        except Exception:  # noqa: BLE001 - beim echten Aufruf kommt der Fehler wieder
            pass

    def _bild_schreiben(self, index: int, sitzung: Sitzung, ctx, ergebnis: ImageResult,
                        ereignis: str | None) -> None:
        """Den Ordner eines Bildes schreiben. ``ereignis`` gesetzt heißt frisch gerechnet:
        dann kommen die Einstellungen dazu und ins Log, was vom Standard abweicht."""
        ordner = self.ordner[index]
        ordner.mkdir(parents=True, exist_ok=True)
        bild = ctx.color if ctx.color is not None else ctx.gray
        gezaehlt = [p.label for p in ctx.pores]
        name = ordner.name
        self._schreiben(f"{name}: Ergebnisbild", ausgabe.write_ergebnisbild,
                        ordner, sitzung.g.bildpfad, bild, ctx.labels, gezaehlt)
        self._schreiben(f"{name}: Maske", ausgabe.write_maske,
                        ordner, sitzung.g.bildpfad, ctx.specimen, ctx.labels, gezaehlt)
        self._schreiben(f"{name}: Histogramm", ausgabe.write_histogramm,
                        ordner, sitzung.g.bildpfad, ctx.pores, ctx.um_per_px)
        self._schreiben(f"{name}: Tabellen", ausgabe.write_bild, ordner, ergebnis)
        # Das Einbettmittel ändert sich durch eine Korrektur nicht - nur beim Auswerten.
        self._schreiben(f"{name}: Daten (JSON)", ausgabe.write_daten, ordner, ctx, ergebnis,
                        ereignis is not None)
        auswertungen.pro_bild(ctx, ergebnis, self.ziel)

        zeilen = ["Ergebnis: " + _kurz(ergebnis)]
        if sitzung.korrekturen:
            zeilen.append(f"Manuelle Korrekturen: {len(sitzung.korrekturen)}")
        if ereignis is None:
            protokoll.bild(ordner, "Korrektur gespeichert", zeilen)
            return
        protokoll.bild_einstellungen(ordner, self.einstellungen)
        abweichend = protokoll.abweichungen(self.einstellungen)
        zeilen = [f"Bild: {sitzung.g.bildpfad}", *zeilen]
        zeilen += (["Abweichungen vom Standard:"] + [f"  {z}" for z in abweichend]
                   if abweichend else ["Einstellungen: Standard (einstellungen.yaml)"])
        protokoll.bild(ordner, ereignis, zeilen)

    def _geaendert(self, index: int, sitzung: Sitzung) -> None:
        """Nach einer Korrektur: Ergebnis dieses Bildes neu, Zielordner nachschreiben."""
        ergebnis, ctx = sitzung.ergebnis()
        self.ergebnisse[index] = ergebnis
        self._bild_schreiben(index, sitzung, ctx, ergebnis, None)
        self.tabellen_schreiben()


def _kurz(ergebnis: ImageResult) -> str:
    teile = [f"{ergebnis.pore_count} Poren"]
    if ergebnis.porosity_pct is not None:
        teile.append(f"{ergebnis.porosity_pct:.3f} % Porosität".replace(".", ","))
    if ergebnis.scale is None:
        teile.append("ohne Maßstab")
    return ", ".join(teile)


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
