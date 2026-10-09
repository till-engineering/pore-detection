"""Der Viewer als eigenes Fenster (pywebview) - ohne Webserver und ohne offenen Port.

Die Seite (``seite.html``) wird als Text in das Fenster geladen, nicht über eine Adresse.
Sie spricht mit Python ausschließlich über die Nachrichtenschnittstelle von WebView2
(``window.pywebview.api``) - das ist eine Verbindung innerhalb des Prozesses. Es gibt
keinen Port, den ein anderer Rechner, ein anderes Programm oder eine Webseite im Browser
erreichen könnte.

Was die Seite aufrufen darf, steht in :func:`_schnittstelle`::

    stapel()                        Anzahl und Namen der Bilder
    bild(n)                         Kennzahlen, Porenliste, Umrisse
    ebene(n, name)                  eine Bildebene als data:-URL (original, untergrund,
                                    kontrast, harz, ausschluss, kandidaten)
    aktion(n, aktion, koerper)      plus | minus | zeichnen | bereich | rueckgaengig | zuruecksetzen
    einstellungen()                 geltende Werte, Standard, Hilfetexte, Auswahllisten
    neu_laden(n, werte)             Einstellungen speichern, Pipeline neu, Bild neu rechnen

Gerechnet wird ausschließlich in Python; die Seite zeichnet nur.

**Threads:** pywebview muss im Haupt-Thread laufen (:meth:`Viewer.ausfuehren`). Das
Startfenster läuft deshalb in einem eigenen Thread und ruft nur :meth:`Viewer.zeigen` und
:meth:`Viewer.beenden` auf - beide dürfen aus jedem Thread kommen.
"""

from __future__ import annotations

import base64
import re
import threading
from collections.abc import Callable
from pathlib import Path

SEITE = Path(__file__).with_name("seite.html")
#: Die Schriften liegen beim Viewer und werden beim Laden in die Seite eingesetzt.
SCHRIFTEN = Path(__file__).with_name("schriften")
TITEL = "Pore Detection – Viewer"

_SCHRIFT = re.compile(r"url\(/schriften/([\w-]+\.woff2)\)")
_AKTIONEN = frozenset({"plus", "minus", "zeichnen", "bereich", "rueckgaengig", "zuruecksetzen"})
#: Solange noch kein Lauf da ist, zeigt das (versteckte) Fenster nur das.
_LEER = "<!doctype html><meta charset='utf-8'><title>Viewer</title>"


def _seite() -> str:
    """Die Seite mit eingesetzten Schriften. Eine als Text geladene Seite hat keine
    Adresse, von der sie Dateien nachladen könnte - also kommt alles mit."""
    def einsetzen(m: re.Match) -> str:
        daten = base64.b64encode((SCHRIFTEN / m.group(1)).read_bytes()).decode("ascii")
        return f"url(data:font/woff2;base64,{daten})"

    return _SCHRIFT.sub(einsetzen, SEITE.read_text(encoding="utf-8"))


def _einstellungen() -> dict:
    """Alles, was das Einstellungsmenü braucht."""
    from .. import einstellungen as einst
    from .. import filter as filtermodul

    hilfe, auswahl = einst.hilfen()
    standard = einst.standard()
    for f in standard.get("analysis", {}).get("filters", []):
        schluessel = f"analysis.filters.{f['name']}"
        teile = [filtermodul.beschreibung(f["name"]), hilfe.get(schluessel, "")]
        hilfe[schluessel] = "\n".join(t for t in teile if t)
    return {"werte": einst.roh(), "standard": standard, "hilfe": hilfe, "auswahl": auswahl,
            "ganzzahl": _ganzzahlig(standard),
            "eigene_datei": einst.EIGENE.name, "eigene_vorhanden": einst.EIGENE.is_file()}


def _ganzzahlig(wert, pfad: str = "") -> list[str]:
    """Schlüssel der Werte, die im Standard ganze Zahlen sind - der Browser kann 4 und
    4.0 nicht unterscheiden. Filter stehen unter ihrem Namen, nicht ihrer Position."""
    if isinstance(wert, dict):
        return [s for k, v in wert.items() for s in _ganzzahlig(v, f"{pfad}.{k}" if pfad else k)]
    if isinstance(wert, list):
        return [s for v in wert if isinstance(v, dict) and "name" in v
                for s in _ganzzahlig(v, f"{pfad}.{v['name']}")]
    if isinstance(wert, int) and not isinstance(wert, bool):
        return [pfad]
    return []


def _schnittstelle(quelle: Callable[[], object]) -> object:
    """Die Funktionen, die die Seite aufrufen darf - und nichts sonst.

    pywebview reicht jedes öffentliche Attribut der Schnittstelle an die Seite weiter,
    auch verschachtelt. Die Klasse hat deshalb nur Methoden; ``quelle`` und die Sperre
    stecken in der Closure. ``quelle`` liefert die aktuelle Mappe (``stapel.Mappe``) -
    ein neuer Lauf im Startfenster ist so ohne Neustart zu sehen.
    """
    # Aufrufe laufen parallel; zwei schnelle Klicks dürfen sich nicht überholen.
    sperre = threading.Lock()

    def mappe(index=None):
        m = quelle()
        if m is None:
            raise LookupError("Noch kein Lauf ausgewertet")
        if index is not None and not 0 <= index < m.anzahl():
            raise LookupError(f"Kein Bild Nr. {index}")
        return m

    class Schnittstelle:
        def stapel(self) -> dict:
            m = mappe()
            return {"anzahl": m.anzahl(), "kennung": m.kennung,
                    "namen": [p.name for p in m.bilder]}

        def bild(self, index) -> dict:
            index = int(index)
            m = mappe(index)
            with sperre:
                daten = m.sitzung(index).ansicht()
            # Während man dieses Bild ansieht, das nächste schon laden.
            m.vorladen(index + 1)
            return daten

        def ebene(self, index, name) -> str:
            index = int(index)
            png = mappe(index).sitzung(index).g.ebene(str(name))
            return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

        def aktion(self, index, aktion, koerper) -> dict:
            index, aktion = int(index), str(aktion)
            if aktion not in _AKTIONEN:
                raise LookupError(f"Unbekannte Aktion {aktion!r}")
            koerper = koerper or {}
            with sperre:
                sitzung = mappe(index).sitzung(index)
                if aktion in ("plus", "minus"):
                    return getattr(sitzung, aktion)(int(koerper["label"]))
                if aktion in ("zeichnen", "bereich"):
                    return getattr(sitzung, aktion)(koerper["punkte"])
                return getattr(sitzung, aktion)()

        def einstellungen(self) -> dict:
            return _einstellungen()

        def neu_laden(self, index, werte) -> dict:
            """Knopf "Bild neu laden": Einstellungen speichern, Pipeline neu aufbauen und
            dieses Bild neu rechnen. Danach holt die Seite das Bild wie gewohnt."""
            index = int(index)
            with sperre:
                return {"aenderungen": mappe(index).neu_laden(index, werte)}

    return Schnittstelle()


class Viewer:
    """Das Viewer-Fenster. Es wird beim Programmstart versteckt angelegt, mit
    :meth:`zeigen` sichtbar und beim Schließen nur wieder versteckt - beendet wird es
    erst mit dem Startfenster (:meth:`beenden`)."""

    def __init__(self, quelle: Callable[[], object], beenden_beim_schliessen: bool = False) -> None:
        """``beenden_beim_schliessen``: Schließen beendet den Viewer, statt ihn zu
        verstecken - für einen Aufruf ohne Startfenster (``synthetik_testen.py``)."""
        self._quelle = quelle
        self._beenden_beim_schliessen = beenden_beim_schliessen
        self._fenster = None
        self._bereit = threading.Event()
        self._ende = False
        self._kennung = None          # Lauf, dessen Seite gerade geladen ist
        self._fehler: str | None = None

    def ausfuehren(self) -> None:
        """Im Haupt-Thread aufrufen. Kehrt zurück, wenn :meth:`beenden` gerufen wurde."""
        try:
            import webview

            fenster = webview.create_window(
                TITEL, html=_LEER, js_api=_schnittstelle(self._quelle), hidden=True,
                width=1500, height=950, min_size=(900, 600))
            fenster.events.closing += self._schliessen
            self._fenster = fenster
        except Exception as exc:
            self._fehler = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._bereit.set()
        if self._ende:
            return
        # Ausdrücklich: kein eingebauter HTTP-Server, keine Entwicklerwerkzeuge, nichts
        # auf der Platte (Cookies, Speicher) - und die Edge-Engine statt eines Rückfalls
        # auf den Internet Explorer, der die Seite nicht darstellen könnte.
        webview.start(gui="edgechromium", http_server=False, debug=False, private_mode=True)

    def zeigen(self) -> None:
        """Fenster zeigen; nach einem neuen Lauf mit frisch geladener Seite."""
        if not self._bereit.wait(60) or self._fenster is None:
            raise RuntimeError(self._fehler or "Viewer-Fenster nicht gestartet")
        mappe = self._quelle()
        kennung = getattr(mappe, "kennung", None)
        if kennung != self._kennung:
            # Erst wenn die leere Startseite geladen ist: WebView2 lädt sie, sobald es
            # bereit ist - und überschriebe damit eine Seite, die vorher kam. Das Fenster
            # bliebe weiß. (Im Startfenster liegt der Lauf dazwischen, im Testskript nicht.)
            self._fenster.events.loaded.wait(60)
            self._fenster.load_html(_seite())
            self._kennung = kennung
        self._fenster.show()
        self._fenster.restore()

    def beenden(self) -> None:
        self._ende = True
        if self._fenster is not None:
            self._fenster.destroy()

    def _schliessen(self):
        """Schließen heißt verstecken - die Mappe und das Fenster bleiben, bis das
        Startfenster endet. ``False`` bricht das Schließen ab."""
        if self._ende or self._beenden_beim_schliessen:
            self._ende = True
            return True
        threading.Thread(target=self._fenster.hide, daemon=True).start()
        return False
