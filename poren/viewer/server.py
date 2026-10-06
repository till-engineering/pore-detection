"""Lokaler Server für den Viewer - nur Python-Standardbibliothek.

Die Seite (``seite.html``) wird einmal geladen und holt sich dann alles einzeln:

    GET  /                               die Seite
    GET  /api/stapel                     Anzahl und Namen der Bilder
    GET  /api/bild/<n>                   Kennzahlen, Porenliste, Umrisse (JSON)
    GET  /api/bild/<n>/ebene/<name>.png  eine Bildebene (original, untergrund, kontrast,
                                         harz, ausschluss, kandidaten)
    POST /api/bild/<n>/<aktion>          plus | minus | zeichnen | rueckgaengig | zuruecksetzen
    GET  /api/einstellungen              geltende Werte, Standard, Hilfetexte, Auswahllisten
    POST /api/bild/<n>/neu_laden         Einstellungen speichern, Pipeline neu, Bild neu rechnen

Gerechnet wird ausschließlich in Python; der Browser zeichnet nur.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import traceback
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SEITE = Path(__file__).with_name("seite.html")

_BILD = re.compile(r"^/api/bild/(\d+)$")
_EBENE = re.compile(r"^/api/bild/(\d+)/ebene/(\w+)\.png$")
_AKTION = re.compile(r"^/api/bild/(\d+)/(plus|minus|zeichnen|rueckgaengig|zuruecksetzen)$")
_NEU_LADEN = re.compile(r"^/api/bild/(\d+)/neu_laden$")


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


def _handler(quelle: Callable[[], object]) -> type[BaseHTTPRequestHandler]:
    """Der Anfrage-Handler. ``quelle`` liefert die aktuelle Mappe (siehe ``stapel.Mappe``) -
    ein neuer Lauf im Startfenster ist so ohne Neustart des Servers zu sehen."""
    # Anfragen laufen parallel; zwei schnelle Klicks dürfen sich nicht überholen.
    sperre = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keine Zeile je Anfrage auf der Konsole
            pass

        # -- Antworten ---------------------------------------------------------------

        def _senden(self, status: int, inhalt: bytes, typ: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", typ)
            self.send_header("Content-Length", str(len(inhalt)))
            self.send_header("Cache-Control", "private, max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(inhalt)

        def _json(self, daten: object) -> None:
            self._senden(200, json.dumps(daten, ensure_ascii=False, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8"),
                         "application/json; charset=utf-8")

        def _fehler(self, status: int, text: str) -> None:
            self._senden(status, text.encode("utf-8"), "text/plain; charset=utf-8")

        def _mappe(self, index: int | None = None):
            mappe = quelle()
            if mappe is None:
                raise LookupError("Noch kein Lauf ausgewertet")
            if index is not None and not 0 <= index < mappe.anzahl():
                raise LookupError(f"Kein Bild Nr. {index}")
            return mappe

        # -- GET -----------------------------------------------------------------------

        def do_GET(self) -> None:
            pfad = self.path.split("?", 1)[0]
            try:
                if pfad in ("/", "/index.html"):
                    self._senden(200, SEITE.read_bytes(), "text/html; charset=utf-8")
                elif pfad == "/api/einstellungen":
                    self._json(_einstellungen())
                elif pfad == "/api/stapel":
                    mappe = self._mappe()
                    self._json({"anzahl": mappe.anzahl(), "kennung": mappe.kennung,
                                "namen": [p.name for p in mappe.bilder]})
                elif m := _BILD.match(pfad):
                    index = int(m.group(1))
                    mappe = self._mappe(index)
                    with sperre:
                        daten = mappe.sitzung(index).ansicht()
                    # Während man dieses Bild ansieht, das nächste schon laden.
                    mappe.vorladen(index + 1)
                    self._json(daten)
                elif m := _EBENE.match(pfad):
                    index = int(m.group(1))
                    png = self._mappe(index).sitzung(index).g.ebene(m.group(2))
                    self._senden(200, png, "image/png", cache=True)
                else:
                    self._fehler(404, "Nicht gefunden")
            except (LookupError, KeyError) as exc:
                self._fehler(404, str(exc))
            except Exception as exc:  # noqa: BLE001 - Fehler gehört in den Browser
                traceback.print_exc()
                self._fehler(500, f"{type(exc).__name__}: {exc}")

        # -- POST ----------------------------------------------------------------------

        def do_POST(self) -> None:
            pfad = self.path.split("?", 1)[0]
            if m := _NEU_LADEN.match(pfad):
                self._neu_laden(int(m.group(1)))
                return
            m = _AKTION.match(pfad)
            if not m:
                self._fehler(404, "Nicht gefunden")
                return
            try:
                koerper = self._koerper()
                index, aktion = int(m.group(1)), m.group(2)
                with sperre:
                    sitzung = self._mappe(index).sitzung(index)
                    if aktion in ("plus", "minus"):
                        daten = getattr(sitzung, aktion)(int(koerper["label"]))
                    elif aktion == "zeichnen":
                        daten = sitzung.zeichnen(koerper["punkte"])
                    else:
                        daten = getattr(sitzung, aktion)()
                self._json(daten)
            except LookupError as exc:
                self._fehler(404, str(exc))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._fehler(500, f"{type(exc).__name__}: {exc}")

        def _koerper(self) -> dict:
            laenge = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(laenge) or b"{}")

        def _neu_laden(self, index: int) -> None:
            """Knopf "Bild neu laden": Einstellungen speichern, Pipeline neu aufbauen und
            dieses Bild neu rechnen. Danach holt die Seite das Bild wie gewohnt."""
            try:
                werte = self._koerper().get("werte")
                with sperre:
                    aenderungen = self._mappe(index).neu_laden(index, werte)
                self._json({"aenderungen": aenderungen})
            except LookupError as exc:
                self._fehler(404, str(exc))
            except Exception as exc:  # noqa: BLE001 - der Grund gehört ins Menü
                traceback.print_exc()
                self._fehler(500, f"{type(exc).__name__}: {exc}")

    return Handler


def freier_port(ab: int = 8765, versuche: int = 20) -> int:
    """Der erste freie Port ab ``ab`` - ein zweiter Viewer soll nicht am ersten scheitern."""
    for port in range(ab, ab + versuche):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"Kein freier Port zwischen {ab} und {ab + versuche - 1}")


def starten(quelle: Callable[[], object], port: int = 8765, oeffnen: bool = True) -> str:
    """Server im Hintergrund starten (endet mit dem Programm). Gibt die Adresse zurück."""
    port = freier_port(port)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(quelle))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    adresse = f"http://127.0.0.1:{port}/"
    if oeffnen:
        webbrowser.open(adresse)
    return adresse
