"""Lokaler Server fuer den Viewer - Bilder ansehen, blaettern und Poren bearbeiten.

Die statische HTML-Datei kann nichts auf die Platte schreiben. Fuer die Bearbeitung
laeuft die Seite deshalb ueber einen kleinen FastAPI-Server auf ``127.0.0.1``: jeder
Klick geht als Anfrage hinaus, der Server wendet die Korrektur an, speichert die
Korrekturdatei und schickt die neu berechneten Kennzahlen, Diagramme und Ebenen zurueck.

Gerechnet wird ausschliesslich in Python, mit denselben Funktionen wie fuer die
statische Seite. Der Browser zeichnet nur - es gibt keine zweite Implementierung der
Kennzahlen in JavaScript, die still von der ersten abweichen koennte.

Der Server kennt weder Pipeline noch Korrekturmodell. Er bekommt eine Quelle fuer die
aktuelle Mappe (siehe ``stapel.Mappe``) und eine Funktion, die aus Daten die Seite baut.
Die Quelle ist eine Funktion und kein festes Objekt: startet das Startfenster einen
neuen Lauf, zeigt der laufende Server ab dem naechsten Seitenaufruf dessen Bilder.
Das ist der Vorgriff auf die Oberflaeche aus P5 - dort kommen dieselben Aufrufe als
Routen in ``poredet.api`` hinein, an Mappe und Sitzung aendert sich nichts.

Adressen: ``/bild/<n>`` zeigt das n-te Bild (ab 0), ``/api/bild/<n>/...`` aendert es.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import webbrowser
from collections.abc import Callable
from typing import Protocol

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel


class Bearbeitbar(Protocol):
    """Was der Server von der Sitzung eines Bildes braucht - mehr nicht."""

    def plus(self, label: int) -> dict: ...
    def minus(self, label: int) -> dict: ...
    def zeichnen(self, punkte: list[list[float]]) -> dict: ...
    def rueckgaengig(self) -> dict: ...
    def zuruecksetzen(self) -> dict: ...


class Stapel(Protocol):
    """Was der Server von einer Mappe braucht."""

    def anzahl(self) -> int: ...
    def seitendaten(self, index: int) -> dict: ...
    def sitzung(self, index: int) -> Bearbeitbar: ...
    def vorladen(self, index: int) -> None: ...


class Pore(BaseModel):
    label: int


class Umriss(BaseModel):
    punkte: list[list[float]]


def _antwort(daten: dict) -> Response:
    """Direkt serialisiert statt ueber FastAPIs generischen Encoder.

    Die Antwort enthaelt die Umrisse aller Poren und wird bei jedem Klick geschickt -
    der generische Encoder laeuft rekursiv durch jede Koordinate und kostet dabei ein
    Vielfaches. NaN wird zu null: ``JSON.parse`` im Browser lehnt NaN ab.
    """
    return Response(json.dumps(_ohne_nan(daten), ensure_ascii=False),
                    media_type="application/json")


def _ohne_nan(wert):
    if isinstance(wert, float):
        return None if math.isnan(wert) or math.isinf(wert) else wert
    if isinstance(wert, dict):
        return {k: _ohne_nan(v) for k, v in wert.items()}
    if isinstance(wert, (list, tuple)):
        return [_ohne_nan(v) for v in wert]
    return wert


def app_bauen(quelle: Callable[[], Stapel], seite: Callable[[dict], str]) -> FastAPI:
    """Die Anwendung: je Bild eine Seite und fuenf Aenderungen."""
    app = FastAPI(title="Schliffbild-Inspektor", docs_url=None, redoc_url=None)
    # FastAPI fuehrt normale Funktionen parallel im Threadpool aus. Zwei schnelle
    # Klicks duerfen sich an der Sitzung nicht ueberholen.
    sperre = threading.Lock()

    def _mappe(index: int) -> Stapel:
        mappe = quelle()
        if not 0 <= index < mappe.anzahl():
            raise HTTPException(404, f"Kein Bild Nr. {index}")
        return mappe

    @app.get("/")
    def startseite() -> RedirectResponse:
        return RedirectResponse("/bild/0")

    @app.get("/bild/{index}", response_class=HTMLResponse)
    def bildseite(index: int) -> str:
        mappe = _mappe(index)
        with sperre:
            daten = mappe.seitendaten(index)
        daten["server"] = True
        # Waehrend man dieses Bild ansieht, das naechste schon rechnen.
        mappe.vorladen(index + 1)
        return seite(daten)

    def _aendern(index: int, aufruf: Callable[[Bearbeitbar], dict]) -> Response:
        mappe = _mappe(index)
        with sperre:
            return _antwort(aufruf(mappe.sitzung(index)))

    @app.post("/api/bild/{index}/plus")
    def plus(index: int, anfrage: Pore) -> Response:
        return _aendern(index, lambda s: s.plus(anfrage.label))

    @app.post("/api/bild/{index}/minus")
    def minus(index: int, anfrage: Pore) -> Response:
        return _aendern(index, lambda s: s.minus(anfrage.label))

    @app.post("/api/bild/{index}/zeichnen")
    def zeichnen(index: int, anfrage: Umriss) -> Response:
        return _aendern(index, lambda s: s.zeichnen(anfrage.punkte))

    @app.post("/api/bild/{index}/rueckgaengig")
    def rueckgaengig(index: int) -> Response:
        return _aendern(index, lambda s: s.rueckgaengig())

    @app.post("/api/bild/{index}/zuruecksetzen")
    def zuruecksetzen(index: int) -> Response:
        return _aendern(index, lambda s: s.zuruecksetzen())

    return app


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


def starten(quelle: Callable[[], Stapel], seite: Callable[[dict], str],
            port: int = 8765, oeffnen: bool = True, hintergrund: bool = False) -> str:
    """Server starten. Gibt die Adresse zurueck.

    Im Vordergrund blockiert der Aufruf bis Strg+C. Im Hintergrund laeuft der Server in
    einem eigenen Thread und endet mit dem Programm - so nutzt ihn das Startfenster.
    """
    port = freier_port(port)
    adresse = f"http://127.0.0.1:{port}/"
    server = uvicorn.Server(uvicorn.Config(
        app_bauen(quelle, seite), host="127.0.0.1", port=port, log_level="warning"))

    if oeffnen:
        # Erst oeffnen, wenn der Server lauscht - sonst zeigt der Browser eine
        # Fehlerseite und man muss neu laden.
        threading.Timer(1.0, webbrowser.open, args=(adresse,)).start()

    if hintergrund:
        threading.Thread(target=server.run, daemon=True).start()
    else:
        print(f"Viewer laeuft unter {adresse}  (beenden mit Strg+C)")
        server.run()
    return adresse
