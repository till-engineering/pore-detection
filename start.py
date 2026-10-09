"""Startfenster - der eigentliche Einstieg ins Programm.

Einlesepfad und Zielpfad waehlen, Start druecken: alle Bilder des Ordners werden
ausgewertet, die Ergebnisse landen im Zielordner (``poren.csv``, ``bilder.csv``,
``verworfen.csv``, ``einstellungen_log.txt`` und je Bild ein Ordner ``<name>/`` mit
Ergebnisbild, Tabellen, Einstellungen und Log). Ist "Viewer starten" angehakt, oeffnet
sich nach dem Lauf der Viewer in einem eigenen Fenster - dort laesst sich zwischen den Bildern
blaettern, jede Pore von Hand korrigieren und ueber das Menue "Einstellungen" jeder
Parameter verstellen. Korrekturen im Viewer schreiben die Dateien im Zielordner sofort
nach.

Alle Parameter stehen in ``einstellungen.yaml`` (Standard); Abweichungen aus dem Viewer
in ``einstellungen_eigene.yaml``.

Tkinter statt einer weiteren Weboberflaeche: es gehoert zu Python, braucht nichts
Zusaetzliches und startet sofort. Die schweren Bibliotheken (OpenCV, OCR, ...) werden
erst beim Start des Laufs geladen, damit das Fenster ohne Wartezeit erscheint.

Die Auswertung laeuft in einem eigenen Thread; das Fenster bekommt ihre Meldungen ueber
eine Warteschlange und bleibt so bedienbar. Das Fenster offen lassen, solange der
Viewer gebraucht wird - mit dem Fenster endet auch der Viewer.

**Threads:** Das Viewer-Fenster (pywebview, ``poren/viewer/fenster.py``) muss im
Haupt-Thread laufen. Das Startfenster laeuft deshalb in einem eigenen Thread; alles, was
Tk betrifft, bleibt in diesem Thread. Der Viewer oeffnet keinen Port - er ist von aussen
nicht erreichbar.

Aufruf::

    .venv\\Scripts\\python.exe start.py
"""

from __future__ import annotations

import gc
import json
import os
import queue
import threading
import traceback
from pathlib import Path
from tkinter import BooleanVar, StringVar, TclError, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

WURZEL = Path(__file__).resolve().parent
EINSTELLUNGEN = WURZEL / "data" / "start_einstellungen.json"


class Startfenster:
    def __init__(self, root: Tk, viewer) -> None:
        self.root = root
        self.viewer_fenster = viewer  # poren.viewer.fenster.Viewer
        self.meldungen: queue.Queue = queue.Queue()
        self.mappe = None             # die zuletzt ausgewertete Mappe - Quelle des Viewers

        gemerkt = _laden()
        self.eingang = StringVar(value=gemerkt.get("eingang", ""))
        self.ziel = StringVar(value=gemerkt.get("ziel", ""))
        self.viewer = BooleanVar(value=gemerkt.get("viewer", True))
        self.status = StringVar(value="Bereit.")

        root.title("Pore Detection")
        root.minsize(620, 420)
        self._aufbauen()
        root.after(100, self._abholen)

    # -- Oberflaeche ---------------------------------------------------------------------

    def _aufbauen(self) -> None:
        rahmen = ttk.Frame(self.root, padding=16)
        rahmen.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        rahmen.columnconfigure(1, weight=1)

        ttk.Label(rahmen, text="Pore Detection", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        ttk.Label(rahmen, text="Einlesepfad").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(rahmen, textvariable=self.eingang).grid(
            row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(rahmen, text="Durchsuchen …",
                   command=lambda: self._ordner(self.eingang, "Ordner mit den Bildern")).grid(
            row=1, column=2, pady=3)

        ttk.Label(rahmen, text="Zielpfad").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(rahmen, textvariable=self.ziel).grid(
            row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(rahmen, text="Durchsuchen …",
                   command=lambda: self._ordner(self.ziel, "Ordner für die Ergebnisse")).grid(
            row=2, column=2, pady=3)

        zeile = ttk.Frame(rahmen)
        zeile.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        ttk.Checkbutton(zeile, text="Viewer starten", variable=self.viewer).pack(side="left")
        self.knopf_start = ttk.Button(zeile, text="Start", command=self._starten)
        self.knopf_start.pack(side="right")
        self.knopf_viewer = ttk.Button(zeile, text="Viewer öffnen", state="disabled",
                                       command=self._viewer_oeffnen)
        self.knopf_viewer.pack(side="right", padx=6)
        self.knopf_ziel = ttk.Button(zeile, text="Zielordner öffnen", state="disabled",
                                     command=self._ziel_oeffnen)
        self.knopf_ziel.pack(side="right")

        self.fortschritt = ttk.Progressbar(rahmen, mode="determinate")
        self.fortschritt.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 2))
        ttk.Label(rahmen, textvariable=self.status).grid(
            row=5, column=0, columnspan=3, sticky="w")

        self.protokoll = ScrolledText(rahmen, height=12, state="disabled",
                                      font=("Consolas", 9), relief="flat", borderwidth=1)
        self.protokoll.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        rahmen.rowconfigure(6, weight=1)

    def _ordner(self, variable: StringVar, titel: str) -> None:
        start = variable.get() or str(WURZEL)
        gewaehlt = filedialog.askdirectory(title=titel, initialdir=start, mustexist=False)
        if gewaehlt:
            variable.set(str(Path(gewaehlt)))

    def _schreiben(self, text: str) -> None:
        self.protokoll.configure(state="normal")
        self.protokoll.insert("end", text + "\n")
        self.protokoll.see("end")
        self.protokoll.configure(state="disabled")

    # -- Lauf ----------------------------------------------------------------------------

    def _starten(self) -> None:
        eingang = Path(self.eingang.get().strip())
        ziel_text = self.ziel.get().strip()
        if not eingang.is_dir():
            messagebox.showerror("Einlesepfad", f"Ordner nicht gefunden:\n{eingang}")
            return
        if not ziel_text:
            messagebox.showerror("Zielpfad", "Bitte einen Zielordner wählen.")
            return
        ziel = Path(ziel_text)

        _speichern({"eingang": str(eingang), "ziel": str(ziel), "viewer": self.viewer.get()})
        self.knopf_start.configure(state="disabled")
        self.knopf_ziel.configure(state="disabled")
        self.fortschritt.configure(value=0, maximum=1)
        self.status.set("Bibliotheken werden geladen …")
        self._schreiben(f"Einlesen: {eingang}\nZiel:     {ziel}")

        threading.Thread(target=self._arbeiten, args=(eingang, ziel), daemon=True).start()

    def _arbeiten(self, eingang: Path, ziel: Path) -> None:
        """Laeuft im Hintergrund. Spricht mit dem Fenster nur ueber die Warteschlange."""
        try:
            from poren import stapel  # erst hier: laedt OpenCV, OCR & Co.

            def melden(nummer, gesamt, pfad, ergebnis, fehler):
                self.meldungen.put(("bild", nummer, gesamt, pfad.name, ergebnis, fehler))

            self.meldungen.put(("status", "Bilder werden ausgewertet …"))
            mappe = stapel.verarbeiten(eingang, ziel, melden=melden)
            self.meldungen.put(("fertig", mappe))
        except Exception as exc:  # noqa: BLE001 - jeder Fehler gehoert ins Fenster
            self.meldungen.put(("fehler", f"{type(exc).__name__}: {exc}",
                                traceback.format_exc()))

    def _abholen(self) -> None:
        """Meldungen des Hintergrund-Threads ins Fenster uebernehmen."""
        try:
            while True:
                self._verarbeiten(self.meldungen.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._abholen)

    def _verarbeiten(self, meldung: tuple) -> None:
        art = meldung[0]
        if art == "status":
            self.status.set(meldung[1])
        elif art == "bild":
            _, nummer, gesamt, name, ergebnis, fehler = meldung
            self.fortschritt.configure(maximum=gesamt, value=nummer)
            self.status.set(f"Bild {nummer} von {gesamt}: {name}")
            self._schreiben(f"{nummer:>4}/{gesamt}  {name}: " + (
                f"FEHLER - {fehler}" if fehler else _kurz(ergebnis)))
        elif art == "fertig":
            self._fertig(meldung[1])
        elif art == "fehler":
            self._schreiben(meldung[2])
            self.status.set("Abgebrochen: " + meldung[1])
            self.knopf_start.configure(state="normal")
            messagebox.showerror("Fehler", meldung[1])
        elif art == "viewerfehler":
            self._schreiben("Viewer nicht geöffnet: " + meldung[1])
            messagebox.showerror("Viewer", meldung[1])

    def _fertig(self, mappe) -> None:
        self.mappe = mappe
        n, fehler = mappe.anzahl(), len(mappe.fehler)
        self.status.set(f"Fertig: {n} Bild(er) ausgewertet"
                        + (f", {fehler} gescheitert" if fehler else "")
                        + f" - Ergebnisse in {mappe.ziel}")
        self._schreiben(f"Fertig. Ergebnisse in {mappe.ziel}")
        if mappe.schreibfehler:
            self._schreiben("ACHTUNG - nicht alle Dateien geschrieben:\n  "
                            + "\n  ".join(mappe.schreibfehler))
            messagebox.showwarning(
                "Dateien nicht geschrieben",
                f"{len(mappe.schreibfehler)} Datei(en) ließen sich nicht schreiben - "
                "vermutlich in Excel geöffnet. Datei schließen und den Lauf "
                "wiederholen.\n\n"
                + "\n".join(mappe.schreibfehler[:5]))
        self.knopf_start.configure(state="normal")
        self.knopf_ziel.configure(state="normal")

        if n and self.viewer.get():
            self._viewer_starten()

    # -- Viewer --------------------------------------------------------------------------

    def _viewer_starten(self) -> None:
        self.knopf_viewer.configure(state="normal")
        self._viewer_oeffnen()

    def _viewer_oeffnen(self) -> None:
        # Im Hintergrund: das Viewer-Fenster kann beim ersten Mal noch im Aufbau sein,
        # und so lange soll das Startfenster nicht haengen.
        threading.Thread(target=self._viewer_zeigen, daemon=True).start()

    def _viewer_zeigen(self) -> None:
        try:
            self.viewer_fenster.zeigen()
        except Exception as exc:  # noqa: BLE001 - der Grund gehoert ins Fenster
            self.meldungen.put(("viewerfehler", f"{type(exc).__name__}: {exc}"))

    def _ziel_oeffnen(self) -> None:
        if self.mappe is not None and self.mappe.ziel is not None:
            os.startfile(self.mappe.ziel)  # noqa: S606 - Windows-Explorer oeffnen


def _kurz(ergebnis) -> str:
    teile = [f"{ergebnis.pore_count} Poren"]
    if ergebnis.porosity_pct is not None:
        teile.append(f"{ergebnis.porosity_pct:.2f} % Porosität")
    if ergebnis.scale is None:
        teile.append("ohne Maßstab")
    return ", ".join(teile)


def _laden() -> dict:
    try:
        return json.loads(EINSTELLUNGEN.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _speichern(werte: dict) -> None:
    try:
        EINSTELLUNGEN.parent.mkdir(parents=True, exist_ok=True)
        EINSTELLUNGEN.write_text(json.dumps(werte, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except OSError:
        pass  # Merken ist Komfort, kein Muss


def _oberflaeche(viewer, halter: dict) -> None:
    """Das Startfenster - laeuft in einem eigenen Thread, siehe oben."""
    root = Tk()
    halter["fenster"] = Startfenster(root, viewer)
    try:
        root.mainloop()
    finally:
        viewer.beenden()
        # Die Tk-Objekte hier abraeumen, im Thread, der sie angelegt hat. Raeumt sie
        # spaeter der Haupt-Thread ab, bricht Tcl mit "Tcl_AsyncDelete" ab.
        halter.clear()
        try:
            root.destroy()
        except TclError:
            pass  # schon zerstoert - der normale Fall nach dem Schliessen
        del root
        gc.collect()


def main() -> int:
    from poren.viewer.fenster import Viewer

    halter: dict = {}
    viewer = Viewer(lambda: halter["fenster"].mappe if "fenster" in halter else None)
    # Das Startfenster zuerst - es erscheint sofort, waehrend der Viewer im Hintergrund
    # (versteckt) aufgebaut wird.
    oberflaeche = threading.Thread(target=_oberflaeche, args=(viewer, halter),
                                   name="Startfenster")
    oberflaeche.start()
    try:
        viewer.ausfuehren()
    except Exception:  # noqa: BLE001 - das Startfenster arbeitet auch ohne Viewer
        traceback.print_exc()
    oberflaeche.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
