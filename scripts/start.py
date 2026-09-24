"""Startfenster - der eigentliche Einstieg ins Programm.

Einlesepfad und Zielpfad waehlen, Start druecken: alle Bilder des Ordners werden
ausgewertet, die Ergebnisse landen im Zielordner (``poren.csv``, ``bilder.csv``,
``verworfen.csv``, ``ergebnis.json``). Ist "Viewer starten" angehakt, oeffnet sich nach
dem Lauf der Viewer im Browser - dort laesst sich zwischen den Bildern blaettern und
jede Pore von Hand korrigieren. Korrekturen im Viewer schreiben die Dateien im
Zielordner sofort nach.

Tkinter statt einer weiteren Weboberflaeche: es gehoert zu Python, braucht nichts
Zusaetzliches und startet sofort. Die schweren Bibliotheken (OpenCV, OCR, ...) werden
erst beim Start des Laufs geladen, damit das Fenster ohne Wartezeit erscheint.

Die Auswertung laeuft in einem eigenen Thread; das Fenster bekommt ihre Meldungen ueber
eine Warteschlange und bleibt so bedienbar. Das Fenster offen lassen, solange der
Viewer gebraucht wird - mit dem Fenster endet auch der Viewer.

Aufruf::

    .venv\\Scripts\\python.exe scripts/start.py
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

WURZEL = Path(__file__).resolve().parents[1]
EINSTELLUNGEN = WURZEL / "data" / "start_einstellungen.json"


class Startfenster:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.meldungen: queue.Queue = queue.Queue()
        self.mappe = None             # die zuletzt ausgewertete Mappe - Quelle des Viewers
        self.viewer_adresse: str | None = None

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
            import stapel  # erst hier: laedt OpenCV, OCR & Co.

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

    def _fertig(self, mappe) -> None:
        self.mappe = mappe
        n, fehler = mappe.anzahl(), len(mappe.fehler)
        self.status.set(f"Fertig: {n} Bild(er) ausgewertet"
                        + (f", {fehler} gescheitert" if fehler else "")
                        + f" - Ergebnisse in {mappe.ziel}")
        self._schreiben(f"Fertig. Ergebnisse in {mappe.ziel}")
        self.knopf_start.configure(state="normal")
        self.knopf_ziel.configure(state="normal")

        if n and self.viewer.get():
            self._viewer_starten()

    # -- Viewer --------------------------------------------------------------------------

    def _viewer_starten(self) -> None:
        if self.viewer_adresse is None:
            import gui_server
            from gui_vorschau import seite_bauen

            # Der Server fragt bei jedem Aufruf nach der aktuellen Mappe - ein neuer
            # Lauf ist damit ohne Neustart des Servers zu sehen.
            self.viewer_adresse = gui_server.starten(
                lambda: self.mappe, seite_bauen, oeffnen=False, hintergrund=True)
            self._schreiben(f"Viewer läuft unter {self.viewer_adresse} "
                            "- dieses Fenster offen lassen.")
        self.knopf_viewer.configure(state="normal")
        # Kurz warten, bis der Server lauscht.
        self.root.after(1200, self._viewer_oeffnen)

    def _viewer_oeffnen(self) -> None:
        if self.viewer_adresse:
            webbrowser.open(self.viewer_adresse)

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


def main() -> int:
    # Die Nachbarmodule (stapel, gui_vorschau, gui_server) liegen neben diesem Skript.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    root = Tk()
    Startfenster(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
