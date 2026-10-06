# Pore Detection

Findet und vermisst Poren in metallographischen Schliffbildern: Maßstab aus dem
eingebrannten Balken lesen, Probe vom Einbettmittel trennen, Poren suchen, vermessen,
filtern – und im Browser von Hand korrigieren.

## Einrichten (einmalig)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Starten

```powershell
.venv\Scripts\python.exe start.py
```

Einlesepfad (Bilderordner) und Zielpfad wählen, **Start**. Im Zielordner landen:

| Datei | Inhalt |
|---|---|
| `<bild>_ergebnis.png` | links das Original, rechts dasselbe mit den gezählten Poren **rot** |
| `poren.csv` | eine Zeile je gezählter Pore |
| `bilder.csv` | eine Zeile je Bild (Porosität, Porenzahl, Maßstab, …) |
| `verworfen.csv` | eine Zeile je verworfenem Objekt, mit Grund |
| `korrekturen/` | manuelle Korrekturen aus dem Viewer, eine JSON je Bild |

Ist **Viewer starten** angehakt, öffnet sich danach der Viewer im Browser. Blättern mit
den Pfeilen oben oder den Pfeiltasten. Werkzeuge links neben dem Bild:

| Werkzeug | Wirkung |
|---|---|
| **+** | Klick auf eine verworfene Pore zählt sie |
| **−** | Klick auf eine gezählte Pore verwirft sie; eine eingezeichnete wird gelöscht |
| **Stift** | Umriss einer übersehenen Pore mit gedrückter Maustaste ziehen |
| **Pfeil** | Rückgängig (auch `Strg+Z`) |
| **Papierkorb** | alle Korrekturen an diesem Bild verwerfen |

Jede Korrektur wird sofort im Zielordner unter `korrekturen/` gespeichert und schreibt CSVs und
Ergebnisbild im Zielordner nach. Das Startfenster offen lassen, solange der Viewer
gebraucht wird.

## Einstellungen

Alle Parameter stehen in **`einstellungen.yaml`**, jeder mit kurzer Erklärung. Wichtigste
Schalter:

* `specimen.method` – Einbettmittel-Trennung: **`grabcut`** (Standard), `lasso`,
  `random_walker`, `watershed`, `chan_vese`, `boundary_path`, `largest_region`, `full_frame`
* `pore.method` – Porendetektion: **`local_contrast`** (Standard), `threshold`, `adaptive`
* `analysis.filters` – welche Filter in welcher Reihenfolge greifen, an/aus und Grenzen

## Aufbau

```
start.py                 Startfenster (Einstieg)
einstellungen.yaml       alle Parameter
poren/
  pipeline.py            Ablauf je Bild: Maßstab -> Einbettmittel -> Poren -> Filter -> Korrektur
  massstab/              Maßstabsbalken finden und Beschriftung lesen (OCR)
  einbettmittel/         Probe vom Einbettmittel trennen (ein Verfahren je Datei)
  detektion/             Porendetektion (ein Verfahren je Datei) + Aufräumen/Trennen/Zusammenführen
  messung.py             Kennwerte je Pore
  filter.py              Filter: was als Pore zählt, mit Grund je verworfenem Objekt
  korrekturen.py         manuelle Korrekturen (entfernen, aufnehmen, einzeichnen)
  ausgabe.py             CSVs und Ergebnisbild
  auswertungen/          zusätzliche Auswertungsmodule (je Datei eins, werden automatisch gefunden)
  stapel.py              Ordner auswerten, Ergebnisse für den Viewer halten
  viewer/                Browser-Viewer: server.py, sitzung.py, seite.html
  modelle.py, einheiten.py, bild.py, verteilung.py, einstellungen.py   Hilfsmodule
```

Eine zusätzliche Auswertung, deren Ergebnis im Zielordner landet, ist eine Datei in
`poren/auswertungen/` – siehe [`AUSWERTUNG_MODULE.md`](AUSWERTUNG_MODULE.md).

Ein neues Verfahren für Einbettmittel oder Poren ist eine neue Datei im jeweiligen
Ordner plus ein Eintrag in dessen `__init__.py` – danach ist es per Namen in der
`einstellungen.yaml` wählbar.

Ohne erkannten Maßstab wird in Pixeln gemessen; die physikalischen Werte bleiben leer.
Findet sich kein Einbettmittel, gilt das ganze Bild als Probe.
