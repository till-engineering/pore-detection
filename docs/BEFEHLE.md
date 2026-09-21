# Befehle — die Pipeline starten

Alle Aufrufe laufen in der Projekt-`.venv` und werden aus dem Projektordner
`E:\Arbeit\Werkstudent SEW\Programme\Pore_Detection_V2` gestartet.
Die Reihenfolge der Stufen ist fest verdrahtet (`src/poredet/core/pipeline.py`):

```
1. Maßstab erkennen         scale/      -> µm/px + Lage des Overlays
2. Overlay ausschließen     scale/      -> dieser Bereich ist kein Gefüge
3. Einbettmittel entfernen  specimen/   -> Probenmaske = Nenner der Porosität
4. Poren suchen + messen    detection/, measurement/, analysis/
```

---

## 0. Umgebung

Einmalig einrichten:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional, zweite OCR-Meinung (braucht eine installierte Tesseract-Binary):

```powershell
python -m pip install -e ".[tesseract]"
```

In jeder neuen Shell nur noch aktivieren:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ohne Aktivieren geht jeder Befehl auch direkt über den Interpreter der venv:

```powershell
.\.venv\Scripts\python.exe -m poredet --version
```

---

## 1. Die vollständige Pipeline — `run`

Das ist der Befehl, der alles zusammen laufen lässt.

```powershell
python -m poredet run "test\Pore_detection_test" -o data\runs\poren_kontrollblatt.png
```

Mit allen Ausgaben — Kontrollblatt, CSV-Tabellen und JSON-Ergebnisbaum:

```powershell
python -m poredet run "test\Pore_detection_test" `
    -o data\runs\poren_kontrollblatt.png `
    --csv data\runs\poren `
    --json data\runs\poren\ergebnis.json
```

Ein einzelnes Bild statt eines Ordners:

```powershell
python -m poredet run "test\GUI_Test\gas-porosity.tif" -o data\runs\einzel.png
```

### Optionen von `run`

| Option | Wirkung |
|---|---|
| `-r`, `--recursive` | Unterordner mitnehmen |
| `-o`, `--sheet PFAD` | Kontrollblatt als PNG (Original, Maske, gefundene Poren, verworfene) |
| `--csv ORDNER` | schreibt `poren.csv`, `bilder.csv` und die Verwerfungsliste |
| `--json PFAD` | vollständiger Ergebnisbaum maschinenlesbar |
| `--pore-method NAME` | `local_contrast` (Standard) \| `threshold` \| `adaptive` |
| `--specimen-method NAME` | `grabcut` (Standard) \| `lasso` \| `random_walker` \| `watershed` \| `chan_vese` \| `boundary_path` |
| `-v`, `--verbose` | je Bild die Zwischenmeldung jeder Stufe |

`-v` steht **vor** dem Unterbefehl:

```powershell
python -m poredet -v run "test\Pore_detection_test" -o data\runs\poren_kontrollblatt.png
```

Auf der Konsole steht danach je Bild: Porenzahl, Porosität, größte Pore,
Porendichte je mm² und die Zahl der von den Filtern verworfenen Kandidaten,
darunter eine Aufschlüsselung, *welcher* Filter wie viel verworfen hat.

---

## 2. Einzelne Stufen prüfen

Die Stufen lassen sich auch für sich laufen lassen — das ist der Weg, wenn
`run` ein Bild verhaut und die Frage ist, *welche* Stufe schuld war.

### Maßstab (Stufe 1 + 2)

```powershell
python -m poredet scale "test\Maßstäbe_Bilder" -o data\runs\scale_uebersicht.png --json data\runs\scale.json
```

Liegt im TIFF eine ImageJ-Kalibrierung, wird zusätzlich die Abweichung dagegen
ausgewiesen; `--no-reference` schaltet diese Gegenprobe ab.
Rückgabewert `1`, wenn nicht in jedem Bild ein Maßstab gefunden wurde.

### Einbettmittel (Stufe 3)

```powershell
python -m poredet specimen "test\Einbett_Material" -o data\runs\specimen_uebersicht.png --json data\runs\specimen.json
```

Masken zusätzlich als PNG ablegen:

```powershell
python -m poredet specimen "test\Einbett_Material" --masks data\runs\masken
```

Ein anderes Verfahren erzwingen (Standard ist `grabcut`):

```powershell
python -m poredet specimen "test\Einbett_Material" --method lasso -o data\runs\specimen_lasso.png
```

### Trennverfahren nebeneinander

Alle sechs Segmentierer auf denselben Signalen, damit der Unterschied wirklich
aus dem Verfahren stammt:

```powershell
python -m poredet specimen-compare "test\Einbett_Material" `
    -o data\runs\verfahrensvergleich.png `
    --json data\runs\verfahrensvergleich.json
```

Nur eine Auswahl vergleichen:

```powershell
python -m poredet specimen-compare "test\Einbett_Material" --methods grabcut,lasso -o data\runs\vergleich_zwei.png
```

---

## 3. Sichtprüfung am Einzelbild — `gui_vorschau.py`

Erzeugt eine eigenständige HTML-Seite mit umschaltbaren Bildebenen. Kein Server,
kein Netz — die Datei öffnet sich nach dem Lauf von selbst im Browser.

```powershell
.venv\Scripts\python.exe scripts\gui_vorschau.py
```

Ausgewertet wird dabei das (zuletzt geänderte) Bild aus `test\GUI_Test`.
Anderes Bild bzw. anderer Ordner:

```powershell
.venv\Scripts\python.exe scripts\gui_vorschau.py --bild "test\GUI_Test\gas-porosity.tif"
.venv\Scripts\python.exe scripts\gui_vorschau.py --ordner "test\Pore_detection_test"
```

| Option | Wirkung |
|---|---|
| `--ordner PFAD` | Ordner mit dem auszuwertenden Bild (Standard `test/GUI_Test`) |
| `--bild PFAD` | einzelnes Bild statt des Ordners |
| `-o`, `--out PFAD` | Ziel der HTML-Datei (Standard `data/runs/gui/index.html`) |
| `--nicht-oeffnen` | Seite nur schreiben, Browser nicht starten |

### Eine einzelne Pore ansehen

Auf eine Pore im Bild oder in der Liste **zeigen** wählt sie aus, ein **Klick** hält
sie fest — ein zweiter Klick oder `Esc` löst wieder. Ohne das Festhalten wäre die
Lupe nicht zu bedienen: der Weg der Maus zum Bedienfeld führt über andere Poren.

Die ausgewählte Pore steht dann vergrößert in der Karte *Ausgewählte Pore* — dasselbe
Mittel wie beim Maßstab, einzelne Bildpixel als Kästchen. Der Regler **Umfeld** stellt
ein, wie viel Umgebung dazugehört (1,5× bis 12× der Porenausdehnung); die Fußzeile
nennt die tatsächliche Vergrößerung und die Kantenlänge des Ausschnitts. Der
gestrichelte Kreis ist der gemeldete Äquivalentdurchmesser — daran ist zu sehen, ob
die eine Zahl die Pore trifft oder ob ihre Form sie unbrauchbar macht.

Die Lupe zeigt immer dieselben Ebenen wie das Hauptbild, mit derselben Deckkraft.
Sie kostet nichts an Dateigröße: geschnitten wird aus den ohnehin eingebetteten
Ebenenbildern, nicht aus einem eigenen PNG je Pore.

---

## 4. Prüfbilder mit bekannter Wahrheit — `stamp_pores.py`

Prägt echten Schliffbildern Poren mit exakt bekannter Lage, Fläche und
Äquivalentdurchmesser auf. Gezeichnet wird nur in die Probe, außerhalb des
Maßstab-Overlays und auf hellem Grund.

**Achtung:** Die Bilder unter `test\Pore_detection_test` werden *an Ort und
Stelle* ersetzt. Die Originale liegen unter `test\Safe_without_Pores`, die
Wahrheit landet in `data\runs\aufgepraegte_poren\wahrheit.json`.

Erst trocken laufen lassen:

```powershell
.venv\Scripts\python.exe scripts\stamp_pores.py --dry-run
```

Dann tatsächlich prägen:

```powershell
.venv\Scripts\python.exe scripts\stamp_pores.py --pores 25 --seed 7
```

| Option | Standard | Wirkung |
|---|---|---|
| `--pores N` | `18` | Poren je Bild |
| `--seed N` | `1` | Zufallssaat, damit der Satz reproduzierbar ist |
| `--input PFAD` | `test/Pore_detection_test` | Quellbilder |
| `--output PFAD` | `data/runs/aufgepraegte_poren` | Wahrheit, Masken, Übersicht |
| `--dry-run` | — | nichts schreiben, nur berichten |

Der übliche Ablauf: prägen → `run` darüber laufen lassen → Fund gegen
`wahrheit.json` halten.

---

## 5. Tests

Die schnellen Tests, wie bei jedem Commit:

```powershell
pytest
```

Die Benchmarks gegen den echten Bildsatz sind über den Marker `benchmark`
ausgenommen und werden gezielt gestartet:

```powershell
pytest -m benchmark
pytest tests\benchmark\test_scale_real.py -v
pytest tests\benchmark\test_specimen_real.py -v
```

Linting und Typprüfung:

```powershell
ruff check .
mypy
```

---

## 6. Was wo landet

| Pfad | Inhalt |
|---|---|
| `data\runs\` | alle Übersichtsblätter, JSON- und CSV-Ausgaben |
| `data\runs\gui\index.html` | die eigenständige Einzelbild-Ansicht |
| `data\runs\aufgepraegte_poren\` | Prüfbilder-Wahrheit, Masken, Übersicht |
| `data\cache\` | Zwischenergebnisse |
| `config\default.yaml` | Referenzkonfiguration; Profile unter `config\profiles\` enthalten nur die Abweichungen |

---

## Hinweise

* `python -m poredet ...` und das installierte Skript `poredet ...` sind
  dasselbe — beides landet in `src/poredet/cli/main.py`.
* Ohne erkannten Maßstab wird in Pixeln weitergemessen; die physikalischen
  Werte bleiben leer. Es wird nirgends ein Faktor unterstellt.
* Findet sich kein Einbettmittel, gilt das ganze Bild als Probe.
* Die FastAPI-Oberfläche (`api/`, `web/`) ist noch Skelett — bis dahin ist
  `gui_vorschau.py` der Weg zur Sichtprüfung.
