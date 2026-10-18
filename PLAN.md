# Pore Detection – Projektplan

> Status: Phase 0–4 fertig (Abgleich an echten Bildern ganz am Ende), Werkzeuge, Filter und Korrekturen (7.3) umgesetzt, Untergrundkorrektur (7.4) umgesetzt, Proben-KOS (7.5), Porenindex (7.6), Plots (7.7), Launcher-Reiter Evaluations (7.8), Ergebnis-Excel (7.9) umgesetzt · Stand: 2026-10-18 ·
> Schnittstellen-Version 0.14

---

## 1. Ziel

Ein Python-Programm, das Poren in Lichtmikroskop-Schliffbildern automatisch erkennt, segmentiert
und quantitativ auswertet (Anzahl, Größe, Form, Porosität) und die Ergebnisse als Tabellen
(JSON, Excel) und Bilder ausgibt. Gestartet wird per Kommandozeile oder GUI-Launcher, angesehen
im Viewer direkt nach dem Lauf.

Grundgedanke: **Ein neues Verfahren oder Auswertungsmodul = eine neue Datei + ein Eintrag in der
Config.** Alles andere (Laden, Kacheln, Parallelisierung, Fehler, Schreiben) erledigt `core`.

---

## 2. Rahmenbedingungen (festgelegt)

| Punkt | Entscheidung |
|-------|--------------|
| Umgebung | Bestehende `.venv` im Projektordner (Python 3.14) |
| Bilddaten | Lichtmikroskop-Schliffbilder, **gestitchte Panoramen** (groß), Farbe. Poren **dunkel auf hell**. Verschiedene Objektive/Vergrößerungen kommen vor |
| Dateiformate | TIFF (auch 16 Bit), PNG, JPG, BMP. Exotische Formate erst bei Bedarf |
| Eingabe | **Ein Lauf = eine ID.** Eingabe ist ein Ordner, dessen Name die ID ist. Bilder liegen direkt darin, nie in Unterordnern |
| Maßstab | Aus dem eingebrannten Maßstabsbalken (Verfahren `scale_bar`, portiert und beschleunigt aus Pore_Detection_V2) |
| Komplexität | So einfach wie möglich: flache Struktur, wenige Module |
| Plug & Play | Oberste Priorität. Jedes Verfahren ist austauschbar und **in sich abgeschlossen**: feste Schnittstelle pro Stufe, Auswahl per Name in der Config, neues Verfahren = neue Datei |
| Datenformat | An allen Schnittstellen **NumPy-Arrays** in OpenCV-Form – keine Torch-Tensoren |
| Arbeitsbild | Vorverarbeitung und Detektion arbeiten auf einem **Graustufenbild**; Farbe bleibt über das Original verfügbar |
| Abhängigkeiten | Alles über `requirements.txt`. PyTorch ist **optional** (eigener Index), nur für GPU-Wege |
| Geschwindigkeit | Höchste Priorität: alles parallelisieren, was sinnvoll geht |
| Hardware | Läuft **immer** auch auf der CPU. NVIDIA-GPU wird automatisch erkannt und genutzt. Testrechner: RTX 2060 (6 GB), Einsatz: deutlich stärkere NVIDIA-Karte → GPU-Nutzung passt sich **dynamisch** an |
| GPU-Verfahren | Ein Name in der Config, **CPU- und GPU-Weg in einer Datei**, Auswahl automatisch |
| Verarbeitung | Bild für Bild: jedes Bild läuft einzeln durch die gesamte Pipeline |
| Parallelität | Innerhalb eines Bildes über Kacheln (Tiling) für lokale Schritte |
| Einheiten | Gerechnet wird in **Pixeln**. Maßstab steht überall zur Verfügung; Parameter dürfen in px oder µm angegeben werden |
| Fehler | Schlägt ein Bild fehl → überspringen, melden, mit dem nächsten Bild weitermachen |
| Config | Einzige Quelle aller Einstellungen, wird als korrekt angenommen (keine Validierung) |
| Tests | Vorerst keine |
| ML | Vorerst **nicht** Teil des Plans. Kann später als weiterer Detektor nachgerüstet werden (gleiche Schnittstelle) |
| Sprache im Code | Englisch: Datei-/Ordnernamen, Bezeichner, Kommentare, Docstrings, Log- und Ausgabetexte |
| Pipeline-Stufen | `sample` → `preprocessing` → `detection` → `postprocessing` → `evaluation` → Schreiben |
| GUI | Ordner `gui/` mit **Launcher** (Pfade wählen, Lauf starten) und **Viewer** (nur direkt nach dem Lauf, nur lesend, schnell). Beide ändern die Config nie |

---

## 3. Installation

```powershell
# 1. Alles Nötige
.venv\Scripts\python -m pip install -r requirements.txt

# 2. Optional: PyTorch für die GPU-Wege der Verfahren
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Ohne PyTorch läuft das Programm vollständig auf der CPU (alle Verfahren nutzen dann ihren CPU-Weg).

`requirements.txt` (geplant):

```
numpy
opencv-python
pandas
xlsxwriter
pyyaml
tqdm
PySide6
```

---

## 4. Projektstruktur

```
Pore_Detection/
├── .venv/
├── PLAN.md
├── README.md
├── requirements.txt
├── config.yaml              # globale Config: ALLE Einstellungen
├── main.py                  # CLI: python main.py (alle Einstellungen aus config.yaml)
├── gui_main.py              # Start der GUI (Launcher, von dort aus Viewer)
│
├── pore_detection/          # Kern – kennt die GUI nicht
│   ├── __init__.py          # öffentliche API: run(), load_config(), list_methods()
│   │
│   ├── core/                # Infrastruktur ("Fließband"), ändert sich selten
│   │   ├── __init__.py
│   │   ├── types.py         # Datentypen (siehe docs/interfaces.md)
│   │   ├── config.py        # liest config.yaml aus dem Projektordner → unveränderliches Config-Objekt
│   │   ├── step.py          # Basisklassen der Stufen
│   │   ├── registry.py      # Registrierung + automatisches Finden der Verfahren
│   │   ├── device.py        # GPU-Erkennung, freier Grafikspeicher, to_gpu()/to_numpy()
│   │   ├── imaging.py       # Hilfsfunktionen: to_gray(), downscale()
│   │   ├── tiling.py        # Kacheln zerlegen / zusammensetzen / überspringen
│   │   ├── io.py            # Bilder finden + laden
│   │   ├── writer.py        # Schreibmodul (fest, kein Plug-in)
│   │   ├── pipeline.py      # Ablauf pro Bild, Zuschneiden auf ROI, Fehlerbehandlung
│   │   ├── geometry.py      # Porengeometrie für Porenschritte (Rahmen, Formwerte als Entscheidungshilfe)
│   │   └── corrections.py   # manuelle Korrekturen aus dem Viewer (fest, kein Plug-in, 7.3)
│   │
│   ├── sample/              # 1. Probenanalyse: je ein Verfahren für ROI, Maßstab und Proben-KOS
│   │   ├── __init__.py
│   │   ├── embedding_medium.py   # Platzhalter: Einbettmittel erkennen → roi_mask
│   │   ├── fixed_scale.py        # fester Maßstab aus der Config
│   │   ├── scale_bar/            # Maßstabsbalken lesen (overlay.py, reading.py, units.py) → µm/px + Maske
│   │   ├── edge_lines.py         # Proben-KOS aus Geraden an den Probenkanten (7.5)
│   │   └── no_frame.py           # `none`: kein Proben-KOS
│   │
│   ├── preprocessing/       # 2. Vorfilter auf Graustufen (optionale Kette)
│   │   ├── gauss.py
│   │   └── background_correction.py   # Untergrund herausrechnen (7.4)
│   │
│   ├── detection/           # 3. Porendetektion (genau EIN Verfahren aktiv)
│   │   ├── threshold.py     # fester Grauwert
│   │   └── hysteresis.py    # zweistufige Schwelle (7.4)
│   │
│   ├── postprocessing/      # 4. Maske → nummerierte Poren (Maskenschritte, Labeling, Porenschritte)
│   │   ├── opening.py, closing.py, fill_holes.py          # Maskenschritte
│   │   ├── connected_components.py                        # Labeling
│   │   ├── min_size.py, merge_close.py, split_touching.py # Porenschritte, die Poren verändern
│   │   ├── min_diameter.py, scratch.py, compactness.py, roundness.py   # Filter (verwerfen mit Grund)
│   │   └── pore_properties.py                             # Werte pro Pore
│   │
│   ├── evaluation/          # 5. Auswertungsmodule (beliebig viele aktiv)
│   │   ├── porosity.py      # Porenanzahl, Porosität
│   │   ├── pore_index.py    # spannungsgewichteter Porenindex (Porenkonzept, 7.6)
│   │   ├── comparison.py    # Vergleichsbild: Original | Poren rot (comparison.jpg)
│   │   └── frame_picture.py # Probe mit KOS, Rahmen, Länge und Dicke bemaßt (frame_picture.jpg)
│   │
│   └── plots/               # 6. Plots als Bilder (beliebig viele aktiv, 7.7)
│       └── phi_over_q.py, pores_over_height.py, pore_sizes.py
│
├── gui/                     # PySide6
│   ├── __init__.py
│   ├── launcher/
│   │   └── __init__.py
│   └── viewer/
│       └── __init__.py
│
├── docs/                    # verbindliche Spezifikationen
│   ├── interfaces.md        # Schnittstellen: Datentypen, Formen, Einheiten, Methoden
│   ├── output_format.md     # Ausgabestruktur (Ordner + Formate festgelegt, Inhalte offen)
│   └── good_to_know.md      # bekannte Eigenheiten, die vorerst so bleiben
│
├── data/                    # Eingabebilder (nicht im Git)
└── output/                  # Ergebnisse (nicht im Git)
```

Regel für Abhängigkeiten (nur in Pfeilrichtung importieren):

```
main.py ──────┐
              ├─► pore_detection (API) ─► core ◄─ sample / preprocessing / detection / postprocessing / evaluation
gui/launcher ─┘          │
     │                   ▼ schreibt <ID>_results/  (Format: docs/output_format.md)
     └─ öffnet ─► gui/viewer ── liest <ID>_results/ + Originale im Eingabeordner (nur lesend)
```

Verfahren importieren nur `core`, nie einander. Der Launcher nutzt nur die öffentliche API.
Der Viewer importiert nichts aus `pore_detection`; er bekommt vom Launcher den Ergebnisordner und
die Darstellungseinstellungen (`gui`) und liest nur die Ausgabedateien.

---

## 5. Schnittstellen (Plug & Play)

Verbindlich festgelegt in **`docs/interfaces.md`** (umgesetzt in `core/types.py` und `core/step.py`).

### 5.1 Grundregeln

- **Jedes Verfahren ist in sich abgeschlossen.** Es bekommt immer dasselbe Format, egal welche
  Schritte davor liefen. Beispiel: Ein Detektor funktioniert mit und ohne Vorfilter.
- **NumPy überall**, OpenCV-Form: Graustufen-Arbeitsbild `(H, W)` `float32` 0..1, Original
  `(H, W, 3)` RGB, Masken `(H, W)` `bool`, Labels `(H, W)` `int32`. Ursprung oben links.
- **Bild-Kontext:** Jeder Schritt ab `preprocessing` bekommt einen nur lesbaren Kontext `ctx`
  (Name, Originalbild in Farbe, Probenmaske, Maßstab) plus genau seine eigenen Daten. Beispiel in
  einer Auswertung: `ctx.roi_mask`, `ctx.pixel_size_um`, `ctx.original`.
- Ein Schritt gibt neue Daten zurück und verändert nie seine Eingabe. `ctx` ist technisch
  schreibgeschützt.
- Einheit steht immer im Namen (`height_px`, `area_um2`, `pixel_size_um`).
- **Keine freien Zusatzfelder.** Braucht ein Verfahren neue Daten, wird zuerst
  `docs/interfaces.md` erweitert (mit Versionsnummer).
- Auswertungsmodule schreiben **keine** Dateien – sie liefern Daten, das Schreibmodul schreibt.
- Hinweis für Modul-Autoren: Kacheln laufen parallel in Threads. Das ist schnell mit OpenCV-/
  NumPy-Funktionen; eigene Python-Schleifen laufen trotzdem nur auf einem Kern.

### 5.2 Stufen

```
ID-Ordner ─► laden ─► sample (ROI + Maßstab) ─► ImageContext + Graustufenbild
          ─► [Vorfilter]* ─► Detektor ─► Maske
          ─► Postprocessing: [Maskenschritte]* ─► Labeling ─► [Porenschritte]* ─► LabeledPores
          ─► [Auswertung]* (parallel) ─► Schreibmodul ─► <ID>_results/
```

| Stufe | Verfahren | Eingabe | Ausgabe |
|-------|-----------|---------|---------|
| `sample.roi` | genau eins | Originalbild | `roi_mask` (Probe = True) |
| `sample.scale` | genau eins | Originalbild | `pixel_size_um` oder `None` |
| `sample.frame` | genau eins | Probenmaske | Proben-KOS (`SampleFrame`) oder `None` |
| `preprocessing` | Kette, optional | `ctx`, Graustufenbild | Graustufenbild |
| `detection` | genau eins | `ctx`, Graustufenbild | Maske: alles, was Pore sein könnte (primitiv, ungefiltert) |
| `postprocessing.mask_steps` | Kette, optional | `ctx`, Maske | Maske |
| `postprocessing.labeling` | genau eins | `ctx`, Maske | `LabeledPores` (Pixel → Poren-Nummern) |
| `postprocessing.label_steps` | Kette, optional | `ctx`, `LabeledPores` | `LabeledPores` (Merge, Filter, Kennwerte …) |
| `evaluation` | beliebig viele | `ctx`, `LabeledPores` | `EvaluationResult` |
| `plots` | beliebig viele | Zeile aus `images.json`, Tabelle `pores.json` | Bild (PNG, schreibt `core`) |

- Die Detektion liefert **alles**, was sie als Pore erkennt. Aussortiert (Rand, Größe, Artefakte)
  und zusammengelegt wird mit Filtern im Postprocessing.
- Die Zuordnung Pixel → Pore (Labeling) ist Teil des Postprocessing und selbst austauschbar.
- `sample`-Verfahren arbeiten immer auf dem **Original**, nicht auf vorgefilterten Bildern.
- Die Umrechnung Farbe → Graustufen macht `core` einmal pro Bild, Gewichte in der Config
  (`input.gray_weights`). Wer Farbe braucht (Detektor, Auswertung), nimmt `ctx.original`.

### 5.3 Kennwerte pro Pore

- Grundwerte pro Pore (Fläche, Bounding Box, Schwerpunkt …) berechnet ein **Postprocessing-Schritt**
  (z. B. `pore_properties`), nicht `core`.
- Er legt sie in `LabeledPores.properties` ab (eine Zeile pro Pore). Weitere Porenschritte dürfen
  Spalten hinzufügen.
- Auswertungsmodule holen sie mit `pores.get("area_px2")`. Fehlt die Spalte → klare Fehlermeldung.
- **Regel (keine automatische Prüfung):** Schritte, die Poren verändern (Merge, Filter), stehen
  in der Config **vor** den Schritten, die Kennwerte berechnen.

### 5.4 Auswertungsmodule

- Liefern pro Bild: Werte für das ganze Bild, optional eine Tabelle pro Pore, optional Bilder.
- **Spaltennamen legt jedes Modul selbst fest** (kein automatisches Präfix). Liefern zwei Module
  denselben Namen, bricht der Lauf beim ersten erfolgreichen Bild mit klarer Meldung ab.
- Optional `finalize()`: wird am Ende des Laufs einmal aufgerufen, bekommt die Ergebnisse aller
  Bilder der ID und liefert Kennwerte über die ganze ID (z. B. Porosität).
- Ziel: neues Modul = eine Datei, seine Werte erscheinen automatisch in JSON/Excel.

### 5.5 Einheiten und Maßstab

- Alle Verfahren rechnen in Pixeln. `ctx.pixel_size_um` steht in jeder Stufe zur Verfügung.
- Größenparameter tragen die Einheit im Namen (`min_area_px2` **oder** `min_area_um2`). Ein Filter
  darf beides anbieten, in der Config steht genau einer.
- µm-Parameter ohne Maßstab → Bild schlägt fehl („scale required but not found“).
- Auswertung: `_px`-Werte immer, `_um`-Werte nur mit Maßstab, sonst leer (NaN).
- Verfahren `scale_bar`: liest den eingebrannten Balken (Länge in px + OCR der Beschriftung) und
  liefert zusätzlich eine **Maske des Maßstabs** (`ScaleResult.scale_mask`); `core` nimmt sie aus
  der `roi_mask` heraus (der schwarze Balken ist nie eine Pore). Im Zweifel kein Maßstab.
  Alternativ `fixed_scale` mit Wert aus der Config.

### 5.6 Registrierung und GPU-Wege (`core/registry.py`, `core/step.py`)

```python
@register("gauss")
class GaussFilter(Filter):
    params = {"sigma_px": "standard deviation of the gaussian kernel"}
    tileable = True
    tile_overlap_px = ...

    def process(self, ctx, image):        # CPU-Weg – Pflicht
        ...
    def process_gpu(self, ctxs, images):  # GPU-Weg – optional, bekommt Stapel von Kacheln
        ...
```

- Beim Start werden alle Dateien in den Stufen-Ordnern automatisch importiert.
- Jedes Verfahren beschreibt seine Parameter in `params` (Name → Beschreibung) – daraus erzeugt
  `list_methods()` die Übersicht.
- **CPU-Weg ist Pflicht, GPU-Weg optional.** `core` nimmt den GPU-Weg, wenn er existiert und eine
  GPU verfügbar ist, sonst den CPU-Weg. Die Config ist auf jedem Rechner gleich.

### 5.7 Globale Konfiguration (`config.yaml`)

- **Eine** globale Config enthält **alle** Einstellungen.
- **Wird als korrekt angenommen** – keine Validierung, keine Standardwerte im Code.
- **Einzige Quelle für Werte:** keine Standardwerte im Code, keine Kommandozeilen-Parameter,
  keine Umgebungsvariablen, die GUI ändert nichts.
- **Ausnahme Eingabe-/Ausgabepfad:** Der Launcher übergibt die gewählten Pfade direkt an `run()`.
  `input.path` / `output.dir` gelten für `main.py` und als Vorbelegung im Launcher.
- **Ausnahme Auswertungseingaben (7.8):** Im Reiter „Evaluations“ geänderte Parameter der
  Auswertungsmodule gelten nur für den nächsten Lauf (Config-Objekt an `run()`), festgehalten in
  `config_used.yaml`. `config.yaml` liefert die Standardwerte und wird nie geschrieben.
- Wird einmal beim Start eingelesen (CLI: Programmstart, Launcher: jeder Klick auf „Start“) und
  danach nur noch als unveränderliches Objekt weitergereicht. Im Launcher gilt eine Änderung an
  `config.yaml` also ab dem nächsten Lauf, ohne Neustart.
- Kopie wird als `config_used.yaml` mit den Ergebnissen gespeichert.

```yaml
runtime:        # device, gpu_memory_fraction, tile_size, tile_batch_size, threads, prefetch
input:          # path (ID-Ordner), extensions, gray_weights
sample:         # roi · scale · frame: je genau ein Verfahren
preprocessing:  # Kette von Vorfiltern (Graustufen)
detection:      # genau ein Verfahren
postprocessing: # mask_steps (Kette) · labeling (genau eins) · label_steps (Kette)
evaluation:     # Liste von Auswertungsmodulen
output:         # dir, save_full_overlay
logging:        # level, file
gui:            # Darstellung
```

### 5.8 Öffentliche API (`pore_detection/__init__.py`)

```python
load_config() -> Config     # liest immer config.yaml im Projektordner, Zugriff per Attribut
list_methods() -> dict[stage, list[MethodInfo]]
run(config, input_path=None, output_dir=None, progress=None, cancel=None) -> RunResult
```

- `input_path` / `output_dir`: `None` = Wert aus der Config (CLI), sonst vom Launcher gewählt
- `progress`: Callback (Bild x von n, aktuelle Stufe) – CLI zeigt `tqdm`, Launcher einen Balken
- `cancel`: Event zum Abbrechen; wirkt **zwischen** Bildern
- `RunResult`: Anzahl erfolgreich / fehlgeschlagen, Liste der Fehler, Ausgabeordner, Laufzeit

---

## 6. Ablauf & Parallelisierung

### 6.1 Was `core` macht

`core` ist **kein Schritt**, sondern das Fließband, das die Verfahren aufruft:

| Aufgabe | Was das heißt |
|---|---|
| Config | einlesen, jedem Verfahren seine Parameter geben |
| Verfahren finden | alle Dateien in den Stufen-Ordnern einsammeln (`@register`) |
| Laden | Bilder im ID-Ordner finden, öffnen, Namen `ID_Bildname` bilden, einmal in Graustufen umrechnen |
| Reihenfolge | Verfahren in der Reihenfolge der Config aufrufen |
| ROI | nach `sample` auf die Bounding Box der Probe zuschneiden; vor dem Labeling alles außerhalb der Probe aus der Maske entfernen |
| Kacheln | lokale Schritte gekachelt und parallel ausführen, Kacheln ohne Probe überspringen |
| GPU | erkennen, freien Speicher messen, Stapelgröße bestimmen, GPU- oder CPU-Weg wählen |
| Hilfsfunktionen | `to_gray()`, `downscale()` (z. B. Einbettmittel-Erkennung auf verkleinerter Übersicht), `to_gpu()`, `to_numpy()` |
| Fehler | Bild überspringen, melden, weitermachen |
| Schreiben | Schreibmodul (JSON, Excel, Bilder) |
| Fortschritt | Konsole bzw. Launcher |

### 6.2 Bild für Bild

```
für jedes Bild im ID-Ordner:
    laden ─► sample ─► zuschneiden ─► preprocessing ─► detection ─► postprocessing
          ─► evaluation ─► Ergebnis schreiben ─► Speicher freigeben
am Ende: finalize() aller Auswertungsmodule ─► ID-Kennwerte schreiben
```

- Immer nur ein Bild in Bearbeitung.
- Ergebnisse eines Bildes werden erst geschrieben, wenn das Bild **komplett** fertig ist.
- Die Werte pro Bild (eine Zeile je Bild) behält `core` für `finalize()` im Speicher (klein).
  Tabellen pro Pore liest `finalize()` aus den geschriebenen **JSON**-Dateien zurück – nie aus Excel.
- **Vorladen** (`runtime.prefetch`, abschaltbar): Das nächste Bild wird im Hintergrund geladen,
  während das aktuelle rechnet. Kostet kurzzeitig Speicher für ein zweites Bild.
- Schreiben läuft im Hintergrund, während das nächste Bild rechnet.
- **Laden:** `cv2.IMREAD_UNCHANGED`, BGR → RGB. Graustufen-Quellbilder werden auf 3 Kanäle kopiert,
  Alpha-Kanal wird verworfen. `ctx.original` bleibt im Quellformat (8/16 Bit, spart Speicher), das
  Arbeitsbild ist Graustufen `float32` 0..1 (`input.gray_weights`).

### 6.3 Zuschneiden auf die Probe

- Nach `sample` schneidet `core` alle Daten auf das umschließende Rechteck der Probe zu.
- Alle folgenden Stufen arbeiten im zugeschnittenen Bereich; `ctx.info.offset_px` kennt den
  Versatz zum Original.
- Kacheln, die komplett außerhalb der Probe liegen, werden nicht gerechnet.
- Einbettmittel innerhalb von Randkacheln bleibt **unverändert** im Bild; die Verfahren können
  `ctx.roi_mask` nutzen.
- Vor dem Labeling setzt `core` alle Maskenpixel außerhalb der Probe auf False – nie wird eine
  Pore im Einbettmittel nummeriert.

### 6.4 Kacheln (Tiling)

- Nur für **lokale** Schritte (Pixel hängt nur von seiner Umgebung ab): Vorfilter, lokale
  Detektoren, Maskenschritte. Jedes Verfahren gibt `tileable` und `tile_overlap_px` an.
- Nicht kachelbar: `sample`, globale Detektoren (z. B. Otsu), Labeling, Porenschritte, Auswertung.
- Ein Verfahren mit `tileable=True` bekommt **eine** Kachel (inkl. passendem Kachel-`ctx` mit
  eigenem `offset_px`) und schreibt seinen Code so, als wäre sie das ganze Bild. `core` ruft es für
  alle Kacheln parallel auf (Threads, alle CPU-Kerne).
- Jede Kachel ist eine **Kopie**, `ctx` ist schreibgeschützt → parallele Kacheln können sich nicht
  gegenseitig verändern.
- Aufeinanderfolgende kachelbare Schritte bleiben gekachelt; die Überlappungen werden
  **aufsummiert**. Am Bildrand wird gespiegelt aufgefüllt (`reflect`).
- Ein Modul funktioniert immer auch mit `tileable=False` – nur langsamer.

### 6.5 GPU (dynamisch)

- `runtime.device: auto` – GPU wird genutzt, wenn PyTorch + CUDA vorhanden, sonst CPU.
- Jedes Verfahren hat einen CPU-Weg; den GPU-Weg (`process_gpu`, Stapel von Kacheln) nutzt `core`
  automatisch, wenn vorhanden und eine GPU verfügbar ist.
- `tile_batch_size: auto`: Vor jedem Bild misst `core` den freien Grafikspeicher
  (`torch.cuda.mem_get_info`, < 1 ms) und bestimmt daraus die Stapelgröße. Passt das ganze Bild,
  wird es am Stück gerechnet.
- Obergrenze `gpu_memory_fraction` (z. B. 0.8) in der Config.
- Speicherfehler (OOM): Stapel halbieren und erneut versuchen.
- Ergebnis: dieselbe Config läuft auf der RTX 2060, auf der großen Karte und ohne GPU.

### 6.6 Fehlerbehandlung

- Jedes Bild läuft in einem eigenen `try`-Block. Schlägt ein Schritt fehl:
  - Bild wird **übersprungen**, nichts von diesem Bild wird geschrieben
  - Fehler wird **gemeldet**: Konsole/Launcher-Log + Fehlerliste in `<ID>_results/`
    (Bild, Stufe, Verfahren, Fehlermeldung)
  - Lauf geht mit dem **nächsten Bild** weiter
- Am Ende: „x von n Bildern erfolgreich, y fehlgeschlagen“.
- Nur Fehler, die alle Bilder betreffen, brechen den Lauf ab (Ausgabeordner nicht beschreibbar,
  doppelte Spaltennamen von Auswertungsmodulen – geprüft beim ersten erfolgreichen Bild).
- Abbrechen (`cancel`) wirkt zwischen Bildern; fertige Bilder bleiben erhalten, die ID-Kennwerte
  werden bei Abbruch nicht geschrieben.

---

## 7. GUI (`gui/`)

Framework: **PySide6**. Zweck: Lauf starten und Ergebnisse **schnell prüfen, um Einstellungen
anzupassen**.

### 7.1 Launcher (`gui/launcher/`)

- ID-Ordner (Eingabe) und Ausgabeordner wählen (vorbelegt aus der Config)
- Aktive Config nur anzeigen – **keine** Bearbeitung. `config.yaml` wird bei jedem Start eines
  Laufs neu gelesen
- Reiter **Evaluations**: Eingaben der Auswertungsmodule für den nächsten Lauf (7.8)
- Lauf starten / abbrechen, Fortschrittsbalken, Log inkl. fehlgeschlagener Bilder
- Pipeline läuft in einem Hintergrund-Thread
- Nach Abschluss: Viewer mit `<ID>_results/` öffnen (gibt Ergebnisordner + `gui`-Einstellungen mit).
  Checkbox „Open viewer after the run“ (an = öffnet automatisch), zusätzlich Knopf „Open viewer“
- Solange ein Viewer offen ist, ist „Start“ gesperrt (ein neuer Lauf überschreibt `<ID>_results/`)
- Ruft nur `run()` der öffentlichen API auf

### 7.2 Viewer (`gui/viewer/`)

- **Nur direkt nach einem Lauf**, geöffnet vom Launcher. Kein eigenständiger Start, kein Öffnen
  alter Ergebnisordner. Viewer zu = Lauf abgeschlossen (Dateien bleiben auf der Platte).
- **Korrekturen** mit den Werkzeugen (siehe 7.3) gelten bis zum nächsten Lauf; sonst nur lesend
- Zeigt immer **ein Bild**, durchklicken zum nächsten/vorigen.
- **Schnell:**
  - **Keine gespeicherten Vorschaubilder.** Pro Bild werden Original (aus dem Eingabeordner),
    `labels.tif` und `roi_mask.tif` in voller Auflösung geladen (parallel, ~1,4 s bei 99 MP);
    das Übersichtsbild für die Gesamtansicht rechnet der Viewer selbst (nie gespeichert)
  - beim Zoomen wird das Overlay nur für den sichtbaren Ausschnitt live gezeichnet
  - nächstes Bild wird im Hintergrund vorgeladen → Blättern ohne Wartezeit
  - Begrenzungsrahmen einer Pore erst beim Anklicken (alle auf einmal kosten ~0,5 s)
  - Tabellen kommen aus JSON, nie aus Excel
- Zeigt: Original / Overlay / Maske (umschaltbar, Zoom) · Werte pro Pore und pro Bild ·
  ID-Kennwerte · verwendete Einstellungen · Fehlerliste
- Gekoppelt an die Pipeline nur über `docs/output_format.md`

**Entscheidungen zum Aussehen** (Vorbild: Viewer aus Pore_Detection_V2, einzeln besprochen):

1. Kopfzeile: `‹ 2 / 5 ›` + Bildname wie in der Ausgabe (`<ID>_<Bild>`); blättern mit Pfeilen
   oder Pfeiltasten. Kein Auswertungszeitpunkt (Laufdaten noch offen).
2. Verfahrenskette als Chips: Maßstab → Probe → Detektor (z. B. `fixed_scale → imagej →
   threshold`), Parameter als Tooltip. Quelle: `config_used.yaml`.
3. Knopf hell/dunkel: wirkt nur, solange der Viewer offen ist; beim Öffnen gilt `gui.theme`.
4. Layout (Skizze `temporäre Sachen/Layout_grob.png`): oben links Block aus Werkzeugspalte |
   Bild | optische Kontrolle (Lupe der ausgewählten Pore mit Zoom); oben rechts Kennzahlen,
   darunter Masken/Ebenen über dem Bild; unten zwei Plätze für Graphen + Porenliste.
5. Werkzeugspalte: Zoom +/−/Einpassen; Korrekturwerkzeuge nach 7.3.
6. Optische Kontrolle (Lupe): Klick auf eine Pore im Bild oder in der Liste wählt sie aus; Lupe
   zeigt das Original in voller Auflösung mit Umgebung (Regler „Umfeld“), die Pore nur als
   **feine Umrisslinie** (keine Füllung); darunter die Werte der Pore.
7. Kennzahlen: feste Auswahl, **fest im Code** des Viewers (nicht automatisch, nicht in der
   Config). Vorerst `porosity_percent`, `pore_count`, `pore_area_px2`, `sample_area_px2`;
   andere Werte werden später im Code eingetragen.
8. Die ganze Seite ist **vertikal scrollbar** (wie V2): oben Bild/Lupe/Kennzahlen/Masken, darunter
   die Plots und die Porenliste (rechts). Der Bereich unter dem Bild ist ein **Raster mit zwei
   Spalten nur für Plots**, das je nach Anzahl der Plots nach unten wächst.
9. ID-Kennwerte, Fehlerliste und verwendete Einstellungen: je ein Knopf in der Kopfzeile, öffnet
   ein eigenes Fenster; der Fehlerknopf zeigt die Anzahl („Errors (2)“).
10. Masken/Ebenen: vorerst zwei Overlays mit Checkbox – **Poren** und **Einbettmittel**
    (Probenmaske; dafür schreibt die Pipeline die ROI-Maske zusätzlich in die Ausgabe). Später
    weitere, z. B. verworfene Poren, Maßstab. Keine reine Masken-Ansicht mehr.
    Dazu (7.5) die Ebene **Sample frame** (Proben-KOS als Vektorgrafik).
    Direkt über den Checkboxen: Knöpfe für das **Grundbild** – Original / Untergrund.
    - Untergrund = geschätzte Hintergrundhelligkeit eines künftigen Vorfilters
      (Hintergrundkorrektur); bis dahin ist der Knopf ausgegraut.
    - Kontrast (Graustufenbild nach allen Vorfiltern) vorerst **entfernt**: gab es nur als
      Vorschaubild; ggf. später als `contrast.tif` in voller Auflösung.
11. Plots: Raster mit zwei Spalten, ein Platz pro Plot des Laufs (Reihenfolge der Config). Die
    Plots sind **Bilder aus der Ausgabe** (Stufe `plots`, 7.7); der Viewer zeigt sie nur an.
12. Porenliste wie in V2: Zeilen „Pore N“ + Fläche, **nach Fläche sortiert** (größte oben).
    Klick auf eine Zeile wählt die Pore aus (Markierung im Bild + Lupe), Klick auf eine Pore im
    Bild markiert ihre Zeile.
13. Dafür schreibt die Pipeline zusätzlich ein **Nummernbild** `labels.tif` (Porennummer je Pixel,
    gleiche Nummern wie `pores.json`, komprimiert).
14. **Optische Kontrolle des Maßstabs** oben in der Spalte der optischen Kontrolle (über der Poren-Lupe): Ausschnitt des Maßstab-Kastens aus
    dem Original, darüber die **gemessene Balkenlänge** als Markierung (Linie mit Endstrichen am
    ersten/letzten Balkenpixel); darunter der gelesene Text, die Balkenlänge in px und das
    Ergebnis in µm/px (keine Grenzen für den Wert: was gelesen wird, gilt). Daten aus
    `images.json` (`scale_label`, `scale_bar_px`, `scale_box_px`, `pixel_size_um`).
    Später: Maßstab als Ebene im Bild (Rechteck aus `scale_box_px`).

### 7.3 Werkzeuge, Filter und Korrekturen (beschlossen und umgesetzt 2026-10-16)

Vorbild: Pore_Detection_V2 (`filter.py`, `detektion/postprocess.py`, `detektion/merge.py`,
`korrekturen.py`, `viewer/sitzung.py`). Übernommen wird Verhalten, nicht Code; eingebaut wird
strikt über die vorhandenen Stufen und Schnittstellen. Kennwerte und Diagramme sind **nicht**
Teil davon (werden noch geändert).

**1. Schnittstelle 0.4 → 0.5**
- `LabeledPores` führt zusätzlich die **verworfenen** Poren mit: Nummer, Filter, Grund.
- Hilfsfunktion `pores.reject(ids, filter, reasons)` in `core`.
- **Ein Nummernraum** für gezählte und verworfene Poren; Nummern sind **stabil**: eine verworfene
  behält ihre Nummer (Lücke bei den gezählten), eine zurückgeholte bekommt sie wieder, eine neue
  die nächste freie (max + 1). Regel „lückenlos 1 … count“ wird zu „eindeutig, aufsteigend“.
- Auswertungsmodule sehen nur die **gezählten** Poren.

**2. Postprocessing-Verfahren** (je Datei eins, alle in `config.yaml` aktiv mit den V2-Werten)

| Stufe | Verfahren | Wirkung | V2-Werte |
|---|---|---|---|
| `mask_steps` | `open` | Morphologisches Öffnen (kachelbar) | Radius 1 px |
| `mask_steps` | `close` | Morphologisches Schließen (kachelbar) | in V2 aus → auskommentiert |
| `mask_steps` | `fill_holes` | nur Löcher bis `max_ratio` × Porenfläche füllen (ganzes Bild) | ratio 3 |
| `label_steps` | `min_size` | Rauschgrenze: **löscht** Objekte (technisch kein Objekt, nicht „verworfen“) | 4 px |
| `label_steps` | `merge_close` | nahe Teile vereinen, Spalt nur innerhalb der Probe, Spalt überbrücken | 1 px |
| `label_steps` | `split_touching` | berührende Poren trennen: Distanztransformation + Watershed, **nur OpenCV** | 3 px, ratio 0.15 |
| `label_steps` | `min_diameter` | Filter: **verwirft** mit Grund | 4 px |
| `label_steps` | `scratch` | Filter: Schleifriefen (lang, dünn, gerade) | wie V2 |
| `label_steps` | `compactness` | Filter: Solidität | 0.40 |
| `label_steps` | `roundness` | Filter: Rundheit über Fläche/Feret | 0.20 |

- Reihenfolge: Maskenschritte → Labeling → `min_size` → `merge_close` → `split_touching` → Filter
  → `pore_properties`.
- Formwerte für die Filter (Achsen, Solidität, Feret …) aus `core/geometry.py`, **nur intern**
  als Entscheidungsgrundlage, nicht in der Ausgabe.
- Weitere V2-Filter (max_area, extent, aspect_ratio, circularity, solidity, edge, edge_distance,
  contrast, intensity) nach Bedarf später als neue Datei.

**3. Ausgabe** (`docs/output_format.md`)
- `labels.tif` gezählte Poren; **`rejected_labels.tif`** verworfene Poren (zweite Datei),
  **`rejected.json`** mit Nummer, Filter, Grund.
- **`corrections.json`** pro Bild: welche Korrektur, wo, welche Poren betroffen.
- `images.json` / Excel: Spalte **`manual_corrections`** (Anzahl).

**4. Korrekturen** – gelten nur für den **aktuellen Ergebnisstand**: sie werden direkt in die
Ergebnisdateien geschrieben; ein neuer Lauf setzt alles auf den automatischen Stand zurück
(keine Wiederanwendung, keine Bindung an den Bildinhalt).
- Werkzeuge im Viewer:
  - **+** Klick auf verworfene Pore: zählen
  - **−** Klick auf gezählte Pore: verwerfen (Grund „manual“); eine eingezeichnete wird gelöscht
  - **Stift**: Umriss ziehen → neue Pore; **ersetzt** überdeckte Poren (gezählt oder verworfen),
    auf die Probe beschnitten, läuft durch keinen Filter
  - **Bereich**: umfahren → gezählte Poren, die **ganz** darin liegen, werden verworfen
  - **Rückgängig** (auch Strg+Z) und **Alle Korrekturen verwerfen** (Verlauf im Speicher, solange
    der Viewer offen ist)
- Nach **jeder** Korrektur wird **alles sofort** neu geschrieben: Bildordner (labels,
  rejected, pores.json, corrections.json), `images.json`, Excel, `id_values.json`
  (Auswertungsmodule + `finalize` neu; im Hintergrund).
- Technik: eine **Korrektursitzung** in `core` (über die öffentliche API), die der Launcher für das
  angezeigte Bild öffnet und dem Viewer durchreicht. Der Viewer importiert weiterhin nichts aus
  `pore_detection`.

**5. Viewer**
- Werkzeugleiste: +, −, Stift, Bereich (aktiver Modus farbig, Fadenkreuz beim Zeichnen),
  Rückgängig, Alles verwerfen.
- Neue Ebenen mit Anzahl: **Verworfen**, **Von Hand entfernt**, **Von Hand aufgenommen**
  (inkl. eingezeichnete).
- Werte-Leiste zeigt bei Klick den Status, bei verworfenen Poren Filter und Grund.
- Porenliste: von Hand aufgenommene / eingezeichnete Poren farbig markiert.
- Kein Hover-Tooltip. Stattdessen (2026-10-18) **Hover-Vorschau**: die Pore, die ein Klick treffen
  würde, bekommt einen breiten, halbtransparenten Umriss (Auswahl: schmale Linie) (bei +/− nur verworfene/gezählte, in der Werkzeugfarbe);
  Fangradius 12 px auf dem Bildschirm (unabhängig vom Zoom), Zeigehand über einer treffbaren Pore,
  sonst Pfeil. Der Klick trifft genau die umrissene Pore (auch bei den Korrekturen).

**6. Bewusst nicht:** Einstellungen im Viewer ändern / Bild neu rechnen (wie V2) – `config.yaml`
bleibt die einzige Stelle für Einstellungen, die GUI ändert sie nie.

**Umsetzungsreihenfolge:** (1) Schnittstelle + Ausgabe → (2) Postprocessing-Verfahren →
(3) Korrektursitzung in `core` (ohne GUI testbar) → (4) Viewer.

### 7.4 Untergrundkorrektur + zweistufige Schwelle (beschlossen 2026-10-18)

Anlass: synthetische Testbilder (`syntetic_test/`). Poren (Kern 15–55) und Gefüge (Netzwerk
80–140, Körner 130–180) liegen weit auseinander, aber Ausleuchtung, Stitching-Nähte und
Belichtung verschieben alles je Bild und Bildbereich → ein fester Grauwert (`threshold`) passt nie
überall; zudem fehlt der weiche Porenrand (96 % der übersehenen Porenpixel, Fläche −16 %).
Prototyp: Porositätsfehler 0,19 → 0,075 Prozentpunkte, Korrelation GT/Detektion 0,59 → 0,83.

**1. Vorfilter `background_correction`** (`preprocessing/background_correction.py`, `Filter`)
- Schätzt die Helligkeit des Gefüges (Untergrund) und teilt das Bild dadurch: Gefüge ≈ 1,
  Poren ≈ 0,2, unabhängig von Ausleuchtung, Nähten, Belichtung. Ergebnis auf 0..1 begrenzt.
- Untergrund: auf einer um `work_scale` verkleinerten Kopie; Einbettmittel (außerhalb
  `ctx.roi_mask`) auf den typischen Gefügewert gesetzt; Graustufen-Closing mit Fenster
  `window_px` (größer als die größte Pore) füllt Poren und Netzwerk; glätten; zurück auf volle Größe.
- Parameter: `window_px` (nur px), `work_scale`.
- Nicht kachelbar (großes Fenster, verkleinerte Kopie ist billiger), kein GPU-Weg.
- Der Untergrund wird **vorerst nicht** ausgegeben (Viewer-Knopf „Untergrund“ bleibt aus);
  dafür später Schnittstelle 0.6.

**2. Detektor `hysteresis`** (`detection/hysteresis.py`, `Detector`)
- Pixel unter `seed_threshold` = sichere Porenkerne; jeder zusammenhängende Bereich unter
  `grow_threshold`, der einen Kern enthält, ist Pore (holt den weichen Rand mit).
- In sich abgeschlossen: schwellt das Bild, das er bekommt – mit oder ohne Vorfilter (ohne
  Vorfilter passen andere Werte). Liefert alles, ungefiltert.
- Kerne und Wachstum nur innerhalb der Probe (`ctx.roi_mask`): sonst wächst das dunkle
  Einbettmittel als „Pore“ über den Probenrand und hinterlässt dort Krümel.
- **Mikroporen** (Parameter `spot_sigma_px`, `spot_threshold`; 0 = aus): eine Pore von 1–3 px
  wird durch die Unschärfe so aufgehellt, dass ihr Kern über `seed_threshold` liegt. Sie ist
  aber ein dunkler **Punkt** (in alle Richtungen gekrümmt), das gleich dünne Netzwerk eine
  **Linie**. Maß: kleinerer Eigenwert der Hesse-Matrix (× σ²). Punkte darüber sind zusätzliche
  Kerne und selbst Pore. Größe einer so kleinen Pore ist nur grob messbar.
- Nicht kachelbar (eine Pore muss als Ganzes mit ihrem Kern verbunden sein).

**3. Rahmen**
- Keine Änderung an `core`, Datentypen oder `docs/interfaces.md` (bleibt 0.5).
- `config.yaml`: `background_correction` + `hysteresis` aktiv, `threshold` als Kommentar;
  Postprocessing bleibt vorerst aus. Keine untere Messgrenze.
- Messung mit `syntetic_test/evaluate.py` gegen den bisherigen Stand.

**4. Ergebnis** (50 synthetische Bilder, Postprocessing aus; Werte `seed` 0,40 · `grow` 0,55 ·
`spot_sigma_px` 0,8 · `spot_threshold` 0,21)

| | vorher (`threshold` 0,3) | nachher |
|---|---|---|
| Porositätsfehler (Mittel) | 0,205 pp | 0,073 pp |
| Korrelation GT/Detektion | 0,61 | 0,87 |
| Pixel-F1 | 90,0 % | 93,9 % |
| gezählte Fehlobjekte | 2034 (36 %) | 99 (2,3 %) |
| Mikroporen < 5 µm erkannt | 17 % | 38 % |
| Poren 5–10 µm erkannt | 68 % | 86 % |

- `spot_threshold` niedriger holt mehr Mikroporen, aber ab ~0,18 steigen die Fehlobjekte stark
  (0,18: 47 % / 569; 0,15: 60 % / 4871) – die künstlichen Dunkelflecken sind dann nicht mehr
  von Mikroporen zu trennen (physikalische Grenze bei 1,48 µm/px; besser: stärkeres Objektiv).
- Zeit bei 130 MP: `background_correction` ~1,0 s, `hysteresis` ~1,2 s, mit Mikroporen ~3,2 s
  (Speicherdurchsatz-begrenzt; Rechnen in Streifen brachte nichts).

### 7.5 Proben-KOS (beschlossen und umgesetzt 2026-10-18)

Anlass: Porenkonzept (Spannungsbewertung, Folie `temporäre Sachen/`): Spannung σ = f(z) über
die Probe → Koordinatensystem, das nur auf der Probe liegt. Das KOS beschreibt die Probe, nicht
die Spannung – darum Teil der Probenanalyse, nutzbar für alle späteren Schritte.

- **Neue Stufe `sample.frame`** (genau ein Verfahren, abschaltbar mit `none`), läuft nach
  `roi` und `scale` auf der Probenmaske (ohne Maßstab). Schnittstelle 0.5 → **0.6**:
  Datentyp `SampleFrame`, `ctx.frame`, `ctx.require_frame()`, `ctx.sample_coords()`.
- **Festlegungen:** immer die ganze Probe im Bild, rechteckig, ungefähr waagerecht.
  **Oberkante = außen**, **Unterkante = innen**. x nach rechts entlang der Oberkante, z nach oben,
  Ursprung links unten, rechtwinklig. Rechnen in px (µm mit Maßstab), bezogene Höhe
  z / Dicke (0 = innen, 1 = außen).
- **Verfahren `edge_lines`:** Kantenpunkte der Probe → Seiten nach Kantenrichtung → Ecken
  weglassen (`corner_trim`) → robuste Geraden (Huber).
  - Oberkante = Vorgabe (exakt, dort höchste Spannung).
  - Unterkante = Parallele im Median-Abstand der Unterkantenpunkte (Ausbrüche, Spalt egal).
  - Ursprung = Unterkante ∩ linke Kante; Länge bis zur rechten Kante.
  - Keilwinkel (eigene Gerade der Unterkante) nur zur Kontrolle in der Ausgabe.
  - Probe berührt den Bildrand → Bild schlägt fehl (klare Meldung).
- **Ausgabe:** `images.json`/Excel `frame_origin_px`, `frame_angle_deg`, `frame_thickness_px/_um`,
  `frame_length_px/_um`, `frame_wedge_deg`. Korrekturen im Viewer lesen das KOS daraus zurück.
- **Prüfung** (synthetische Bilder, 31 mit ganzer Probe): Winkel im Mittel 0,26° daneben
  (max 0,7°), Dicke −1,7 px bei ~800 px, Ursprung 3,7 px von der nominellen Ecke (abgerundete
  Ecken), ~6 ms pro 1-MP-Bild.
- **Viewer:** Ebene „Sample frame“ (ein/aus wie die Masken, Standard an) – als Vektorgrafik über
  dem Bild, bei jedem Zoom scharf: Probenrechteck gestrichelt, Achsenpfeile x/z am Ursprung,
  Beschriftung „outer“/„inner“; als Anzahl die Dicke (µm, sonst px). Ohne KOS im Lauf ausgegraut.
  Daten aus den `frame_*`-Spalten von `images.json` (Viewer importiert weiter nichts aus `pore_detection`).
- Spannungsmodul: siehe 7.6.

### 7.6 Porenindex – Porenkonzept (beschlossen und umgesetzt 2026-10-18, erster Schritt)

Auswertungsmodul `evaluation/pore_index.py` (Stufe `evaluation`, braucht das Proben-KOS).

- **Formeln** (Folie „Porenkonzept“): Φ_q = Σ A_i g_i^q / A_Mess;
  Φ_eff,q = w_o Σ_(i∈o) A_i g_i^q / A_Mess,o + w_u Σ_(i∈u) A_i g_i^q / A_Mess,u;
  w_o = (n_o+1)/(n_o+n_u+2), w_u = (n_u+1)/(n_o+n_u+2).
- **Spannung g** über die bezogene Höhe ζ = z / Dicke (0 innen, 1 außen). Standard
  `stress: bending` (Wechselbiegung, g = |2ζ − 1|), alternativ `stress: pulsating` (schwellende
  Beanspruchung laut Folie, g = ζ; ergänzt 2026-10-18). Weitere Lastfälle sind noch nicht festgelegt; später
  ein **beliebiges Spannungsfeld**, das im Launcher eingegeben wird (dann klären, wie das zur Regel
  „config.yaml einzige Quelle“ passt).
- **Spannung einer Pore** (`pore_stress`, genau eine Variante, später Auswahl im Launcher):
  `centroid` (g am Schwerpunkt × Fläche), `area` (jedes Pixel mit seinem g), `max` (höchstes g × Fläche).
- **Zonen:** Grenze bei halber Dicke; A_Mess,o/u = Probenfläche oben/unten. Poren über der Grenze
  werden **pixelweise geteilt**, jeder Teil zählt wie eine eigene Pore.
- **Config:** `stress`, `pore_stress`, `q_values` (1; 1,5; 2; 3; 5; 10), `cracks_outer` (n_o),
  `cracks_inner` (n_u), `hardness_outer` / `hardness_inner` (HBW, leer = nicht angegeben).
- **Ausgabe pro Bild:** `phi_q…`, `phi_eff_q…` je q (z. B. `phi_q1_5`), `w_outer`, `w_inner`,
  `sample_area_outer_px2`, `sample_area_inner_px2`. **Pro Pore:** `zeta` und `stress_centroid`
  (Schwerpunkt der ganzen Pore), `stress_max`, sowie ihr Beitrag A g^q je q
  (`weighted_area_q…_px2`, mit der gewählten Variante; Summe über die Poren / A_Mess = Φ_q).
- **Nur pro Bild** – noch keine Zusammenfassung über die ID (Auswahl mehrerer Schliffe später).
- Geprüft gegen eine unabhängige Rechnung über das volle Pixelraster: alle drei Varianten gleich
  bis auf Rundung (10⁻¹⁵).
- Offen: Lastfälle / beliebiges Spannungsfeld, Eingabe im Launcher (n_o, n_u, Varianten),
  Werte über die ID, Spannungsfeld als Ebene im Viewer, Härte außen/innen.
- **Härte außen/innen** (Folie, 2026-10-18): nur Eingabe (Launcher, config.yaml) und Ausgabe, noch
  ohne Wirkung auf Φ. Offen, ob g = σ/σ_max oder Auslastung σ/σ_W (σ_W aus der Härte) gemeint ist.
  g ist nur in `PoreIndex.relative_stress` definiert; dort ist die Härte für Modelländerungen greifbar.

### 7.9 Ergebnis-Excel nach Themen (beschlossen und umgesetzt 2026-10-18)

- **Eine** Datei `<ID>_results.xlsx`, Blätter nach Thema: **Pores** (Porenanzahl, Flächen,
  Porosität, Probendicke/-länge, manuelle Korrekturen), **Stress** (Eingaben Lastfall/Variante/n_o/n_u,
  Φ_q, Φ_eff,q, w_o, w_u), **Detection** (µm/px, gelesener Maßstab, Drehwinkel, Keilwinkel),
  **ID**, **Errors**. Jedes Blatt beginnt mit der Bildspalte.
- **JSON bleibt vollständig** (Programm, Viewer, Korrekturen). Nur in JSON: Pfad, Pixelpositionen
  von Maßstab/Kasten, KOS-Ursprung, Zonenflächen.
- **Modular:** jedes Auswertungsmodul gibt `excel_sheet` (Blatt, Standard: Modulname) und
  `excel_hidden` (nur JSON) an; die Spalten von `core` sind fest in `pipeline.CORE_EXCEL`.
  Schnittstelle 0.10.
- `pore_index` schreibt seine Eingaben (Lastfall, Variante, n_o, n_u, Härte außen/innen als `hardness_outer_hbw` / `hardness_inner_hbw`) mit in die Werte pro Bild.
- Auch nach einer Korrektur im Viewer wird die Excel so neu geschrieben.

### 7.8 Launcher: Reiter „Evaluations“ (beschlossen und umgesetzt 2026-10-18)

- Eingaben der Auswertungsmodule (z. B. `pore_index`: n_o, n_u, `pore_stress`, q-Werte) im
  Launcher, Reiter „Evaluations“ (`gui/launcher/evaluations.py`).
- **Generisch:** aus den Auswertungsmodulen in `config.yaml` gebaut – Zahl → Zahlenfeld, Liste von
  Zahlen → Textfeld „1, 1.5, 2“, ja/nein → Checkbox, Text mit `choices` des Moduls → Auswahl.
  Neues Modul erscheint ohne GUI-Code. Neues optionales Attribut `Step.choices` (Schnittstelle 0.8).
- **config.yaml = Standardwerte**, wird nie geschrieben. Nur **geänderte** Felder (fett) gelten
  für den nächsten Lauf; „Reset to config.yaml“ setzt zurück. Unveränderte Felder folgen
  `config.yaml`, auch wenn sie sich bei offenem Launcher ändert.
- Beim Start stehen die Änderungen im Log; der Lauf speichert sie in `config_used.yaml`.
- Ungültige Eingabe (z. B. Buchstabe in der q-Liste) → Lauf startet nicht, Meldung.
- **Eingaben pro Bild** (2026-10-18, Schnittstelle 0.14): Werte, die ein Modul als `image_params`
  angibt (`pore_index`: n_o, n_u, Härte außen/innen), stehen nicht als Feld, sondern als kompakte,
  scrollbare **Tabelle im Kasten des Moduls**: eine Zeile pro Bild des ID-Ordners (nur die
  Dateinamen werden gelesen). Startet bei jedem Ordner **leer**; leeres Feld = Wert aus `config.yaml`.
  Vor dem Bild: **grüner Haken** = alle Felder gefüllt, **gelber Kreis** = teilweise, nichts = leer.
  Checkbox **„Same values for all images“**: eine Zeile für alle (Liste ausgegraut); Haken weg →
  die gemeinsamen Werte stehen in jeder Zeile (nur Ausnahmen ändern). Start auch mit leeren Zeilen
  erlaubt (Log: „inputs for x of y images“). Die Werte gehen als `per_image` an den Lauf, stehen in
  `config_used.yaml` und pro Bild in `images.json` / Excel; Korrekturen im Viewer nutzen dieselben.
- Offen: beliebiges Spannungsfeld eingeben; Vorausfüllen der Tabelle aus dem letzten Ergebnis
  (bewusst nicht – bei Bedarf später).

### 7.7 Plots (beschlossen und umgesetzt 2026-10-18)

- **Plots sind Bilder**, erzeugt von der Pipeline, gespeichert im Ergebnisordner
  (`<ID>_<Bild>/plots/<name>.png`), vom Viewer nur angezeigt. Für Berichte ohne GUI nutzbar.
- **Eigene Stufe `plots`** nach `evaluation` (Schnittstelle 0.7). Ein Plot bekommt nur die
  **fertigen Werte** (Zeile aus `images.json`, Tabelle `pores.json`), kennt keine Auswertungsmodule.
- **Modular:** neuer Plot = eine Datei in `pore_detection/plots/` (`@register`, Klasse `Plot`,
  Methode `plot(ax, image, pores)` zeichnet nur in die matplotlib-Achse) + Eintrag unter `plots:`
  in `config.yaml`. Größe, Aussehen, Speichern macht `core/plotting.py` einheitlich.
- **Fehlende Werte** (z. B. `pore_index` aus): Plot wird übersprungen und im Log gemeldet, das Bild
  schlägt nicht fehl; der Viewer zeigt „not available“.
- Gezeichnet beim Schreiben (Hintergrund) und **nach jeder Korrektur neu** (der Viewer bekommt die
  neuen Bilder sofort, die Dateien werden im Hintergrund ersetzt).
- matplotlib (`requirements.txt`) wird nur geladen, wenn Plots konfiguriert sind; fehlt es, bricht
  der Lauf mit klarer Meldung ab.
- **Erste Plots:** `phi_over_q` (Φ_q und Φ_eff,q über q, Porosität als Linie), `pores_over_height`
  (Spannung am Porenschwerpunkt über der Höhe ζ, Punktgröße = Fläche), `pore_sizes`
  (Histogramm der Äquivalentdurchmesser, µm mit Maßstab), `pore_areas` (Größenverteilung der
  Porenfläche wie in V2: log-Klassen nach Freedman-Diaconis, Dichtekurve, D50/D90/Mittel; seit
  2026-10-18 im gemeinsamen Stil (Rahmen, kein Gitter); Rechnung gegen V2 `verteilung.py` geprüft). Steht als
  erster Plot in `config.yaml`.
- **Zweite Ansicht eines Plots** (Schnittstelle 0.9, `Plot.variant`): wird mitgezeichnet und als
  `plots/<name>.<variant>.png` gespeichert; der Viewer zeigt im Plotfeld eine Checkbox zum Umschalten
  (Zustand bleibt beim Blättern). Erste: `pore_areas` → `normalized` (Anteil der Poren in %, Balken
  zusammen 100 %).
- **Aussehen** (für alle Plots, `core/plotting.py`): wissenschaftlich – schwarzer Rahmen rundum,
  kein Gitter, Teilstriche innen auf allen Seiten; weißer Hintergrund (auch im dunklen Viewer).
  Gespeichert ohne Außenrand, 960 × 600 px.
- **Viewer:** Plot über die ganze Breite seines Platzes, Höhe nach Seitenverhältnis; einmal sauber
  verkleinert (scharfe Schrift).
- **Plots über die ganze ID** (2026-10-18): dieselbe Plot-Datei, optionale Methode `plot_id(ax, images,
  pores)` (alle Bilder; Porentabelle mit Spalte `image`), gespeichert in `<ID>_results/plots/`. Gezeichnet
  am Ende eines vollständigen Laufs und nach jeder Korrektur (wie die ID-Werte). Häufigkeitsplots
  (`pore_areas`, `pore_sizes`): **nur Dichtekurven, eine pro Bild, auf denselben Klassen** (aus allen
  Poren der ID). `phi_over_q`: eine Farbe pro Bild (Φ_q durchgezogen, Φ_eff,q gestrichelt);
  `pores_over_height`: alle Poren, Farbe nach Bild. Klassen und Dichte zentral in `core/plotting.py`.
- **Dichtekurven** (2026-10-18): echte Dichte pro Dekade (Kerndichte auf dem Logarithmus), Fläche unter
  der Kurve = Porenanzahl (Ansicht Anzahl: „Poren pro Dekade“) bzw. 100 % (normalized: „% der Poren
  pro Dekade“); Werte > 100 % sind möglich, wenn die Poren enger als eine Dekade liegen. Im Histogramm
  auf einer **zweiten y-Achse rechts** (andere Einheit als die Balken).
- **`critical_pores`** (2026-10-18): Pareto der maßgebenden Poren – Balken = Beitrag A g^q der Top-Poren
  (Nummer wie in der Porenliste, Farbe = Zone außen/innen), Summenkurve auf zweiter Achse; Parameter
  `q` (einer der q_values) und `top`. Über die ID: Poren aller Schliffe, beschriftet Bild/Pore.
  Rechts daneben die **Probe** (gerade ausgerichtet über das KOS, Außenseite oben, auf die Probe
  zugeschnitten mit schmalem Rand Einbettmittel), die Top-Poren eingekreist und mit ihrer Nummer
  beschriftet (Top 5 größer). Dafür neu: `Plot.needs_picture` → `core` liefert das Probenbild
  (`plotting.sample_picture`, nur bei Bedarf) und `Plot.size_in` (breiterer Plot). Schnittstelle 0.12.
  Ohne KOS und im ID-Plot nur der Pareto-Teil.
- Offen: ID-Plots im Viewer anzeigen.

---

## 8. Ausgabe

Verbindlich in **`docs/output_format.md`**. Festgelegt:

```
<Ausgabeordner>/
└── <ID>_results/
    ├── <ID>_<Bild>/              ← je Bild: pores.json, mask.tif, labels.tif, roi_mask.tif, (overlay.tif)
    ├── images.json               ← eine Zeile pro Bild
    ├── id_values.json            ← Kennwerte über die ganze ID (aus finalize)
    ├── errors.json               ← fehlgeschlagene Bilder
    ├── <ID>_results.xlsx         ← Blätter nach Thema: Pores, Stress, Detection, ID, Errors (7.9)
    └── config_used.yaml
```

| Was | Format | Für wen | Wann |
|---|---|---|---|
| Tabellen | **JSON** | Programm (Viewer, `finalize`) | immer |
| Tabellen pro Bild / ID / Fehler | **Excel** (`xlsxwriter`) | Menschen | immer, wird nie wieder eingelesen. Porentabellen nur als JSON |
| Vorschaubilder | – | – | **nie** (der Viewer rechnet seine Übersicht selbst) |
| Maske in voller Auflösung | 8-Bit-TIFF (0/255), PackBits | Viewer-Zoom, Weiterverarbeitung | immer |
| Original in voller Auflösung | – | – | nie (liegt im Eingabeordner) |
| Overlay in voller Auflösung | TIFF | Berichte | nur mit `output.save_full_overlay: true` |

- Schreibmodul ist **fest** in `core` (kein Plug-in). Es ist **generisch**: schreibt alle Spalten
  und Bilder, die die Auswertungsmodule liefern, ohne sie zu kennen.
- Ein neuer Lauf **überschreibt** `<ID>_results/` – gelöscht werden dabei nur Dateien/Ordner, die
  das Programm selbst anlegt.
- Details (Dateinamen, JSON-Aufbau): `docs/output_format.md`.

---

## 9. Phasen

| Phase | Inhalt | Ergebnis | Stand |
|-------|--------|----------|-------|
| 0 | `requirements.txt`, `core/` (types, Step, Registry, Device, Imaging, IO) | Schnittstellen stehen | fertig |
| 1 | Einfache Verfahren pro Stufe (`full_image`, `fixed_scale`, ein Detektor, Labeling, `pore_properties`, eine Auswertung), Bild-für-Bild-Pipeline mit Fehlerbehandlung, Schreibmodul, CLI, vollständige Config | Funktioniert korrekt | fertig |
| 2 | Zuschneiden auf ROI, Tiling + Parallelisierung, dynamische GPU-Nutzung, Vorladen/Hintergrund-Schreiben | Schnell | fertig (5 × 99 MP: 31 s → 9 s); GPU-Weg in `core` + `gauss` als erstes Verfahren |
| 3 | GUI: Launcher + Viewer | Bedienbar, Ergebnisse schnell prüfbar | fertig (Beta): Launcher fertig, Viewer mit Lupe, Overlays, Porenliste, volle Auflösung ohne Vorschaubilder, optische Kontrolle des Maßstabs; offen: Plots, Werkzeugspalte, Maßstab als Ebene |
| 4 | Einbettmittel-Erkennung, Einbindung Maßstabsmodul, Abgleich an echten Bildern | Praxistauglich | ROI-Verfahren `imagej`, `grabcut` vorhanden; Maßstab `scale_bar` eingebunden (99 MP: 0,4 s statt 14 s in V2) |
| 5 | Weitere Verfahren und Auswertungsmodule | Nach Bedarf | Werkzeuge, Filter, Korrekturen (7.3), Untergrundkorrektur + `hysteresis` (7.4), Proben-KOS (7.5), Porenindex pro Bild (7.6), Plots (7.7) umgesetzt; weitere nach Bedarf |

Grundsatz: Erst korrekt (Phase 1), dann schnell (Phase 2).

---

## 10. Noch offen (bewusst später)

1. Laufdaten (Softwareversion, Gerät, Zeiten) in der Ausgabe
2. Konkrete Verfahren (Detektor, Filter, Labeling, Auswertungen)
3. Warnungen (nicht fatale Probleme pro Bild) – vorerst gar nicht vorgesehen
4. Viewer: Bedienung im Detail
5. Typische Panoramagröße → nötiger Arbeitsspeicher
6. Schneller Testlauf zum Einstellen (nur ein Bild oder ein Ausschnitt), damit Anpassen der
   Einstellungen nicht jedes Mal einen vollen Lauf braucht

---

## 11. Risiken

- Verwechslung von Poren mit Artefakten (Kratzer, Ausbrüche, Einschlüsse, Schmutz)
- Einbettmittel hat ähnliche Helligkeit wie Poren → ROI-Maske ist entscheidend
- Ungleichmäßige Ausleuchtung / Vignettierung in gestitchten Panoramen → Helligkeitskorrektur als
  Vorfilter
- Aneinanderliegende Poren werden als eine erkannt → Merge/Split im Postprocessing
- Falsche Reihenfolge im Postprocessing (verändernder Schritt nach Kennwert-Schritt) → veraltete
  Kennwerte, wird nicht automatisch erkannt
- Nahtfehler an Kachelgrenzen, wenn `tile_overlap_px` eines Verfahrens zu klein angegeben ist
- Verschiedene Vergrößerungen: px-Parameter bedeuten je Bild unterschiedliche physikalische Größen
  → wo nötig µm-Parameter verwenden
- Große Panoramen brauchen viel Arbeitsspeicher (z. B. 400 MP: Original 1,2 GB + Graustufen 1,6 GB
  + Labels 1,6 GB; mit Vorladen ein zweites Original) → Mindest-RAM ermitteln
- Falsche Werte in der Config werden nicht abgefangen (bewusste Entscheidung) → Fehler zeigen
  sich erst beim Lauf als übersprungene Bilder
- PyTorch-Wheels für Python 3.14 + CUDA prüfen (betrifft nur die GPU-Wege)
- Eingabeordner nach dem Lauf verschoben → Viewer-Zoom auf das Original geht nicht (unkritisch, da
  Viewer nur direkt nach dem Lauf)
