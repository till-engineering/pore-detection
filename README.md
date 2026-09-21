# Pore Detection V2

Detektion und Vermessung von Poren in metallographischen Schliffbildern.
Liest einen Ordner ein, bestimmt den Maßstab aus dem eingebrannten Balken, trennt die
Probe vom Einbettmittel, vermisst die Poren und erzeugt daraus einen PDF-Bericht.

Der Aufbau ist bewusst modular: jeder austauschbare Baustein — Maßstabserkennung,
Einbettmittel-Trennung, Porendetektion, Filter, Berichtsformat — steckt hinter einem
Protokoll und wird über die Konfiguration per Name ausgewählt. Ein neuer
Detektionsalgorithmus ist eine neue Datei plus ein Eintrag in der YAML.

Der Architekturplan steht in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Stand

Fertig und im Einsatz sind die **Maßstabserkennung** (P1) und die **Trennung Probe /
Einbettmittel** (P2). Alles andere ist dokumentiertes Skelett.

**Maßstab** — an den 14 eingemessenen Testbildern: 14 von 14 erkannt, maximale Abweichung
0,395 % gegenüber der ImageJ-Kalibrierung. Diese 0,395 % stecken im Bild, nicht in der
Messung: bei `1.035 m` und 97 px/m müsste der Balken 100,4 px lang sein, ImageJ zeichnet
ihn als 100 px.

**Einbettmittel** — sechs Verfahren gebaut und am Vergleichssatz gegeneinander laufen
lassen; als Standard gesetzt ist **`grabcut`**. An den 12 Schliffen mit Harz am Rand
werden alle 12 getrennt. Die Kehrseite dieser Wahl ist gemessen: an 14 Aufnahmen *ohne*
Einbettmittel meldet `grabcut` in 5 Fällen welches, `lasso` nur in einem. Beides hält
`tests/benchmark/test_specimen_real.py` fest.

## Einrichtung

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Es wird ausschließlich in dieser `.venv` gearbeitet.
Optional `pip install -e ".[tesseract]"` für die zweite OCR-Meinung; sie braucht eine
installierte Tesseract-Binary und ist nicht zwingend.

## Maßstab lesen

```powershell
python -m poredet scale "test\Maßstäbe_Bilder" -o data\runs\uebersicht.png
```

Schreibt eine Tabelle auf die Konsole und ein Übersichtsblatt zur Sichtprüfung: je Bild
der Fundort, der Ausschnitt mit dem vermessenen Balken und das Ergebnis. Liegt in einem
TIFF eine ImageJ-Kalibrierung, wird zusätzlich die Abweichung dagegen ausgewiesen
(`--no-reference` schaltet das ab). `--json` legt die Ergebnisse maschinenlesbar ab.

So arbeiten spätere Module damit:

```python
from poredet.io.image_reader import read_image
from poredet.scale import ScaleResolver, overlay_mask

image = read_image("probe.tif")
outcome = ScaleResolver().resolve(image.gray, image.path)

if outcome.scale:
    durchmesser_um = outcome.scale.to_um(durchmesser_px)
    flaeche_um2    = outcome.scale.to_um2(flaeche_px)
    nicht_werten   = overlay_mask(image.gray.shape, outcome.scale, pad=4)
```

Ohne erkannten Maßstab ist `outcome.scale` `None` — dann wird in Pixeln weitergemessen
und die physikalischen Werte bleiben leer. Es wird nirgends ein Faktor unterstellt.

## Einbettmittel entfernen

```powershell
python -m poredet specimen "test\Einbett_Material" -o data\runs\uebersicht.png
```

Das Übersichtsblatt zeigt je Bild vier Dinge: das Original, die erkannte Grenze, die
freigestellte Probe und die Zwischenschritte. Die letzte Spalte ist die wichtigste, wenn
etwas schiefgeht — sie zeigt, *welches* Kriterium versagt hat.

```python
from poredet.specimen import segment
from poredet.scale import overlay_mask

mask = segment(image.gray, color=image.color)   # Standardverfahren: grabcut
bezugsflaeche_px = mask.specimen_area_px        # Nenner der Porosität, Poren eingeschlossen
auswertbar       = mask.specimen & ~overlay_mask(mask.shape, scale, pad=4)
```

`color` ist optional, sollte aber übergeben werden, wo es vorliegt: GrabCut schätzt seine
Modelle über die Farbkanäle. Die anderen Verfahren ignorieren es.

Ein Maßstab wird dafür nicht gebraucht — viele Schliffbilder kommen ohne Balken an.
Findet sich kein Einbettmittel, ist `mask.resin` leer und das ganze Bild gilt als Probe.

### Verfahren vergleichen

Sechs Segmentierer stehen zur Wahl, alle über `specimen.method` austauschbar.
**Standard ist `grabcut`** — es wird von allen weiteren Modulen verwendet:

| Verfahren | Ansatz |
|---|---|
| **`grabcut`** (Standard) | Graph-Schnitt mit selbst gelernten Farbmodellen |
| `lasso` | Schwellen auf Helligkeit und Textur, dann Randoffenheit und Materialprüfung |
| `random_walker` | Diffusion von Saatpunkten aus, global gelöst |
| `watershed` | Wasserscheide zwischen zwei Saaten auf dem Gradientenbild |
| `chan_vese` | Levelset-Kontur auf Regionenstatistik |
| `boundary_path` | die Trennlinie als billigster Weg quer durchs Bild |

```powershell
python -m poredet specimen-compare "test\Einbett_Material" -o data\runs\vergleich.png
```

Alle Verfahren rechnen auf **denselben** Signalen (`specimen/signals.py`) und durchlaufen
**denselben** Abschluss (`specimen/postprocess.py`). Unterschiede im Ergebnis stammen
damit aus der Segmentierung und nicht aus unterschiedlich gründlicher Vorverarbeitung.

## Aufbau

| Paket | Aufgabe |
|---|---|
| `core/` | Datenmodell, Plugin-Registry, Pipeline, Cache, Events — ohne I/O und Framework |
| `config/` | Pydantic-Schema, Profile, aufgelöste Laufkonfiguration |
| `io/` | Bilder lesen, Ergebnisse exportieren, Run-Store |
| `preprocessing/` | Beleuchtung, Rauschen, Normierung |
| `scale/` | Maßstabsbalken finden, OCR, Plausibilität, manuelle Korrektur |
| `specimen/` | Probe vom Einbettmittel trennen — liefert die Bezugsfläche der Porosität |
| `detection/` | Porendetektion (austauschbare Algorithmen) |
| `measurement/` | Kennwerte je Pore, je Bild, Verteilungen |
| `analysis/` | Filter, räumliche Auswertung, Statistik, Grenzwerte |
| `reporting/` | ReportModel, Diagramme, HTML-Vorschau, PDF |
| `api/` + `web/` | FastAPI und serverseitig gerenderte Oberfläche |
| `cli/` | Kommandozeile |
| `dev/` | Synthetische Testbilder, Benchmarks |
