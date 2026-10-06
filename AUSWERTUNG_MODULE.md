# Auswertungsmodule nachrüsten

Diese Anleitung beschreibt, wie man eine **zusätzliche Auswertung** einbaut, deren
Ergebnis im Zielordner landet (CSV, Bild, Text …). Die GUI muss dafür nicht angefasst
werden. *Tipp: Diese Datei als Kontext an Copilot geben („Halte dich an
AUSWERTUNG_MODULE.md“).*

---

## 1. Kurzfassung

1. `poren/auswertungen/_vorlage.py` kopieren, z. B. nach `poren/auswertungen/groessenklassen.py`
   (Dateiname **ohne** `_` am Anfang, nur Kleinbuchstaben/Ziffern/`_`).
2. Funktion `pro_bild` und/oder `gesamt` ausfüllen.
3. Ohne GUI testen:
   ```powershell
   .venv\Scripts\python.exe auswertung_testen.py groessenklassen "test\real_pores\6_cropped_1_1.tif"
   ```
4. Fertig. Beim nächsten Lauf über das Startfenster wird das Modul automatisch mit ausgeführt.

Es muss **nichts registriert** werden – jede Datei in `poren/auswertungen/` wird gefunden.

---

## 2. Der Vertrag

Ein Modul ist eine Python-Datei mit einer oder beiden dieser Funktionen:

```python
from pathlib import Path
from poren.auswertungen import BildDaten
from poren.modelle import ImageResult

def pro_bild(bild: BildDaten, ordner: Path) -> None:
    """Wird für JEDES Bild aufgerufen."""

def gesamt(bilder: list[ImageResult], ordner: Path) -> None:
    """Wird EINMAL am Ende über alle Bilder aufgerufen."""
```

| | `pro_bild` | `gesamt` |
|---|---|---|
| Wann? | nach jedem ausgewerteten Bild | am Ende des Laufs |
| Erneut? | nach jeder Korrektur **dieses** Bildes im Viewer | nach jeder Korrektur im Viewer |
| Bekommt | alle Daten eines Bildes inkl. Bildpixeln und Masken | die Kennzahlen aller Bilder (ohne Pixel) |

`ordner` ist `<Zielordner>/auswertungen/<modulname>/` und existiert bereits.

**Regeln:**
- **Nur in `ordner` schreiben.** Keine anderen Dateien im Zielordner oder Projekt ändern.
- **Dateien überschreiben, nicht anhängen.** Das Modul läuft nach Korrekturen erneut;
  ein Dateiname je Bild (`f"{Path(bild.name).stem}.csv"`) bzw. fester Name in `gesamt`.
- **Eingabedaten nicht verändern.** Arrays bei Bedarf mit `.copy()` kopieren.
- **Schnell bleiben.** Es läuft nach jedem Klick im Viewer mit. Teures nur in `gesamt`
  oder einmal je Bild, keine Schleife über alle Pixel in Python (numpy/OpenCV nutzen).
- **Ein Fehler bricht den Lauf nicht ab**, er landet in `<Zielordner>/auswertungen/fehler.txt`.
  Beim Testen mit `auswertung_testen.py` sieht man den Traceback direkt.
- Abschalten ohne Löschen: `AKTIV = False` ganz oben ins Modul schreiben.
- Keine Imports aus `poren.viewer` oder `start.py` – das Modul ist unabhängig von der GUI.

---

## 3. Welche Daten gibt es?

### 3.1 `BildDaten` (in `pro_bild`)

| Feld | Typ | Bedeutung |
|---|---|---|
| `name` | `str` | Dateiname des Bildes, z. B. `"probe_01.tif"` |
| `pfad` | `Path` | Pfad zum Originalbild |
| `ergebnis` | `ImageResult` | Kennzahlen des Bildes (siehe 3.3) |
| `poren` | `list[Pore]` | **gezählte** Poren – nach Filtern und manuellen Korrekturen |
| `verworfen` | `list[RejectedPore]` | verworfene Objekte: `.pore`, `.filter_name`, `.reason` |
| `grau` | `np.ndarray uint8 (H, W)` | Graubild |
| `farbe` | `np.ndarray uint8 (H, W, 3)` oder `None` | Farbbild in **BGR** (OpenCV), `None` bei Graubildern |
| `labels` | `np.ndarray int32 (H, W)` | Label-Bild: `0` = keine Pore, sonst Nummer = `Pore.label` |
| `probe` | `np.ndarray bool (H, W)` | auswertbare Probenfläche (ohne Einbettmittel, ohne Maßstabskasten) |
| `einbettmittel` | `np.ndarray bool (H, W)` | erkanntes Einbettmittel |
| `overlay` | `np.ndarray bool (H, W)` | Maßstabskasten (aus der Auswertung ausgeschlossen) |
| `massstab` | `ScaleInfo` oder `None` | erkannter Maßstab (`.um_per_px`, `.label_text`, …) |
| `um_pro_px` | `float` oder `None` | Mikrometer je Pixel; **`None`, wenn kein Maßstab erkannt wurde** |

Wichtig zu `labels`: Es enthält **alle** Objekte, auch verworfene. Nur die Nummern aus
`bild.poren` sind gezählte Poren. Maske aller gezählten Poren:

```python
import numpy as np
gezaehlt = np.zeros(int(bild.labels.max()) + 1, dtype=bool)
gezaehlt[[p.label for p in bild.poren]] = True
maske = gezaehlt[bild.labels]                    # bool (H, W)
```

Maske einer einzelnen Pore (schnell, nur im umschließenden Kasten):

```python
ys, xs = pore.bbox.slices()                       # Slices (Zeilen, Spalten)
teil = bild.labels[ys, xs] == pore.label          # bool im Kasten
```

### 3.2 `Pore` (`poren/modelle.py`)

Alle Geometrie in **Pixeln**; die `_um`-Werte sind **`None` ohne Maßstab**.

| Feld / Eigenschaft | Bedeutung |
|---|---|
| `label` | Nummer im Label-Bild |
| `area_px`, `area_um2` | Fläche |
| `perimeter_px`, `perimeter_um` | Umfang |
| `equivalent_diameter_px`, `equivalent_diameter_um` | Durchmesser des flächengleichen Kreises |
| `feret_max_px`, `feret_max_um` | größter Feret-Durchmesser |
| `major_axis_px`, `minor_axis_px` (+ `_um`) | Achsen der angepassten Ellipse |
| `centroid_px` | Schwerpunkt `(x, y)`; `centroid_um` in µm |
| `bbox` | umschließender Kasten `.x .y .w .h`, `.slices()` |
| `circularity` | 4πA/U², 1 = Kreis |
| `aspect_ratio` | Haupt-/Nebenachse (kann `inf` sein!) |
| `solidity` | Fläche / konvexe Hülle |
| `eccentricity`, `orientation_deg` | Ellipsenform und -winkel |
| `mean_intensity`, `min_intensity` | Grauwerte in der Pore |
| `contrast` | wie viel dunkler als der Untergrund (Graustufen) |
| `touches_image_edge`, `touches_specimen_edge` | angeschnitten am Bild- bzw. Probenrand |

### 3.3 `ImageResult` (in `gesamt` und als `bild.ergebnis`)

| Feld / Eigenschaft | Bedeutung |
|---|---|
| `name`, `path` | Bildname / Pfad |
| `width`, `height` | Bildgröße in px |
| `scale` | `ScaleInfo` oder `None` |
| `pores`, `rejected` | wie oben |
| `pore_count` | Anzahl gezählter Poren |
| `porosity_pct` | Porosität in % (bezogen auf die Probenfläche), `None` ohne Probe |
| `porosity_pct_excl_edge` | dito ohne angeschnittene Poren |
| `specimen_area_px`, `specimen_area_mm2` | Probenfläche (mm² `None` ohne Maßstab) |
| `pore_density_per_mm2` | Poren je mm² (`None` ohne Maßstab) |
| `largest_pore` | größte Pore oder `None` |
| `warnings` | Warnungen der Auswertung (Liste von Texten) |

In `gesamt` gibt es **keine Bildpixel** – nur diese Kennzahlen. Braucht man Pixel über
alle Bilder, in `pro_bild` je Bild ein Zwischenergebnis in `ordner` schreiben und in
`gesamt` wieder einlesen.

---

## 4. Konventionen

- **Einheiten:** Mit Maßstab in µm rechnen, ohne Maßstab in px – und das in Spaltennamen
  kenntlich machen (`_um` / `_px`). Niemals einen Faktor annehmen, wenn `um_pro_px is None`.
- **CSV:** Semikolon als Trenner, Dezimalkomma, `encoding="utf-8-sig"` – dann öffnet Excel
  die Datei direkt richtig:
  ```python
  with datei.open("w", newline="", encoding="utf-8-sig") as f:
      w = csv.writer(f, delimiter=";")
      w.writerow(["bild", "wert"])
      w.writerow([bild.name, f"{wert:.4g}".replace(".", ",")])
  ```
- **Bilder speichern:** Pfade können Umlaute enthalten – `cv2.imwrite` scheitert daran.
  Stattdessen:
  ```python
  ok, puffer = cv2.imencode(".png", bild_bgr)
  puffer.tofile(str(ordner / f"{Path(bild.name).stem}.png"))
  ```
- **Diagramme** (falls matplotlib genutzt wird): `matplotlib.use("Agg")` vor `pyplot`
  setzen, Figur nach dem Speichern mit `plt.close(fig)` schließen. Neue Pakete in
  `requirements.txt` eintragen.
- **Große Bilder:** Bilder können sehr groß sein (zusammengesetzte Aufnahmen). Keine
  vollen Kopien in float64 anlegen, wenn es nicht nötig ist.

---

## 5. Beispiel: Ergebnisbild mit Porennummern

```python
"""Nummeriert jede gezählte Pore im Bild - zum Abgleich mit poren.csv."""
from pathlib import Path

import cv2

from poren.auswertungen import BildDaten


def pro_bild(bild: BildDaten, ordner: Path) -> None:
    ausgabe = (bild.farbe if bild.farbe is not None
               else cv2.cvtColor(bild.grau, cv2.COLOR_GRAY2BGR)).copy()
    for pore in bild.poren:
        x, y = (int(v) for v in pore.centroid_px)
        cv2.putText(ausgabe, str(pore.label), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
    ok, puffer = cv2.imencode(".png", ausgabe)
    puffer.tofile(str(ordner / f"{Path(bild.name).stem}_nummern.png"))
```

---

## 6. Prompt-Vorlage für Copilot

> Erstelle ein neues Auswertungsmodul `poren/auswertungen/<name>.py` nach
> `AUSWERTUNG_MODULE.md`. Es soll: <was genau berechnet / ausgegeben werden soll>.
> Halte dich an den Vertrag (Funktionen `pro_bild(bild, ordner)` und/oder
> `gesamt(bilder, ordner)`), schreibe nur in `ordner`, beachte `um_pro_px is None`,
> CSV mit Semikolon und Dezimalkomma. Ändere keine anderen Dateien.

Test danach:

```powershell
.venv\Scripts\python.exe auswertung_testen.py <name> "<pfad zu einem Bild>"
```

Die Ausgabe liegt dann in `test_ausgabe\auswertungen\<name>\`.
