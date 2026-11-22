# Plan – Korrekturfenster „Check“ im Viewer

> Stand: 2026-11-20 · Status: umgesetzt (synthetisch getestet), Abgleich am Mikroskop offen

## Ziel

Schnell und vollständig kontrollieren, ob alle Poren eines Bildes richtig erkannt wurden: Ausschnitte
in fester Größe nacheinander ansehen und direkt darin korrigieren.

## Entscheidungen

### Score pro Pore (Pipeline)

- Das U-Net liefert zusätzlich zur Maske seine Wahrscheinlichkeitskarte (optionale Erweiterung der
  Detektor-Schnittstelle; andere Detektoren unverändert). `core` setzt sie aus den Kacheln zusammen.
- Score einer Pore = **mittlere** Wahrscheinlichkeit über ihre Pixel, nach dem Postprocessing,
  nur für **gezählte** Poren. Spalte `unet_score` in `pores.json`. Die Karte wird nicht gespeichert.
- Von Hand eingezeichnete oder mit „+“ zurückgeholte Poren haben keinen Score.

### Viewer

- Knopf „Light / dark“ entfällt (Design nur noch aus `gui.theme`), an seiner Stelle „Check“.
- „Check“ öffnet ein **modales** Fenster für das gezeigte Bild. Der Viewer ist solange gesperrt.
- Korrekturen im Fenster gehen nicht live in den Viewer: Fenster schließen („Done“ oder X, beides
  gleich, Korrekturen bleiben immer) → der Viewer lädt das Bild neu.

### Korrekturfenster

- Kopf: Modus-Auswahl, Fortschritt („214 / 340 pores checked“ bzw. „tile 12 / 48“), in Modus 1
  Nummer und Score der Pore in der Mitte („Pore 57 · score 0.62“).
- Werkzeugleiste: „+“, „−“, „✎“, „⬚“, „↶“ (Strg+Z). Kein Zoom, kein Rahmen-Werkzeug, kein Reset.
- Ausschnitt in fester Größe `gui.check_patch_px` (Standard 1000 × 1000 px), immer ganz sichtbar.
  - Alle gezählten Poren nur mit feinem rotem Umriss (ohne Füllung), Deckkraft per Regler.
  - Pore in der Mitte (Modus 1) nicht extra markiert: sie liegt immer in der Mitte des Ausschnitts.
  - Verworfene Poren per Haken ein-/ausblendbar (Standard an), Farbe wie im Viewer.
  - Von Hand hinzugefügte Poren in der Farbe des Viewers; von Hand entfernte Poren werden gar nicht mehr umrandet.
  - Am Bildrand mit Grau aufgefüllt, die Pore bleibt in der Mitte.
- Übersicht: ganzes Bild verkleinert mit Rechteck um den Ausschnitt; schon angeschaute Ausschnitte (beide Modi zusammen) grün umrandet.
- Bedienung: „Next“ / „Back“ und Pfeiltasten → / ←; „Back“ geht im Verlauf zurück.
  Nach einer Korrektur bleibt der Ausschnitt stehen.
- **Rücktaste** (nur Modus 1): verwirft nur die Pore in der Mitte (wie „−“ auf sie).
- **Entf** (beide Modi): verwirft alle gezählten Poren, die ganz im sichtbaren Ausschnitt liegen (wie „⬚“
  mit dem Rechteck des Ausschnitts); angeschnittene Poren am Rand bleiben.
- Nach Rücktaste / Entf bleibt der Ausschnitt stehen.
- Alles durch: Meldung („All pores checked“ / „All tiles checked“), das Fenster bleibt offen.
- Kontroll-Stand nur solange das Fenster offen ist (nicht gespeichert), getrennt pro Modus.

### Modus 1 – nach Score

- Nur gezählte Poren mit `unet_score`, aufsteigend (unsicherste zuerst).
- Die unkontrollierte Pore mit dem kleinsten Score kommt in die Mitte; jede Pore, die **ganz** im
  Ausschnitt liegt, gilt danach als kontrolliert.
- Ohne `unet_score` (klassischer Detektor, alter Lauf) ausgegraut.

### Modus 2 – Kacheln

- Kachelgröße = `gui.check_patch_px`, Überlappung `gui.check_tile_overlap` (Standard 0.1).
- Schlangenlinie (Zeile 1 links → rechts, Zeile 2 rechts → links, …).
- Kacheln ohne Probe (nur Einbettmittel) werden übersprungen.

### Config (`gui:`)

```yaml
gui:
  check_patch_px: 1000      # edge of the check window's patch / tile in image px
  check_tile_overlap: 0.1   # overlap of the tiles in mode 2, share of the tile edge
```

## Umsetzung

1. Score in der Pipeline – erledigt: `Detector.score_column` / `score_threshold` (Schnittstelle 0.17),
   Kacheln setzen die Karte zusammen, `core.pipeline.pore_scores` (Mittelwert in Streifen),
   `unet_score` in `pores.json`; die Korrekturen behalten die Scores des Laufs; `detect_mask()` für
   `tools/find_settings.py` und `dataset_eval/tune.py`
2. Korrekturfenster `gui/viewer/check.py` – erledigt
3. Viewer: „Check“ statt „Light / dark“, danach Bild neu laden – erledigt
4. Test mit synthetischen Daten – erledigt; Abgleich am Mikroskop durch den Nutzer – offen
