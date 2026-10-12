# Zonenauswertung

Teilt die Probe in Zonen parallel zur Oberkante und berechnet die Porosität je Zone.
Eigenständiges Unterprogramm: es liest nur die Ausgabe eines Laufs und importiert nichts
aus `poren/`.

## Starten

```powershell
# Ganzer Lauf: Pipeline + Viewer wie start.py, Zonen nach jedem Bild und jeder Korrektur
.venv\Scripts\python.exe start_with_zones.py

# Nur die Zonen, über einen fertigen Zielordner (oder einen einzelnen Bildordner)
.venv\Scripts\python.exe -m zonen "D:\Ergebnisse\lauf_01"
.venv\Scripts\python.exe -m zonen "D:\Ergebnisse\lauf_01" --einstellungen meine_zonen.yaml
```

Im Ordner jedes Bildes entstehen:

| Datei | Inhalt |
|---|---|
| `<bild>_zonen.csv` | Kenngrößen (Dicke, Winkel, Fehler an der Unterkante) und darunter die Zonentabelle |
| `<bild>_zonen.png` | Kontrollbild: Einbettmittel blau, Poren rot, Kantenpunkte (grau = nicht genutzt), Porosität je Zone. Alle Linien gelb: 0 und 100 % kräftig, Zonengrenzen dünn, Mittellinie (50 %) gestrichelt, echte Unterkante gepunktet. Rechts neben dem Bild die Abstände zwischen den Linien und die Gesamtdicke, senkrecht zur Oberkante gemessen, in µm (ohne Maßstab in px) |

Ohne Einbettmittel oder ohne erkennbare Kante steht in der CSV nur der Grund.

## Im Viewer

Nur beim Start über `start_with_zones.py`; mit `start.py` sieht der Viewer aus wie immer.

* **Ebene „Zonen“** (Karte *Ebenen*): Oberkante, Zonengrenzen und 100-%-Linie gelb im
  Bild, Mittellinie gestrichelt, echte Unterkante gepunktet.
* **Karte „Porosität je Zone“**: Diagramm mit der Tiefe senkrecht (Oberkante oben) und
  der Porosität waagerecht. Links stehen die Maßkette der Zonen und die Gesamtdicke,
  darunter Dicke, Gesamtporosität und Fehler an der Unterkante. Ein Klick vergrößert
  die Karte, die Maus über einer Zone zeigt ihre Werte.

Beides wird nach jeder Korrektur neu gerechnet. Die Zonengrenzen werden nur in
`einstellungen_zonen.yaml` eingestellt, nicht im Viewer. Eine Änderung greift beim
nächsten Bildwechsel oder bei der nächsten Korrektur.

Anbindung: `zonen/bruecke.py` (`pro_bild` für die Dateien, `fuer_viewer` für den
Viewer). `start_with_zones.py` trägt beides ein: das eine als Auswertungsmodul, das
andere in `poren/viewer/zusaetze.py`. `zonen/` importiert nichts aus `poren/`. Der
Viewer weiß nur, wie er Zonendaten darstellt, und rechnet sie nicht selbst.

## Einstellungen

`zonen/einstellungen_zonen.yaml`, vor allem:

```yaml
grenzen_pct: [0, 10, 25, 50, 100]   # Zonen 0-10, 10-25, 25-50, 50-100 % der Dicke
```

## Rechnung

1. **Eingelesen** werden das Einbettmittel (`daten/einbettmittel.json`), die gezählten
   Poren mit Korrekturen (`daten/poren.json`) und die Probenfläche (`<bild>_maske.png`,
   ohne Maßstabskasten).
2. **Kantenpunkte:** je Bildspalte der oberste bzw. unterste Probenpunkt, an dem direkt
   Einbettmittel anliegt. Die äußeren `kante.rand_pct` der Probenbreite fallen weg, weil
   dort Ecken und Flanken liegen.
3. **Regression:** RANSAC sucht die Gerade, die die meisten Kantenpunkte trägt. Danach
   werden die kleinsten Quadrate über genau diese Punkte gerechnet. Ausreißer wie
   Luftblasen, Ausbrüche oder Flankenreste kippen die Gerade so nicht.
4. **Dicke T = 100 %:** der mittlere senkrechte Abstand der Unterkantenpunkte von der
   Oberkantengeraden.
5. **Zonen:** Die Grenze bei p % ist die Parallele zur Oberkante im senkrechten Abstand
   p/100 · T. Jeder Pixel zählt in der Zone, in der seine Mitte liegt. Eine Pore auf
   einer Grenze wird also **pixelgenau** auf beide Zonen verteilt.
   Porosität der Zone = Porenfläche / Probenfläche der Zone; Einbettmittel und
   Maßstab zählen nicht mit. Probe außerhalb der ersten bzw. letzten Grenze (weil die
   Unterkante nicht parallel ist) steht als eigene Zeile `< 0 %` bzw. `> 100 %` in der
   Tabelle. Die Zeile `gesamt` entspricht der Porosität in `bilder.csv`.

## Fehler an der Unterkante

Die Zonen setzen voraus, dass die Unterkante parallel zur Oberkante liegt. Wie gut das
stimmt, steht in der CSV. Vorzeichen: **+ heißt, die Probe ist dort dicker als T.**

| Zeile | Bedeutung |
|---|---|
| `fehler_winkel` | Winkel der Unterkanten-Geraden gegen die Oberkante (Grad) |
| `fehler_links`, `fehler_rechts` | Abstand der Unterkanten-Geraden von der 100-%-Linie am linken bzw. rechten Ende der Kante |
| `fehler_max` | größte Abweichung eines einzelnen Kantenpunkts von der 100-%-Linie |
| `fehler_rms` | mittlere (quadratische) Abweichung der Kantenpunkte |

Jede Länge steht auch als `_rel` in % der Dicke T. Längen sind in µm angegeben, ohne
Maßstab in px.
