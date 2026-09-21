# Architektur

## Leitgedanken

1. **Der Core kennt kein Framework.** `core/` und alle Algorithmus-Pakete importieren
   weder FastAPI noch eine UI. Die Web-Oberfläche ist ein Konsument der Bibliothek,
   genau wie die CLI.
2. **Algorithmen sind Plugins, keine `if`-Zweige.** Jeder austauschbare Baustein
   implementiert ein Protokoll und registriert sich unter einem Namen; die Config wählt
   per Name aus.
3. **Jeder Lauf ist reproduzierbar.** Der Run-Ordner enthält die aufgelöste
   Konfiguration, Versionen, Ergebnisse und Kontrollbilder. Der Bericht entsteht aus
   diesem Ordner, nicht aus dem Arbeitsspeicher.
4. **Zwischenschritte sind Produkt, nicht Abfall.** Jede Stage legt ihre Masken und
   Debug-Overlays im Kontext ab — das ist die Grundlage für GUI, Bericht und Fehlersuche.

## Getroffene Entscheidungen

| Thema | Entscheidung | Begründung |
|---|---|---|
| Verhältnis zu V1 | Kompletter Neuanfang | V1 (`..\Pore_Detection`) bleibt nur als fachliche Referenz |
| Oberfläche | FastAPI + Browser | — |
| Frontend | Jinja2 serverseitig, schlichtes JS ohne Build-Schritt | Vorgabe „nur Python" — keine zweite Toolchain |
| Bericht | PDF über ReportLab | rein pip-installierbar, keine externe Binary |
| Maßstab | ausschließlich eingebrannter Balken + OCR | Vorgabe; Metadaten nur als Ground Truth im Benchmark |
| Einbettmittel | **grabcut** als Standardverfahren, fünf weitere bleiben wählbar | am Vergleichssatz ausgewählt; deckt den Saum am vollständigsten ab |
| … Kehrseite | grabcut erfindet Harz in 5/14 Bildern ohne welches (lasso: 1/14) | bewusst in Kauf genommen, per Test festgehalten |
| OCR | `rapidocr-onnxruntime`, hinter `OcrEngine`-Protokoll | pip-installierbar; Tesseract bleibt als Adapter möglich |

## Die zentralen Verträge

```python
# core/stage.py
class Stage(Protocol):
    name: str
    def run(self, ctx: PipelineContext) -> None: ...   # mutiert ctx

# scale/base.py
class ScaleDetector(Protocol):
    def detect(self, gray, cfg: ScaleConfig) -> list[OverlayCandidate]: ...
class OcrEngine(Protocol):
    def read(self, patch) -> OcrResult: ...

# specimen/base.py
class SpecimenSegmenter(Protocol):
    def segment(self, gray, cfg: SpecimenConfig) -> SpecimenMask: ...

# detection/base.py
class PoreDetector(Protocol):
    def detect(self, ctx: PipelineContext) -> LabelImage: ...

# analysis/filters.py
class PoreFilter(Protocol):
    def keep(self, pore: Pore, ctx) -> FilterDecision: ...   # inkl. Verwerfungsgrund

# reporting/renderers/base.py
class ReportRenderer(Protocol):
    def render(self, model: ReportModel, out: Path) -> Path: ...
```

`PipelineContext` transportiert Originalbild, vorverarbeitetes Bild, Masken
(`specimen`, `excluded`, `pores`), `ScaleInfo`, Porenliste, Warnungen und
Debug-Artefakte. Stages lesen daraus und schreiben hinein — sie kennen einander nicht.

## Ablauf je Bild

```
laden → Maßstab erkennen → Overlay ausschließen → Beleuchtung/Rauschen
      → Probenmaske (Einbettmittel entfernen) → Porenkandidaten
      → vermessen → filtern → Zusatzanalysen → Ergebnis
```

Die Maßstabserkennung steht **vor** der Beleuchtungskorrektur: das eingebrannte Overlay
lebt von absoluten Grauwerten, die eine Flatfield-Korrektur verschieben würde.

## Fachliche Kernpunkte

**Maßstab (umgesetzt).** Die Suche läuft **vom Balken aus**, nicht vom Kasten. Der
naheliegende Weg — erst den weißen Kasten finden — scheitert genau dort, wo er gebraucht
wird: bei hellem Gefüge verschmilzt der Kasten mit dem Untergrund. Gemessen an den
Testbildern findet er 6 von 14 Overlays.

Der Balken dagegen ist ein Objekt, das im Schliff nicht zufällig entsteht: ein massiver,
waagerechter, sehr breiter schwarzer Block, ringsum von Weiß umgeben. Er wird in allen 14
Bildern gefunden, mit genau einem Kandidaten je Bild. Von ihm aus wird der Kasten
aufgespannt — über eine Eigenschaft, die das Gefüge nie hat: **der Kasten ist einfarbig
reines Weiß, exakt 255.** Gefüge besteht aus Zwischentönen. Der Kasten ist damit eine
reinweiße Fläche mit Löchern (Balken, Schrift); werden die Löcher gefüllt, ist er ein
massives Rechteck. Das Füllen ist nötig, weil die Schrift **kantengeglättet** ist: ein
zeilenweises Wachsen bricht an den Zwischentönen der Buchstabenränder ab.

Die Beschriftung wird nur *innerhalb* des Kastens gesucht — so kann kein Gefüge in den
OCR-Ausschnitt geraten. Der Balken wird darin weiß übermalt, nicht weggeschnitten: liegt
er mit im Bild, liest die OCR ihn als Zeichen mit (aus „47 km" wurde „471 km", Faktor
zehn daneben), aber ein eng beschnittener Ausschnitt verschlechtert die Erkennung.

**Gelesen wird abgestimmt, nicht nach Rang.** Keine einzelne Kombination aus Engine und
Vergrößerung liest alle 14 Bilder richtig — die beste erreicht 13. Die Fehler von
RapidOCR und Tesseract sind aber unkorreliert, und ihre Mehrheit über mehrere
Vergrößerungen liegt durchweg richtig. „Erster Treffer gewinnt" war hier der eigentliche
Konstruktionsfehler: es liefert stillschweigend ein falsches Ergebnis, sobald die erste
Engine etwas Falsches liefert, das sich parsen lässt. Abgebrochen wird, sobald ein Wert
genug Stimmgewicht ohne Gegenstimme hat — der Normalfall bleibt schnell.

Herkunft, Konfidenz und überstimmte Lesungen wandern bis in den Bericht. Die manuelle
Korrektur ist Kernfeature, keine Notlösung. Der Overlay-Bereich wird maskiert, sonst
zählt der schwarze Balken als riesige Pore.

Weitere Bauformen (Geräte-Infoband, freistehender Balken ohne Kasten) sind als Detektoren
vorgesehen, aber noch nicht gebaut — in den vorhandenen Bildern kommen sie nicht vor.

**Einbettmittel (umgesetzt).** Harz und Poren sind gleich dunkel — keine Grauwertschwelle
trennt sie. Genutzt werden deshalb zwei andere Unterschiede.

*Harz ist strukturlos, Gefüge nicht.* An den zehn Testbildern liegt das Harz je Bild 61
bis 178 Graustufen unter der Probe und ist 2,3- bis 8,6-mal glatter. Dieses Signal trennt
Harz von **dunklem Gefüge** — von Perlitbändern und Ätzkontrast, die dunkel, aber nie
strukturlos sind. Zwei Details entscheiden darüber, ob es trägt: die Textur wird in
**voller Auflösung** gemessen (Verkleinern mittelt die feine Gefügestruktur weg und macht
ein Perlitband künstlich glatt — der Unterschied schrumpft von 3,2 : 9,9 auf 4,5 : 6,4),
und ihre Schwelle wird **nur über die dunklen Pixel** bestimmt. Über das ganze Bild
gerechnet landet Otsu im Ausläufer der Verteilung; an einem Testbild galten damit 99,5 %
als strukturlos und das Kriterium war wirkungslos.

*Harz ist nach außen offen, Poren sind es nicht.* Das Sichtfeld liegt innerhalb des
Einbettlings, also läuft Harz über den Bildrand hinaus. Ein Feld muss groß genug sein und
**am Rand entlang** liegen; eine angeschnittene Pore berührt ihn nur kurz. Kommen mehrere
Felder vor, darf keines bis in die Bildmitte reichen — dann durchquert die Probe das Bild
und teilt den Saum in Randbänder. Ein einzelnes Feld darf das sehr wohl: die Probe kann
ein Span in der Bildecke sein.

Darauf folgen zwei Schritte aus derselben Einsicht — **Harz ist ein Werkstoff, kein
Helligkeitsbereich**: das Aussehen des Fundes wird an seinem Kern gemessen und der Rest
daran gehalten, und zum Schluss wird geprüft, ob er sich vom Rest des Bildes überhaupt
abhebt. Ohne diese Abnahme meldete das Verfahren in 10 von 14 Bildern ohne jedes
Einbettmittel welches; mit ihr in einem.

Eingeschlossene Inseln — typisch große Luftblasen im Harz — werden dem Material
zugeschlagen, dem sie in der Helligkeit näher stehen. Ohne diese Zuordnung wird eine
Blase zum eigenen „Probenstück" und später zur größten Pore des Bildes.

Die Probenmaske ist zugleich die Bezugsfläche der Porosität und **schließt die Poren
ein** — sie sind von Metall umschlossen und liegen damit innerhalb der Maske.

**Sechs Wege, eine Frage — Standard ist `grabcut`.** Neben `lasso` stehen fünf nicht
schwellenbasierte Verfahren zur Wahl: Random Walker, Marker-Watershed, GrabCut,
Chan-Vese und Grenzverfolgung als kürzester Pfad. Ausgewählt und für alle weiteren
Module gesetzt ist **GrabCut** (`SpecimenConfig.method`), weil es den Saum am
vollständigsten abdeckt und auch die beiden Fälle fängt, an denen `lasso` zu wenig
findet. Die Kehrseite ist gemessen und per Test festgehalten: in der Gegenprobe an
Bildern ohne Einbettmittel meldet GrabCut in 5 von 14 Fällen welches, `lasso` nur in
einem. Umstellen ist ein Konfigurationswert, kein Eingriff. Alle rechnen auf denselben Signalen (`signals.py`), starten von derselben
Saat (`seeds.py`) und durchlaufen denselben Abschluss (`postprocess.py`) - nur so misst
ein Vergleich die Verfahren und nicht ihre Vorverarbeitung.

Eine Regel ist dabei aus `lasso` in den gemeinsamen Abschluss gewandert, weil sie keinem
Verfahren gehört: **Harz berührt den Bildrand.** Was von Metall umschlossen ist, ist eine
Pore. Die rein erscheinungsbasierten Verfahren brauchen sie besonders dringend - GrabCut
lernt das Aussehen des Harzes und findet es in jeder Pore wieder; ohne diese Regel
entfernte es an einem synthetischen Prüfbild 98 % der Porenfläche.

Bekannte Grenze: ein dunkles Band *innerhalb* der Probe, das oben und unten den Bildrand
berührt und sich kaum vom Harz unterscheidet, wird mit abgetrennt. Die Probe zerfällt
dann in zwei Stücke — sichtbar an `components`, und deshalb steht die Zahl im
Kontrollbild.

**Messgrößen.** Je Pore: Fläche, äquivalenter Durchmesser, Umfang, Feret max/min,
Rundheit, Seitenverhältnis, Solidität, Grauwerte, Randberührung, Randabstand. Je Bild:
Porosität % (mit/ohne angeschnittene Poren), Anzahl, Dichte [1/mm²], größte Pore,
Größenverteilung, D50/D90, Größenklassen.

**Parametertuning.** Der wichtigste Bildschirm der GUI ist nicht der Batch-Lauf, sondern
das Einzelbild mit umschaltbaren Overlays und Live-Parametern. Das funktioniert nur mit
dem Stage-Cache — deshalb steht er in P0 und nicht als spätere Optimierung.

## Phasen

| Phase | Inhalt | Ergebnis |
|---|---|---|
| P0 | Skelett, Config, Modelle, Registry, Pipeline, Cache, Events, Bild-I/O, Run-Store, CLI | `poredet run` läuft mit Dummy-Stage durch |
| **P1** | **Maßstab: Balken- und Kastenerkennung, OCR-Abstimmung, Resolver, Benchmark, Kontrollbild** | **fertig — 14/14, max. 0,395 %** |
| **P2** | Beleuchtung/Rauschen + **Probenmaske (fertig)** | **10/10 getrennt, 13/14 Gegenprobe sauber** |
| P3 | Porendetektion (2 Algorithmen) + Messgrößen + Filter | CSV/JSON mit echten Zahlen |
| P4 | Statistik, Verteilungen, Grenzwerte, PDF-Bericht | Bericht als Deliverable |
| P5 | FastAPI + Web-Oberfläche | GUI |
| P6 | Weitere Algorithmen, Normbewertung, Ausbau | — |

Bis einschließlich P4 ist die CLI das Arbeitswerkzeug. Sie zwingt den Core dazu, ohne
UI vollständig zu funktionieren.

## Testen

- **Unit** gegen synthetische Bilder aus `dev/synthetic.py` — bekannte Porenzahl,
  bekannte Fläche, bekannter Maßstab. Die einzige Möglichkeit, absolute Korrektheit zu
  prüfen statt nur Konstanz.
- **Benchmark** gegen `test/Maßstäbe_Bilder/`: die ImageJ-Kalibrierung in den TIFF-Tags
  dient als Ground Truth (`io/metadata.py`, nicht im Produktivpfad). Die Wahrheit steht
  bewusst nicht als Tabelle im Test — sonst prüft der Test irgendwann nur noch die
  Tabelle. Läuft über `pytest -m benchmark`.
- **Integration**: Ordner rein → CSV/JSON/PDF raus.

## Offene Punkte

1. ~~**Python-Version**~~ — geklärt: unter 3.14.6 sind `opencv-python` 5.0, `numpy` 2.5,
   `scipy` 1.18 und `rapidocr-onnxruntime` installierbar und laufen. Erst wenn in P6
   ML-Pakete dazukommen, wird 3.12 wieder ein Thema.
2. **Grenzwerte/Norm** — gibt es eine Vorgabe (max. Porendurchmesser, Porositätsgrenze),
   gegen die `analysis/acceptance.py` bewerten soll?
3. **Betrieb** — Web-UI nur lokal oder auf einem Server für mehrere Nutzer?
4. **Bildgrößen und Mengen** — ab ca. 50 MPix bzw. mehreren hundert Bildern je Lauf
   braucht es Kachelung und Parallelisierung.
5. **Mehrere Proben je Bild?** — betrifft das Datenmodell (`ImageResult` mit einer oder
   mehreren Probenregionen).
