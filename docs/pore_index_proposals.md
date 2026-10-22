# Porenindex – Verbesserungsvorschläge

> Stand 2026-10-22 · Bezug: `pore_detection/evaluation/pore_index.py`, PLAN.md 7.6 (Porenkonzept).
> Nur Vorschläge, nichts davon ist umgesetzt. Neue Varianten kommen als **neue Module** neben
> `pore_index` – das bestehende Modul bleibt unverändert.
> Formeln und Faktoren nach Murakami sind aus dem Gedächtnis wiedergegeben und **vor einer Umsetzung
> an der Literatur zu prüfen** (Murakami, *Metal Fatigue: Effects of Small Defects and Nonmetallic
> Inclusions*; ASTM E2283).

---

## 1. Das heutige Modell in einem Satz

```
Φ_q     = Σ A_i g_i^q / A_Mess
Φ_eff,q = w_o Σ_o A_i g_i^q / A_Mess,o + w_u Σ_u A_i g_i^q / A_Mess,u,   w = (n + 1) / (n_o + n_u + 2)
```

Jede Porenfläche wird mit der relativen Spannung g (0..1) an ihrem Ort gewichtet, q betont die hoch
beanspruchten Bereiche. Das Ergebnis ist eine **spannungsgewichtete Porosität** (Flächenanteil).

---

## 2. Problem: Φ ist immer extrem klein

### Warum

Zwei Effekte verkleinern den Wert:

1. **Porenfläche / Probenfläche** ist schon die Porosität – typisch 0,1 … 2 %, also 10⁻³ … 10⁻².
2. **g^q mit g < 1** verkleinert weiter, und zwar umso stärker, je größer q. Der Nenner A_Mess wird
   dabei *nicht* gewichtet, nur der Zähler.

Rechnung für gleichmäßig verteilte Poren (Porosität P) bei `bending` (g = |2ζ − 1|) und ebenso bei
`pulsating` (g = ζ): der Mittelwert von g^q über die Höhe ist 1 / (q + 1), also

```
Φ_q = P / (q + 1)
```

| q | Φ_q bei P = 0,5 % |
|---|---|
| 1 | 0,25 % = 2,5·10⁻³ |
| 3 | 0,125 % = 1,25·10⁻³ |
| 10 | 0,045 % = 4,5·10⁻⁴ |

Der Wert hängt also schon bei *gleicher* Probe stark von q ab, ist kaum lesbar und zwischen
verschiedenen q nicht vergleichbar. Ob Poren „gefährlich liegen“, sieht man daran nicht direkt.

### Vorschläge (beliebig kombinierbar, als neue Ausgabespalten)

**2a. Gewichteter Nenner – empfohlen.** Auch die Probenfläche mit g^q gewichten:

```
Φ*_q = Σ A_i g_i^q / ∫_Probe g^q dA
```

- Gleichmäßig verteilte Poren → Φ*_q = P **für jedes q** (gleiche Größenordnung wie die Porosität).
- Poren vor allem an hoch beanspruchten Stellen → Φ*_q > P; vor allem in der neutralen Faser → < P.
- Der Nenner ist die „effektiv beanspruchte Fläche“ (wie die effektive Fläche / das effektive Volumen
  in Weibull-Modellen) und lässt sich pixelweise genauso zählen wie heute A_Mess,o/u.

**2b. Lagefaktor (dimensionslos, um 1).**

```
L_q = Φ*_q / P
```

1 = Poren liegen „durchschnittlich“, 2 = doppelt so gefährlich wie gleichmäßig verteilt. Trennt die
Frage *wie viel* Pore (P) von *wo* (L_q) – beide Werte sind einzeln gut lesbar.

**2c. Mittlere Spannung der Porenfläche (0..1).**

```
ḡ_q = (Σ A_i g_i^q / Σ A_i)^(1/q)
```

Potenzmittel der Spannung über die Porenfläche: „im Mittel liegen die Poren bei 70 % der
Randspannung“. Unabhängig von der Porosität, für q → ∞ die höchste Spannung, an der eine Pore liegt.

**2d. Einheit.** Mindestens als ppm oder ‰ ausgeben statt als Bruch – ändert nichts am Modell, nur an
der Lesbarkeit.

---

## 3. Grundsätzliche Schwächen für Ermüdung

| # | Punkt | Warum es stört |
|---|---|---|
| 1 | **Fläche addiert sich** | 100 kleine Poren = 1 große Pore gleicher Gesamtfläche. Für Ermüdung entscheidet aber meist die *größte* Pore an hoch beanspruchter Stelle (Weakest Link) |
| 2 | **q nur auf der Spannung** | Rissgefahr hängt von Spannung *und* Defektgröße ab, K ≈ Y·σ·√(π·√area). Die Größe geht nur linear über die Fläche ein |
| 3 | **Fläche in der Schliffebene** | Maßgeblich ist die Ausdehnung **senkrecht zur Hauptspannung** (bei Biegung: in z). Eine in x gestreckte, flache Pore ist harmloser als eine gleich große, die in z steht |
| 4 | **Abstand zur Oberfläche fehlt** | Oberflächennahe / angeschnittene Poren sind deutlich kritischer als innere; nur indirekt über g(ζ) erfasst |
| 5 | **Härte ohne Wirkung** | Wird eingegeben, geht aber nicht ein. Über Murakami bestimmt sie direkt die ertragbare Spannung (Abschnitt 4A) |
| 6 | **Teilung an der Zonengrenze** | Eine Pore über ζ = 0,5 wird pixelweise in zwei Poren geteilt. Physikalisch ein Defekt – für größenabhängige Modelle verfälscht das die Größe |
| 7 | **Gewichte aus Versuchen** | w aus den Anrissorten n_o, n_u steckt das Versuchsergebnis in den Index. Zur *Vorhersage* von Anrissort / Lebensdauer zirkulär; als Kalibrierung besser mit einem Fit (4D) |
| 8 | **q frei gewählt** | 1 … 10 sind Annahmen. q entspricht dem Weibull-Modul eines Weakest-Link-Modells und ließe sich aus Versuchen bestimmen |
| 9 | **Spannungsprofile stark vereinfacht** | |2ζ − 1| setzt einen geraden, symmetrischen Querschnitt voraus. Bei gekrümmten Bauteilen (Ring) wandert die neutrale Faser nach innen; Eigenspannungen (Randschicht) und Kerben fehlen |
| 10 | **Ein Schliff ist eine Stichprobe** | Ein 2D-Schnitt trifft Poren zufällig und unterschätzt ihre Größe. Ob die gefährlichste Pore des Bauteils im Schliff liegt, ist Zufall |
| 11 | **Nachbarporen** | Liegen zwei Poren näher als die kleinere groß ist, wirken sie wie eine große (Murakami: umhüllende Fläche) |
| 12 | **Normierung auf die Messfläche** | Ein Dichtewert: gut zum Vergleich von Schliffen, sagt aber nichts über den Größeneinfluss (größeres Bauteil → mehr Defekte → wahrscheinlicheres Versagen) |

---

## 4. Vorschläge für neue Module

### A. `murakami` – Auslastung jeder Pore (empfohlen als erster Schritt)

Pro Pore:

- **Größe** √area senkrecht zur Last. Aus dem Schliff: Ausdehnung h_z der Pore in z (Höhe im
  Proben-KOS), daraus z. B. √area ≈ √(π/4 · h_z²) (Annahme: im unbekannten dritten Maß so tief wie
  hoch) – oder vorerst einfach √A als Näherung.
- **Lage**: Oberfläche (angeschnitten), oberflächennah (Abstand kleiner als ihre Größe → wie
  Oberflächendefekt) oder innen.
- **Härte an ihrer Höhe**: aus `hardness_outer` / `hardness_inner` über ζ interpoliert
  (HBW → HV umrechnen; für Stahl grob HV ≈ 1,05 HBW – prüfen).
- **Ertragbare Spannung** (Murakami, σ_w in MPa, HV in kgf/mm², √area in µm):

  ```
  σ_w = C (HV + 120) / (√area)^(1/6),   C = 1,43 Oberfläche · 1,41 oberflächennah · 1,56 innen
  ```
  Mittelspannung über den Faktor ((1 − R)/2)^α, α = 0,226 + HV·10⁻⁴.

- **Auslastung** η_i = σ(ζ_i) / σ_w,i – mit σ(ζ) = σ_nenn · g(ζ). Ohne Nennspannung bleibt η relativ
  (Verhältnis zwischen Poren und Schliffen ist trotzdem gültig).

Ausgabe pro Bild: höchste Auslastung η_max, die kritische Pore (id, Lage, √area), vorhergesagte
Anrissseite (außen / innen = Seite der Pore mit η_max). Pro Pore: √area, Lage, σ_w, η.
Beantwortet die offene Frage aus PLAN.md 7.6 („g = σ/σ_max oder σ/σ_W?“): **σ/σ_W**.

### B. `extreme_value` – größte Pore im Bauteil (über die ID)

- Je Schliff die größte √area (oder die größte Auslastung) → Gumbel-Verteilung über alle Schliffe
  der ID (ASTM E2283).
- Hochrechnung auf die Fläche / das Volumen des Bauteils (Wiederkehrperiode) → größte zu erwartende
  Pore → mit A: Dauerfestigkeit des Bauteils.
- Löst Punkt 10 (Stichprobe) und 12 (Größeneinfluss). Braucht mehrere Schliffe pro ID – passt zur
  offenen ID-Auswertung in PLAN.md 7.6.

### C. `weakest_link` – Versagenswahrscheinlichkeit

```
P_f = 1 − exp( − Σ_i (η_i / η_0)^m )
```

Verallgemeinert Φ_q: statt Fläche × g^q geht die **Auslastung jeder Pore** (Spannung *und* Größe *und*
Härte) mit dem Weibull-Modul m ein. m und η_0 aus Versuchen (D). Ohne Kalibrierung ist
Σ η_i^m als relativer Index nutzbar – wie heute Φ_q, aber physikalisch begründet.

### D. Kalibrierung aus den Versuchen (Werkzeug, kein Pipeline-Modul)

- Parameter (m bzw. q, Größenexponent, Zonengewichte) so fitten, dass der Index die Versuchsergebnisse
  am besten erklärt: Anrissseite (logistische Regression / Maximum Likelihood), wenn vorhanden auch
  Lebensdauer.
- Ersetzt die festen Gewichte w = (n + 1)/(n_o + n_u + 2) und die frei gewählten q (Punkte 7, 8).
- Mit Prüfung auf zurückgehaltenen Proben (wie `dataset_eval/tune.py`), sonst passt das Modell nur auf
  die Proben, an denen es gefittet wurde.

### E. Ergänzungen

- **Porencluster** zusammenfassen, bevor bewertet wird (Punkt 11): Abstand < Größe der kleineren Pore
  → umhüllende Fläche. Nur für die Bewertung, die erkannten Poren bleiben unverändert.
- **Ganze Poren statt Teilung** an der Zonengrenze (Punkt 6): die Pore zählt in der Zone ihres
  kritischsten Punkts.
- **Spannungsfeld frei vorgeben** (Punkt 9): gekrümmter Balken (neutrale Faser verschoben),
  Eigenspannungsprofil, oder σ(ζ) als Tabelle aus FE. War in PLAN.md 7.6 schon als offen vermerkt.

---

## 5. Vorgeschlagene Reihenfolge

1. **2a + 2b** (gewichteter Nenner, Lagefaktor) – klein, löst das „immer extrem klein“, ändert die
   Idee des Modells nicht. Als neue Spalten eines neuen Moduls oder einer Variante von `pore_index`.
2. **A `murakami`** – nutzt die schon eingegebene Härte, anerkanntes Modell, keine neuen Daten nötig.
3. **E** (Cluster, ganze Poren) – macht A belastbarer.
4. **B, C, D** – sobald mehrere Schliffe pro ID und Versuchsergebnisse vorliegen.

## 6. Offene Fragen

- **Wofür ist der Index gedacht?** Nur Schliffe vergleichen (gut / schlecht), Anrissseite
  vorhersagen, oder Dauerfestigkeit / Lebensdauer abschätzen? Bestimmt, welche Vorschläge nötig sind.
- Gibt es eine Nennspannung (oder Lastamplitude) pro Versuch? Dann wird η absolut statt relativ.
- Liegen Lebensdauern vor oder nur die Anrissseite?
- Welcher Werkstoff / welche Wärmebehandlung (Gültigkeit der Murakami-Konstanten, HBW → HV)?
- Wie viele Schliffe pro Bauteil sind realistisch (Extremwertstatistik braucht etwa 6 und mehr)?
