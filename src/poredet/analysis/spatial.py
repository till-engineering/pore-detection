"""Räumliche Auswertung: wie nah stehen die Poren beieinander?

Die Porosität allein sagt über das Bauteil wenig. Zwei Schliffe mit derselben
Porosität verhalten sich völlig verschieden, je nachdem ob die Poren gleichmäßig
verteilt sind oder sich zu Nestern ballen: **tragend ist nicht das Loch, sondern der
Steg dazwischen.** Reißt der Steg, wachsen zwei Poren zu einem Riss zusammen.

**Gemessen wird deshalb von Rand zu Rand, nicht von Mitte zu Mitte.** Der
Schwerpunktabstand ist die verbreitetere, aber für diese Frage falsche Größe: zwei
große Poren mit 200 px Mittenabstand können einen Steg von 20 px haben, zwei kleine
mit demselben Mittenabstand einen von 190 px. Die Zahl, die zählt, ist die zweite.
Der Schwerpunktabstand wird trotzdem mitgeführt - er ist die Grundlage der üblichen
Verteilungsmaße und kostet nichts.

**Wie gerechnet wird.** Exakt über die Konturpunkte, aber nicht über alle Paare: für
jede Pore werden zuerst die nächstgelegenen Poren nach Schwerpunkt gesucht
(``kandidaten``) und nur für diese die Randpunkte gegeneinander vermessen. Der
randnächste Nachbar liegt praktisch immer unter den schwerpunktnächsten; bei stark
unterschiedlich großen Poren genügt ein größeres ``kandidaten``, um das abzusichern.
Alle Paare exakt zu rechnen wäre bei einigen hundert Poren quadratisch und ohne
Erkenntnisgewinn.

Der Abstand ist **symmetrisch**, die Nachbarschaft nicht: die nächste Nachbarin von A
kann B sein, während C näher an B liegt. Deshalb steht je Pore ihr eigener nächster
Nachbar, nicht eine Paarliste.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Nachbarschaft:
    """Die nächste Nachbarin einer Pore und der Steg dazwischen."""

    label: int
    nachbar: int
    #: Kürzester Abstand Rand zu Rand - die Breite des Stegs.
    steg_px: float
    #: Abstand der Flächenschwerpunkte.
    mitten_px: float
    #: Die beiden Randpunkte, zwischen denen gemessen wurde - fürs Einzeichnen.
    von: tuple[float, float]
    nach: tuple[float, float]
    um_per_px: float | None = None

    def _um(self, wert: float) -> float | None:
        return None if self.um_per_px is None else wert * self.um_per_px

    @property
    def steg_um(self) -> float | None:
        return self._um(self.steg_px)

    @property
    def mitten_um(self) -> float | None:
        return self._um(self.mitten_px)


def _konturpunkte(labels: np.ndarray, label: int) -> np.ndarray:
    """Randpunkte eines Objekts als (N, 2)-Feld in (x, y).

    Nur der Rand, nicht die Fläche: der kürzeste Abstand zweier getrennter Gebiete
    wird immer zwischen Randpunkten angenommen, und das sind bei einer runden Pore
    wenige Dutzend Punkte statt einiger tausend Pixel.
    """
    maske = (labels == label).astype(np.uint8)
    umrisse, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not umrisse:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack([u.reshape(-1, 2) for u in umrisse]).astype(np.float64)


def nachbarabstaende(
    labels: np.ndarray,
    um_per_px: float | None = None,
    kandidaten: int = 12,
) -> list[Nachbarschaft]:
    """Für jede Pore die nächste Nachbarin und den Steg dazwischen.

    Bei weniger als zwei Objekten gibt es keine Nachbarschaft - dann ist die Liste
    leer. Das ist kein Fehler, sondern der Normalfall bei einer Einzelpore.
    """
    vorhanden = [int(v) for v in np.unique(labels) if v != 0]
    if len(vorhanden) < 2:
        return []

    raender = {lab: _konturpunkte(labels, lab) for lab in vorhanden}
    vorhanden = [lab for lab in vorhanden if len(raender[lab])]
    if len(vorhanden) < 2:
        return []

    mitten = np.array([raender[lab].mean(axis=0) for lab in vorhanden])

    # Vorauswahl über die Schwerpunkte - nur diese Paare werden exakt vermessen.
    diff = mitten[:, None, :] - mitten[None, :, :]
    mittenabstand = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(mittenabstand, np.inf)
    anzahl = min(kandidaten, len(vorhanden) - 1)

    ergebnis: list[Nachbarschaft] = []
    for i, lab in enumerate(vorhanden):
        naechste = np.argsort(mittenabstand[i])[:anzahl]
        eigene = raender[lab]

        bester = np.inf
        bester_j = -1
        bester_paar: tuple[np.ndarray, np.ndarray] | None = None
        for j in naechste:
            fremde = raender[vorhanden[j]]
            # Volle Abstandsmatrix zwischen zwei Randpunktmengen. Beide sind klein,
            # deshalb ist das billiger als jede Beschleunigungsstruktur.
            d = np.sqrt(((eigene[:, None, :] - fremde[None, :, :]) ** 2).sum(axis=2))
            k = int(np.argmin(d))
            wert = float(d.flat[k])
            if wert < bester:
                bester = wert
                bester_j = int(j)
                a, b = divmod(k, len(fremde))
                bester_paar = (eigene[a], fremde[b])

        if bester_paar is None:
            continue
        von, nach = bester_paar
        ergebnis.append(Nachbarschaft(
            label=lab,
            nachbar=vorhanden[bester_j],
            steg_px=bester,
            mitten_px=float(mittenabstand[i, bester_j]),
            von=(float(von[0]), float(von[1])),
            nach=(float(nach[0]), float(nach[1])),
            um_per_px=um_per_px,
        ))
    return ergebnis


@dataclass(frozen=True)
class Kante:
    """Der Abstand zwischen zwei Poren - eine Kante im Nachbarschaftsgraphen."""

    a: int
    b: int
    #: Kürzester Abstand Rand zu Rand: der Steg.
    steg_px: float
    #: Abstand der Flächenschwerpunkte.
    mitten_px: float
    #: Steg geteilt durch den kleineren der beiden Äquivalentdurchmesser.
    relativ_klein: float
    #: Steg geteilt durch den größeren. Beide, weil die Regelwerke sich hier
    #: unterscheiden und die Entscheidung noch nicht gefallen ist.
    relativ_gross: float
    von: tuple[float, float]
    nach: tuple[float, float]


def _freie_sicht(
    labels: np.ndarray, von: tuple[float, float], nach: tuple[float, float],
    a: int, b: int,
) -> bool:
    """Liegt auf der Verbindungsstrecke eine **dritte** Pore?

    Der Steg ist definiert als das Metall zwischen zwei Hohlräumen. Läuft die
    Verbindung durch eine weitere Pore, ist zwischen den beiden gar kein
    durchgehender Steg - die gemessene Länge beschreibt dann nichts, was tragen
    könnte, und das Paar ist für eine Zusammenfassung ohne Bedeutung: es ist bereits
    über die dazwischenliegende Pore vermittelt.

    Geprüft wird die Strecke zwischen genau den beiden Randpunkten, an denen gemessen
    wurde, mit zwei Stützstellen je Pixel - grober abgetastet ließe sich eine kleine
    Pore überspringen.
    """
    (x1, y1), (x2, y2) = von, nach
    schritte = max(int(np.hypot(x2 - x1, y2 - y1) * 2), 2)
    hoehe, breite = labels.shape
    xs = np.clip(np.round(np.linspace(x1, x2, schritte)).astype(int), 0, breite - 1)
    ys = np.clip(np.round(np.linspace(y1, y2, schritte)).astype(int), 0, hoehe - 1)
    unterwegs = labels[ys, xs]
    return not np.any((unterwegs != 0) & (unterwegs != a) & (unterwegs != b))


def alle_abstaende(
    labels: np.ndarray,
    durchmesser_px: dict[int, float],
    max_faktor: float = 3.0,
    max_px: float | None = None,
    nur_freie_sicht: bool = True,
) -> list[Kante]:
    """Alle Porenpaare mit ihrem Steg - die Grundlage für jedes Zusammenfassungskriterium.

    Im Unterschied zu :func:`nachbarabstaende`, das je Pore **einen** Nachbarn liefert,
    steht hier jedes Paar, das für eine Zusammenfassung überhaupt in Frage kommt. Der
    nächste Nachbar allein genügt dafür nicht: liegen drei Poren dicht beieinander, ist
    für jede nur eine der beiden anderen die nächste, und die dritte Verbindung fehlt.

    **Warum eine Schranke.** Vollständig wären es ``n·(n-1)/2`` Paare - bei 150 Poren
    gut zehntausend, bei 500 schon 125 000, jeweils mit einer exakten Randmessung. Die
    allermeisten davon sind Poren an entgegengesetzten Bildrändern, die kein Kriterium
    je zusammenfassen wird. ``max_faktor`` begrenzt auf Paare, deren Steg kleiner ist
    als das Vielfache des **größeren** Durchmessers; der Standard 3.0 liegt über allen
    gebräuchlichen Regeln und lässt die Wahl des Kriteriums damit offen.

    ``max_px`` setzt zusätzlich eine absolute Obergrenze, falls die Kandidatenmenge
    trotzdem zu groß wird. Ohne Angabe gilt nur der relative Faktor.

    **Verdeckte Paare fallen heraus** (``nur_freie_sicht``). Liegt zwischen zwei Poren
    eine dritte, gibt es zwischen ihnen keinen durchgehenden Steg, und die gemessene
    Länge beschreibt nichts Tragendes - die Nachbarschaft ist bereits über die Pore
    dazwischen vermittelt. Das betrifft vor allem die weiten Paare: bei kurzen Stegen
    passt nichts dazwischen, bei drei Durchmessern Abstand ist an einem dicht
    besetzten Schliff rund ein Fünftel verdeckt.
    """
    vorhanden = [int(v) for v in np.unique(labels) if v != 0]
    if len(vorhanden) < 2:
        return []

    raender = {lab: _konturpunkte(labels, lab) for lab in vorhanden}
    vorhanden = [lab for lab in vorhanden if len(raender[lab])]
    if len(vorhanden) < 2:
        return []
    mitten = {lab: raender[lab].mean(axis=0) for lab in vorhanden}

    kanten: list[Kante] = []
    for i, a in enumerate(vorhanden):
        for b in vorhanden[i + 1:]:
            da = durchmesser_px.get(a, 0.0)
            db = durchmesser_px.get(b, 0.0)
            grenze = max_faktor * max(da, db)
            if max_px is not None:
                grenze = min(grenze, max_px)
            if grenze <= 0:
                continue

            # Grobfilter über die Schwerpunkte. Der Steg kann höchstens um die beiden
            # Radien kleiner sein als der Mittenabstand - liegt der darüber, ist das
            # Paar sicher zu weit auseinander und die teure Randmessung entfällt.
            spanne = float(np.hypot(*(mitten[a] - mitten[b])))
            if spanne - (da + db) / 2.0 > grenze:
                continue

            A, B = raender[a], raender[b]
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
            k = int(np.argmin(d))
            steg = float(d.flat[k])
            if steg > grenze:
                continue

            ia, ib = divmod(k, len(B))
            if nur_freie_sicht and not _freie_sicht(
                labels, (float(A[ia][0]), float(A[ia][1])),
                (float(B[ib][0]), float(B[ib][1])), a, b
            ):
                continue

            klein, gross = min(da, db), max(da, db)
            kanten.append(Kante(
                a=a, b=b, steg_px=steg, mitten_px=spanne,
                relativ_klein=steg / klein if klein > 0 else float("inf"),
                relativ_gross=steg / gross if gross > 0 else float("inf"),
                von=(float(A[ia][0]), float(A[ia][1])),
                nach=(float(B[ib][0]), float(B[ib][1])),
            ))
    return sorted(kanten, key=lambda k: k.steg_px)


def kennwerte(nachbarn: list[Nachbarschaft]) -> dict[str, float | None]:
    """Kurzfassung für Kontrollblatt und Bericht.

    Der **kleinste** Steg im Bild ist die eigentliche Kennzahl: er ist die schwächste
    Stelle, und über die entscheidet sich das Bauteil - nicht über den Mittelwert.
    """
    if not nachbarn:
        return {"n": 0, "min": None, "median": None, "mittel": None}
    werte = sorted(n.steg_px for n in nachbarn)
    mitte = len(werte) // 2
    median = werte[mitte] if len(werte) % 2 else (werte[mitte - 1] + werte[mitte]) / 2
    return {
        "n": len(werte),
        "min": werte[0],
        "median": float(median),
        "mittel": sum(werte) / len(werte),
    }
