"""Größenverteilungen: Klassen und Quantile.

Porengrößen sind **rechtsschief**. Das ist keine Eigenheit dieser Bilder, sondern die
Regel: wenige große Hohlräume, sehr viele kleine. Zwei Festlegungen folgen daraus.

**Logarithmische Klassen.** Lineare Klassen über eine rechtsschiefe Verteilung legen
alles in die erste Klasse und lassen den Rest leer - die Form, auf die es ankommt, ist
dann nicht zu sehen. Über den Logarithmus wird aus dem Abfall eine lesbare Kurve, und
gleich breite Klassen entsprechen gleichen *Faktoren* statt gleichen Differenzen. Das
ist auch die Konvention der Partikelmesstechnik.

**Quantile statt Mittelwert.** Ein einzelner Lunker zieht den Mittelwert beliebig weit;
D50 und D90 tun das nicht. D90 heißt: 90 % der Poren sind kleiner als dieser Wert - die
Aussage, die in der Abnahme gebraucht wird.

Die Klassengrenzen entstehen **einmal für alle Gruppen** (:func:`kanten`), damit
gezählte und verworfene Poren im selben Diagramm übereinander liegen können. Getrennt
berechnete Kanten wären nicht vergleichbar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Verteilung:
    """Eine Häufigkeitsverteilung über feste Klassengrenzen."""

    #: ``len(anzahlen) + 1`` Grenzen, aufsteigend.
    kanten: tuple[float, ...]
    #: Besetzung je Klasse.
    anzahlen: tuple[int, ...]
    #: Ob die Klassen logarithmisch geteilt sind - die Anzeige braucht das für die Achse.
    logarithmisch: bool

    @property
    def summe(self) -> int:
        return sum(self.anzahlen)

    @property
    def maximum(self) -> int:
        """Höchste Besetzung - die Obergrenze der Werteachse."""
        return max(self.anzahlen, default=0)

    def mitten(self) -> tuple[float, ...]:
        """Klassenmitten. Bei log-Klassen das geometrische Mittel, nicht das arithmetische."""
        paare = zip(self.kanten[:-1], self.kanten[1:], strict=True)
        if self.logarithmisch:
            return tuple(math.sqrt(a * b) for a, b in paare)
        return tuple((a + b) / 2.0 for a, b in paare)


def klassenzahl(werte: Sequence[float], logarithmisch: bool = True) -> int:
    """Wie viele Klassen die Daten hergeben - nach Freedman-Diaconis.

    Eine feste Klassenzahl ist immer für einen Teil der Bilder falsch: bei wenigen
    Poren wird das Histogramm zum Kamm aus Einzelfällen, bei vielen verschenkt es
    Auflösung. Die Regel bindet die Klassenbreite an ``2·IQR·n^(-1/3)`` und damit an
    das, was die Daten tatsächlich tragen.

    **Warum Freedman-Diaconis und nicht Sturges.** Sturges leitet sich aus der
    Normalverteilung her und unterschätzt die nötige Klassenzahl bei allem, was
    schief ist - und Porengrößen sind es immer. Der Interquartilsabstand ist zudem
    unempfindlich gegen den einen großen Lunker, der die Spannweite dominiert.

    Bei logarithmischen Klassen wird auf den Logarithmen gerechnet: dort sind die
    Klassen gleich breit, und nur dort ist die Regel anwendbar.
    """
    brauchbar = [float(w) for w in werte if w is not None and (w > 0 or not logarithmisch)]
    if len(brauchbar) < 4:
        return max(1, len(brauchbar))

    daten = sorted(math.log10(w) for w in brauchbar) if logarithmisch else sorted(brauchbar)
    spanne = daten[-1] - daten[0]
    if spanne <= 0:
        return 1

    q1 = quantil(daten, 0.25)
    q3 = quantil(daten, 0.75)
    iqr = (q3 or 0.0) - (q1 or 0.0)
    if iqr <= 0:
        # Entartet: mehr als die Haelfte der Werte identisch. Dann traegt die
        # Wurzelregel noch am ehesten.
        return max(4, min(20, round(math.sqrt(len(daten)))))

    breite = 2.0 * iqr * len(daten) ** (-1 / 3)
    return max(5, min(24, math.ceil(spanne / breite)))


def dichtekurve(
    werte: Sequence[float],
    grenzen: tuple[float, ...],
    stuetzstellen: int = 96,
) -> tuple[tuple[float, float], ...]:
    """Kerndichteschätzung, skaliert auf die Anzahlachse des Histogramms.

    Das Histogramm zeigt die Verteilung immer durch die Brille seiner Klassengrenzen;
    verschiebt man sie um eine halbe Klasse, sieht die Form anders aus. Die Kurve
    kommt ohne Klassen aus und macht sichtbar, was davon Struktur ist und was
    Einteilung.

    Gerechnet wird im **Logarithmus**, weil dort auch die Klassen liegen - eine im
    linearen Raum geglättete Kurve würde die kleinen Poren verschmieren. Die
    Bandbreite folgt Silverman; multipliziert mit ``n`` und der Klassenbreite hat die
    Dichte dieselbe Einheit wie die Balkenhöhen und darf deshalb auf dieselbe Achse -
    eine zweite Skala wäre hier der klassische Diagrammfehler.

    Ergebnis sind Punkte ``(x, hoehe)`` mit ``x`` im ursprünglichen Wertebereich.
    """
    daten = sorted(math.log10(float(w)) for w in werte if w is not None and w > 0)
    n = len(daten)
    if n < 5 or len(grenzen) < 2:
        return ()

    streuung = math.sqrt(sum((x - sum(daten) / n) ** 2 for x in daten) / (n - 1))
    q1, q3 = quantil(daten, 0.25) or 0.0, quantil(daten, 0.75) or 0.0
    robust = (q3 - q1) / 1.349
    sigma = min(streuung, robust) if robust > 0 else streuung
    if sigma <= 0:
        return ()
    h = 0.9 * sigma * n ** (-1 / 5)

    lo, hi = math.log10(grenzen[0]), math.log10(grenzen[-1])
    klassenbreite = (hi - lo) / (len(grenzen) - 1)
    # Die Dichte traegt 1/(n·h·√(2π)); auf die Anzahlachse gebracht wird sie mit
    # n·Klassenbreite, das n kuerzt sich dabei heraus.
    faktor = klassenbreite / (h * math.sqrt(2 * math.pi))

    punkte: list[tuple[float, float]] = []
    for i in range(stuetzstellen):
        x = lo + (hi - lo) * i / (stuetzstellen - 1)
        summe = 0.0
        for wert in daten:
            u = (x - wert) / h
            if abs(u) < 5.0:          # jenseits davon ist der Beitrag bedeutungslos
                summe += math.exp(-0.5 * u * u)
        punkte.append((10 ** x, faktor * summe))
    return tuple(punkte)


def kanten(
    werte: Sequence[float], klassen: int = 14, logarithmisch: bool = True
) -> tuple[float, ...]:
    """Klassengrenzen über **alle** Werte, die später verglichen werden sollen.

    Bei logarithmischer Teilung werden die Grenzen auf glatte Faktoren gelegt, damit die
    Achse runde Beschriftungen bekommt statt krummer Messwerte.
    """
    brauchbar = [float(w) for w in werte if w is not None and w > 0]
    if not brauchbar:
        return (0.0, 1.0)

    unten, oben = min(brauchbar), max(brauchbar)
    if oben <= unten:
        oben = unten * 1.05 if logarithmisch else unten + 1.0

    if not logarithmisch:
        schritt = (oben - unten) / klassen
        return tuple(unten + i * schritt for i in range(klassen + 1))

    # Auf die nächste glatte Dekadenstufe aufziehen - so fallen die Achsenticks auf
    # runde Werte und nicht auf den zufaelligen kleinsten Messwert.
    lo = math.floor(math.log10(unten) * 4) / 4
    hi = math.ceil(math.log10(oben) * 4) / 4
    schritt = (hi - lo) / klassen
    return tuple(10 ** (lo + i * schritt) for i in range(klassen + 1))


def verteilen(
    werte: Sequence[float], grenzen: tuple[float, ...], logarithmisch: bool = True
) -> Verteilung:
    """Werte in vorgegebene Klassen einsortieren.

    Werte oberhalb der letzten Grenze fallen in die letzte Klasse statt heraus: die
    Grenzen stammen aus denselben Daten, ein Ausreißer darüber ist Rundung.
    """
    anzahlen = [0] * (len(grenzen) - 1)
    for wert in werte:
        if wert is None or wert <= 0:
            continue
        for i in range(len(anzahlen)):
            if wert < grenzen[i + 1] or i == len(anzahlen) - 1:
                anzahlen[i] += 1
                break
    return Verteilung(tuple(grenzen), tuple(anzahlen), logarithmisch)


def quantil(werte: Sequence[float], anteil: float) -> float | None:
    """Lineare Interpolation zwischen den Rangplätzen - wie ``numpy.percentile``.

    Hier ohne numpy, weil das Modul sonst nichts davon braucht und die Verteilung auch
    aus dem Bericht heraus aufgerufen wird.
    """
    sortiert = sorted(float(w) for w in werte if w is not None)
    if not sortiert:
        return None
    if len(sortiert) == 1:
        return sortiert[0]

    platz = anteil * (len(sortiert) - 1)
    unten = math.floor(platz)
    oben = min(unten + 1, len(sortiert) - 1)
    rest = platz - unten
    return sortiert[unten] * (1 - rest) + sortiert[oben] * rest


def kennwerte(werte: Sequence[float]) -> dict[str, float | None]:
    """D10, D50, D90 und Spannweite - die Zahlen, die neben dem Diagramm stehen."""
    brauchbar = [float(w) for w in werte if w is not None]
    if not brauchbar:
        return {"n": 0, "d10": None, "d50": None, "d90": None, "min": None, "max": None}
    return {
        "n": len(brauchbar),
        "d10": quantil(brauchbar, 0.10),
        "d50": quantil(brauchbar, 0.50),
        "d90": quantil(brauchbar, 0.90),
        "min": min(brauchbar),
        "max": max(brauchbar),
    }
