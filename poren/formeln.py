"""Alle Formeln der Kennwerte an einer Stelle.

Wer eine Berechnung ändern will, ändert sie hier - Porenmodell, Messung, Filter,
Verteilung, Tabellen, Viewer und Histogramm holen ihre Werte über diese Funktionen.

Nicht hier stehen die Grundmessungen, die direkt aus der Pixelmaske kommen
(``skimage.measure.regionprops`` in :mod:`poren.messung`): Fläche in Pixeln (Anzahl der
Pixel), Umfang, Haupt- und Nebenachse der angepassten Ellipse, Exzentrizität,
Orientierung, maximaler Feret-Durchmesser, konvexe Fläche und Grauwerte. Alles, was
daraus *berechnet* wird, steht hier.

Längen und Flächen werden in Pixeln gerechnet; ``um_per_px`` rechnet erst am Ende in
µm um (siehe :mod:`poren.messung`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# --------------------------------------------------------------------------------------
# Größe einer Pore
# --------------------------------------------------------------------------------------


def aequivalentdurchmesser(flaeche: float) -> float:
    """Durchmesser des flächengleichen Kreises: d = √(4A/π)."""
    return math.sqrt(4.0 * flaeche / math.pi)


def laenge_um(laenge_px: float, um_per_px: float | None) -> float | None:
    """Länge in µm: L_µm = L_px · (µm/px). ``None`` ohne Maßstab."""
    return None if um_per_px is None else laenge_px * um_per_px


def flaeche_um2(flaeche_px: float, um_per_px: float | None) -> float | None:
    """Fläche in µm²: A_µm² = A_px · (µm/px)². ``None`` ohne Maßstab."""
    return None if um_per_px is None else flaeche_px * um_per_px * um_per_px


# --------------------------------------------------------------------------------------
# Form einer Pore
# --------------------------------------------------------------------------------------


def zirkularitaet(flaeche: float, umfang: float) -> float:
    """Rundheit über den Umfang: 4πA / U² - 1,0 für den Kreis, kleiner für zerklüftete
    Formen. Spalte "rundheit" in poren.csv und "Rundheit" im Viewer."""
    if umfang <= 0:
        return 0.0
    return 4.0 * math.pi * flaeche / umfang ** 2


def rundheit_feret(flaeche: float, feret_max: float) -> float | None:
    """Rundheit über den Feret-Durchmesser: 4A / (π · F_max²) - 1,0 für den Kreis.

    Robust gegen Pixelierung, weil weder Umfang noch Randverlauf eingehen. Genutzt vom
    Filter "roundness" und in verworfen.csv. ``None`` ohne Feret-Durchmesser.
    """
    if feret_max <= 0:
        return None
    return 4.0 * flaeche / (math.pi * feret_max ** 2)


def seitenverhaeltnis(hauptachse: float, nebenachse: float) -> float:
    """Haupt- durch Nebenachse der angepassten Ellipse. Groß bei Kratzern und Riefen."""
    if nebenachse <= 0:
        return math.inf
    return hauptachse / nebenachse


def soliditaet(flaeche: float, konvexe_flaeche: float) -> float:
    """Fläche durch Fläche der konvexen Hülle - 1,0 für konvexe Formen."""
    if konvexe_flaeche <= 0:
        return 0.0
    return flaeche / konvexe_flaeche


# --------------------------------------------------------------------------------------
# Kennzahlen eines Bildes
# --------------------------------------------------------------------------------------


def porositaet_pct(porenflaeche: float, probenflaeche: float) -> float | None:
    """Porosität in %: 100 · Σ A_Pore / A_Probe.

    Bezogen auf die **Probenfläche**, nicht auf das ganze Bild - sonst hinge die Zahl vom
    Anteil des Einbettmittels im Bildausschnitt ab.
    """
    if probenflaeche <= 0:
        return None
    return 100.0 * porenflaeche / probenflaeche


def porendichte(porenzahl: int, probenflaeche_mm2: float | None) -> float | None:
    """Poren je mm² Probenfläche: n / A_Probe."""
    if not probenflaeche_mm2:
        return None
    return porenzahl / probenflaeche_mm2


# --------------------------------------------------------------------------------------
# Statistik über viele Poren
# --------------------------------------------------------------------------------------


def mittelwert(werte: Sequence[float]) -> float | None:
    """Arithmetisches Mittel: Σx / n."""
    return sum(werte) / len(werte) if werte else None


def quantil(werte: Sequence[float], anteil: float) -> float | None:
    """Quantil mit linearer Interpolation zwischen den Rangplätzen - wie
    ``numpy.percentile``. D10, D50, D90 sind ``anteil`` = 0,1 / 0,5 / 0,9."""
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
