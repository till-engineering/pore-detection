"""Kanten finden, Zonen legen, Porosität je Zone - reine Rechnung auf Masken.

Koordinaten: Pixel, Ursprung oben links, x nach rechts, y nach unten; ein Pixel (x, y)
hat seine Mitte bei (x, y). Die Kantenpunkte liegen auf der Pixelgrenze (± 0,5).

Ablauf:

1. **Probenkörper** = größte zusammenhängende Fläche, die weder Einbettmittel noch
   Maßstabskasten ist.
2. **Kantenpunkte:** je Bildspalte der oberste und der unterste Pixel des Körpers - aber
   nur, wenn direkt darüber bzw. darunter Einbettmittel liegt (sonst ist es Bildrand oder
   Maßstab). Die äußeren ``rand_pct`` der Probenbreite fallen weg (Ecken, Flanken).
3. **Geraden:** RANSAC sucht die Gerade, an der die meisten Punkte liegen, danach
   kleinste Quadrate über genau diese Punkte. Ausreißer (Luftblasen, Flankenreste,
   Ausbrüche) ziehen die Gerade so nicht schief.
4. **Dicke T (= 100 %)** = mittlerer senkrechter Abstand der Unterkantenpunkte von der
   Oberkantengeraden.
5. **Zonen:** Grenze p liegt parallel zur Oberkante im senkrechten Abstand p/100 · T.
   Jeder Pixel zählt in der Zone, in der seine Mitte liegt - eine Pore auf einer Grenze
   wird so pixelgenau aufgeteilt. Porosität = Porenpixel / Probenpixel der Zone.
6. **Fehler an der Unterkante:** Abweichung der echten Unterkante von der 100-%-Linie,
   positiv = Probe dort dicker als T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .einstellungen import Einstellungen

_ZEILEN_JE_BLOCK = 512       # Zonenzuordnung blockweise - große Bilder, wenig Speicher
_RANSAC_VERSUCHE = 400


class KeineKante(Exception):
    """Im Bild lässt sich keine Ober- oder Unterkante bestimmen."""


@dataclass(frozen=True)
class Gerade:
    """y = steigung · x + achse"""
    steigung: float
    achse: float

    def y(self, x):
        return self.steigung * x + self.achse

    @property
    def winkel_grad(self) -> float:
        """Winkel zur Bildwaagerechten, positiv = nach rechts abfallend (y nach unten)."""
        return math.degrees(math.atan(self.steigung))


@dataclass
class Kante:
    gerade: Gerade
    x: np.ndarray            # alle Kantenpunkte (nach Abzug der Ecken)
    y: np.ndarray
    genutzt: np.ndarray      # bool - Punkte, die die Gerade tragen

    @property
    def anteil_genutzt(self) -> float:
        return float(self.genutzt.mean()) if len(self.genutzt) else 0.0


@dataclass
class Zone:
    name: str
    von_pct: float | None    # None = offen (vor der ersten / hinter der letzten Grenze)
    bis_pct: float | None
    von_px: float | None     # senkrechter Abstand von der Oberkante
    bis_px: float | None
    probe_px: int
    poren_px: int

    @property
    def porositaet_pct(self) -> float | None:
        return 100.0 * self.poren_px / self.probe_px if self.probe_px else None


@dataclass
class Unterkantenfehler:
    """Wie weit die echte Unterkante von der 100-%-Linie abweicht (px, + = dicker)."""
    winkel_grad: float       # Unterkante gegen Oberkante
    x_links: float
    links_px: float          # Regressionsgerade der Unterkante am linken Ende
    x_rechts: float
    rechts_px: float         # ... am rechten Ende
    x_max: float
    max_px: float            # größte Abweichung eines Kantenpunkts (mit Vorzeichen)
    rms_px: float            # mittlere (quadratische) Abweichung der Kantenpunkte


@dataclass
class ZonenErgebnis:
    bild: str
    breite_px: int
    hoehe_px: int
    um_pro_px: float | None
    grenzen_pct: tuple[float, ...]
    oben: Kante
    unten: Kante
    dicke_px: float
    zonen: list[Zone]
    gesamt: Zone
    fehler: Unterkantenfehler
    x_von: int               # waagerechte Ausdehnung des Probenkörpers
    x_bis: int
    warnungen: list[str] = field(default_factory=list)

    def abstand_px(self, x, y):
        """Senkrechter Abstand von der Oberkante, positiv in Richtung Unterkante."""
        g = self.oben.gerade
        return (y - g.y(x)) / math.hypot(1.0, g.steigung)

    def grenzlinie(self, prozent: float) -> Gerade:
        """Die Gerade im Abstand prozent/100 · Dicke parallel zur Oberkante."""
        g = self.oben.gerade
        versatz = prozent / 100.0 * self.dicke_px * math.hypot(1.0, g.steigung)
        return Gerade(g.steigung, g.achse + versatz)


def berechnen(bild: str, probe: np.ndarray, poren: np.ndarray, einbettmittel: np.ndarray,
              ausgeschlossen: np.ndarray | None, um_pro_px: float | None,
              einst: Einstellungen) -> ZonenErgebnis:
    """Alle Masken bool (H, W). ``probe`` = auswertbare Probenfläche inkl. Poren,
    ``poren`` = gezählte Poren, ``ausgeschlossen`` = Maßstabskasten (oder None)."""
    hoehe, breite = probe.shape
    if ausgeschlossen is None:
        ausgeschlossen = np.zeros_like(probe)
    if not einbettmittel.any():
        raise KeineKante("kein Einbettmittel erkannt - ohne Einbettmittel keine Oberkante")

    koerper = _groesste_flaeche(~einbettmittel & ~ausgeschlossen)
    spalten = np.flatnonzero(koerper.any(axis=0))
    x_von, x_bis = int(spalten[0]), int(spalten[-1])
    rand = einst.rand_pct / 100.0 * (x_bis - x_von)
    innen = (spalten >= x_von + rand) & (spalten <= x_bis - rand)
    xs = spalten[innen]

    oben_y = koerper[:, xs].argmax(axis=0)
    unten_y = hoehe - 1 - koerper[::-1, xs].argmax(axis=0)
    # Kante nur, wo direkt daneben Einbettmittel liegt - nicht Bildrand, nicht Maßstab.
    ok_oben = (oben_y > 0) & einbettmittel[np.maximum(oben_y - 1, 0), xs]
    ok_unten = (unten_y < hoehe - 1) & einbettmittel[np.minimum(unten_y + 1, hoehe - 1), xs]

    toleranz = max(einst.toleranz_min_px, einst.toleranz_pct / 100.0 * (x_bis - x_von))
    rng = np.random.default_rng(0)   # fester Startwert: gleiche Maske, gleiches Ergebnis
    oben = _kante("Oberkante", xs[ok_oben].astype(float), oben_y[ok_oben] - 0.5,
                  toleranz, einst, rng)
    unten = _kante("Unterkante", xs[ok_unten].astype(float), unten_y[ok_unten] + 0.5,
                   toleranz, einst, rng)

    s = math.hypot(1.0, oben.gerade.steigung)
    abstand = (unten.y - oben.gerade.y(unten.x)) / s
    dicke = float(abstand[unten.genutzt].mean())
    if dicke <= 0:
        raise KeineKante("Unterkante liegt nicht unter der Oberkante")

    grenzen_px = np.asarray(einst.grenzen_pct) / 100.0 * dicke
    probe_n, poren_n = _zaehlen(probe, poren & probe, oben.gerade, grenzen_px)
    zonen = _zonen(einst.grenzen_pct, grenzen_px, probe_n, poren_n)
    gesamt = Zone("gesamt", None, None, None, None, int(probe_n.sum()), int(poren_n.sum()))

    fehler = _unterkantenfehler(oben, unten, abstand, dicke)
    warnungen = []
    for kante in (oben, unten):
        if kante.anteil_genutzt < 0.5:
            name = "Oberkante" if kante is oben else "Unterkante"
            warnungen.append(f"{name}: nur {kante.anteil_genutzt:.0%} der Kantenpunkte "
                             "liegen auf der Geraden - Kontrollbild prüfen")
    return ZonenErgebnis(bild, breite, hoehe, um_pro_px, tuple(einst.grenzen_pct), oben,
                         unten, dicke, zonen, gesamt, fehler, x_von, x_bis, warnungen)


def _groesste_flaeche(maske: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(maske.astype(np.uint8), 8)
    if n < 2:
        raise KeineKante("keine Probe neben dem Einbettmittel gefunden")
    groesste = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return labels == groesste


def _kante(name: str, x: np.ndarray, y: np.ndarray, toleranz: float, einst: Einstellungen,
           rng: np.random.Generator) -> Kante:
    if len(x) < einst.min_punkte:
        raise KeineKante(f"{name}: nur {len(x)} Kantenpunkte am Einbettmittel "
                         f"(mindestens {einst.min_punkte}) - grenzt die Probe dort an "
                         "Einbettmittel?")
    max_steigung = math.tan(math.radians(einst.max_winkel_grad))

    # RANSAC: Gerade durch zwei zufällige Punkte, gezählt wird, wer nahe genug liegt.
    i = rng.integers(0, len(x), _RANSAC_VERSUCHE)
    j = rng.integers(0, len(x), _RANSAC_VERSUCHE)
    ok = x[i] != x[j]
    i, j = i[ok], j[ok]
    m = (y[j] - y[i]) / (x[j] - x[i])
    ok = np.abs(m) <= max_steigung
    m, c = m[ok], (y[i] - m * x[i])[ok]
    if not len(m):
        raise KeineKante(f"{name}: keine Gerade flacher als {einst.max_winkel_grad:g}°")
    treffer = np.zeros(len(m), dtype=np.int64)
    for a in range(0, len(m), 50):
        b = slice(a, a + 50)
        abstand = np.abs(y[None, :] - m[b, None] * x[None, :] - c[b, None])
        treffer[b] = (abstand / np.hypot(1.0, m[b])[:, None] < toleranz).sum(axis=1)
    beste = int(treffer.argmax())
    gerade = Gerade(float(m[beste]), float(c[beste]))

    # Verfeinern: kleinste Quadrate über die Punkte an der Geraden, zweimal.
    for _ in range(2):
        genutzt = np.abs(y - gerade.y(x)) / math.hypot(1.0, gerade.steigung) < toleranz
        if genutzt.sum() < 2:
            break
        steigung, achse = np.polyfit(x[genutzt], y[genutzt], 1)
        gerade = Gerade(float(steigung), float(achse))
    genutzt = np.abs(y - gerade.y(x)) / math.hypot(1.0, gerade.steigung) < toleranz
    return Kante(gerade, x, y, genutzt)


def _zaehlen(probe: np.ndarray, poren: np.ndarray, oben: Gerade,
             grenzen_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probe- und Porenpixel je Fach. Fach 0 = vor der ersten Grenze, Fach k = zwischen
    Grenze k-1 und k, letztes Fach = hinter der letzten Grenze."""
    hoehe, breite = probe.shape
    faecher = len(grenzen_px) + 1
    s = math.hypot(1.0, oben.steigung)
    xs = np.arange(breite, dtype=np.float64)
    probe_n = np.zeros(faecher, dtype=np.int64)
    poren_n = np.zeros(faecher, dtype=np.int64)
    for a in range(0, hoehe, _ZEILEN_JE_BLOCK):
        b = min(a + _ZEILEN_JE_BLOCK, hoehe)
        ys = np.arange(a, b, dtype=np.float64)[:, None]
        fach = np.searchsorted(grenzen_px, (ys - oben.y(xs)[None, :]) / s, side="right")
        probe_n += np.bincount(fach[probe[a:b]], minlength=faecher)
        poren_n += np.bincount(fach[poren[a:b]], minlength=faecher)
    return probe_n, poren_n


def _zonen(grenzen_pct, grenzen_px, probe_n, poren_n) -> list[Zone]:
    zonen = []
    if probe_n[0]:
        zonen.append(Zone(f"< {_pct(grenzen_pct[0])} %", None, grenzen_pct[0], None,
                          float(grenzen_px[0]), int(probe_n[0]), int(poren_n[0])))
    for k in range(1, len(grenzen_pct)):
        zonen.append(Zone(str(k), grenzen_pct[k - 1], grenzen_pct[k],
                          float(grenzen_px[k - 1]), float(grenzen_px[k]),
                          int(probe_n[k]), int(poren_n[k])))
    if probe_n[-1]:
        zonen.append(Zone(f"> {_pct(grenzen_pct[-1])} %", grenzen_pct[-1], None,
                          float(grenzen_px[-1]), None, int(probe_n[-1]), int(poren_n[-1])))
    return zonen


def _unterkantenfehler(oben: Kante, unten: Kante, abstand: np.ndarray,
                       dicke: float) -> Unterkantenfehler:
    s = math.hypot(1.0, oben.gerade.steigung)
    x = unten.x[unten.genutzt]
    abweichung = abstand[unten.genutzt] - dicke

    def am_ende(xe: float) -> float:
        return float((unten.gerade.y(xe) - oben.gerade.y(xe)) / s - dicke)

    x_links, x_rechts = float(x.min()), float(x.max())
    k = int(np.abs(abweichung).argmax())
    return Unterkantenfehler(
        winkel_grad=unten.gerade.winkel_grad - oben.gerade.winkel_grad,
        x_links=x_links, links_px=am_ende(x_links),
        x_rechts=x_rechts, rechts_px=am_ende(x_rechts),
        x_max=float(x[k]), max_px=float(abweichung[k]),
        rms_px=float(np.sqrt(np.mean(abweichung ** 2))),
    )


def _pct(wert: float) -> str:
    return f"{wert:g}".replace(".", ",")
