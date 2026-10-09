"""Synthetische Schliffbilder mit bekannter Wahrheit.

Ein Bild besteht aus heller Probe mit leichtem Rauschen und Helligkeitsverlauf, dunklem,
glattem Einbettmittel hinter einem gewellten Rand und dunklen Formen als Poren. Die Formen
haben harte Kanten - so ist jedes Pixel eindeutig Pore oder nicht, und die Wahrheit ist
pixelgenau.

Drei Rand-Bauformen: Bänder oben und unten, eine Seite, eine Insel in der Mitte. Der Rand
ist eine Summe von Sinuswellen mit kleinem, schnellem Zittern darauf - nicht trivial, aber
reproduzierbar über den Zufallswert (``seed``).

Formen je Bild (``kategorie``):

* ``sauber``   Kreis, Ellipse, gedrehtes Rechteck, unregelmäßiges Polygon - frei in der
               Probe, mit Abstand zu allem. Erwartung: gezählt, Größe stimmt.
* ``rand``     Kreis oder Ellipse auf dem Probenrand. Wahrheit ist nur der Teil in der
               Probe. Erwartung offen - hängt an Probenmaske und Filtern.
* ``paar``     zwei sich berührende Kreise. Erwartung: zwei Poren (Trennen ist im
               Standard an).
* ``winzig``   kleiner als ``min_diameter`` (4 px). Erwartung: nicht gezählt.
* ``kratzer``  dünne lange Linie. Erwartung: nicht gezählt (Filter ``scratch``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .kennwerte import kennwerte

#: Grauwerte der Bildbestandteile (Mittel, Streuung des Rauschens).
PROBE_GRAU, PROBE_RAUSCHEN = 205.0, 6.0
HARZ_GRAU, HARZ_RAUSCHEN = 58.0, 2.5
PORE_GRAU, PORE_RAUSCHEN = 22.0, 3.0
KRATZER_GRAU = 45.0
#: Mindestabstand zwischen zwei Formen in px - deutlich über dem Spalt, den das
#: Zusammenführen überbrückt (``merge.max_gap_px``).
ABSTAND_PX = 10


def erzeugen(anzahl: int, seed: int, bilder: Path, wahrheit: Path) -> list[Path]:
    """``anzahl`` Bilder nach ``bilder``, ihre Wahrheit nach ``wahrheit``."""
    bilder.mkdir(parents=True, exist_ok=True)
    wahrheit.mkdir(parents=True, exist_ok=True)
    pfade = []
    for nummer in range(1, anzahl + 1):
        rng = np.random.default_rng([seed, nummer])
        stem = f"synth_{nummer:03d}"
        bild, probe, formen_bild, formen = _ein_bild(rng)
        pfad = bilder / f"{stem}.png"
        _png(pfad, bild)
        _png(wahrheit / f"{stem}_probe.png", np.where(probe, 255, 0).astype(np.uint8))
        _png(wahrheit / f"{stem}_formen.png", formen_bild.astype(np.uint16))
        (wahrheit / f"{stem}.json").write_text(json.dumps({
            "bild": pfad.name, "seed": seed, "nummer": nummer,
            "breite_px": int(bild.shape[1]), "hoehe_px": int(bild.shape[0]),
            "probenflaeche_px": int(probe.sum()),
            "formen": formen,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        pfade.append(pfad)
    return pfade


# --------------------------------------------------------------------------------------
# Ein Bild
# --------------------------------------------------------------------------------------


def _ein_bild(rng: np.random.Generator):
    breite = int(rng.integers(1000, 1401))
    hoehe = int(rng.integers(750, 1001))
    bauform = str(rng.choice(["baender", "seite", "insel"]))
    probe = {"baender": _baender, "seite": _seite, "insel": _insel}[bauform](rng, breite, hoehe)

    platz = _Platz(probe)
    formen: list[dict] = []
    formen_bild = np.zeros((hoehe, breite), dtype=np.int32)   # Wahrheit: Form-ID je Pixel
    gemalt = np.zeros((hoehe, breite), dtype=bool)            # dunkel gemalt (auch außerhalb)
    kratzer = np.zeros((hoehe, breite), dtype=bool)

    def aufnehmen(maske: np.ndarray, **eintrag) -> None:
        nummer = len(formen) + 1
        wahr = maske & probe
        formen_bild[wahr] = nummer
        formen.append({"id": nummer, **eintrag, "wahr": kennwerte(wahr)})

    for _ in range(int(rng.integers(14, 25))):
        form = _saubere_form(rng, platz)
        if form is not None:
            maske, eintrag = form
            gemalt |= maske
            aufnehmen(maske, kategorie="sauber", erwartung="gezaehlt", **eintrag)

    for _ in range(2):
        form = _randform(rng, platz)
        if form is not None:
            maske, eintrag = form
            gemalt |= maske
            aufnehmen(maske, kategorie="rand", erwartung="offen", **eintrag)

    paar = _paar(rng, platz)
    if paar is not None:
        erste = len(formen) + 1
        for i, (maske, eintrag) in enumerate(paar):
            gemalt |= maske
            aufnehmen(maske, kategorie="paar", erwartung="gezaehlt",
                      partner=erste + 1 - i, **eintrag)

    for _ in range(2):
        form = _winzig(rng, platz)
        if form is not None:
            maske, eintrag = form
            gemalt |= maske
            aufnehmen(maske, kategorie="winzig", erwartung="nicht_gezaehlt", **eintrag)

    form = _kratzer(rng, platz)
    if form is not None:
        maske, eintrag = form
        kratzer |= maske
        aufnehmen(maske, kategorie="kratzer", erwartung="nicht_gezaehlt", **eintrag)

    bild = _malen(rng, probe, gemalt, kratzer)
    for f in formen:
        f["bauform"] = bauform
    return bild, probe, formen_bild, formen


def _malen(rng, probe, gemalt, kratzer) -> np.ndarray:
    hoehe, breite = probe.shape
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    # Leichter Helligkeitsverlauf über die Probe, wie bei ungleichmäßiger Beleuchtung.
    gx, gy = rng.uniform(-12, 12, size=2)
    verlauf = gx * (xx / breite - 0.5) + gy * (yy / hoehe - 0.5)

    def rauschen(sigma: float, glatt: float) -> np.ndarray:
        """Rauschen mit Streuung ``sigma``; geglättet ergibt es eine feine Textur. Das
        Glätten senkt die Streuung - danach wird sie wieder auf ``sigma`` gebracht."""
        r = rng.normal(0.0, 1.0, size=probe.shape).astype(np.float32)
        if glatt > 0:
            r = cv2.GaussianBlur(r, (0, 0), glatt)
        return r * (sigma / max(1e-6, float(r.std())))

    bild = np.full(probe.shape, HARZ_GRAU, dtype=np.float32) + rauschen(HARZ_RAUSCHEN, 1.5)
    probenwert = PROBE_GRAU + verlauf + rauschen(PROBE_RAUSCHEN, 0.8)
    bild = np.where(probe, probenwert, bild)
    bild = np.where(kratzer & probe, KRATZER_GRAU + rauschen(2.0, 0), bild)
    bild = np.where(gemalt, PORE_GRAU + rauschen(PORE_RAUSCHEN, 0), bild)
    return np.clip(np.round(bild), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Gewellter Rand
# --------------------------------------------------------------------------------------


def _welle(rng, n: int, amplitude: float) -> np.ndarray:
    """Summe dreier Sinuswellen plus schnelles Zittern - der gewellte Rand."""
    x = np.arange(n, dtype=np.float64)
    welle = np.zeros(n)
    for k in range(3):
        a = amplitude * rng.uniform(0.4, 1.0) / (k + 1)
        f = rng.uniform(0.6, 3.0) * (k + 1)
        welle += a * np.sin(2 * math.pi * f * x / n + rng.uniform(0, 2 * math.pi))
    welle += rng.uniform(0.8, 2.0) * np.sin(2 * math.pi * rng.uniform(25, 50) * x / n
                                            + rng.uniform(0, 2 * math.pi))
    return welle


def _baender(rng, breite, hoehe) -> np.ndarray:
    oben = hoehe * rng.uniform(0.12, 0.2) + _welle(rng, breite, hoehe * 0.04)
    unten = hoehe * rng.uniform(0.8, 0.88) + _welle(rng, breite, hoehe * 0.04)
    yy = np.arange(hoehe)[:, None]
    return (yy > oben[None, :]) & (yy < unten[None, :])


def _seite(rng, breite, hoehe) -> np.ndarray:
    seite = str(rng.choice(["links", "rechts", "oben", "unten"]))
    laengs = breite if seite in ("oben", "unten") else hoehe
    quer = hoehe if seite in ("oben", "unten") else breite
    tiefe = quer * rng.uniform(0.18, 0.3) + _welle(rng, laengs, quer * 0.05)
    if seite in ("oben", "unten"):
        y = np.arange(hoehe)[:, None]
        harz = y < tiefe[None, :] if seite == "oben" else y > hoehe - 1 - tiefe[None, :]
    else:
        x = np.arange(breite)[None, :]
        harz = x < tiefe[:, None] if seite == "links" else x > breite - 1 - tiefe[:, None]
    return ~harz


def _insel(rng, breite, hoehe) -> np.ndarray:
    cx = breite / 2 + rng.uniform(-0.05, 0.05) * breite
    cy = hoehe / 2 + rng.uniform(-0.05, 0.05) * hoehe
    r0 = 0.42 * min(breite, hoehe)
    yy, xx = np.mgrid[0:hoehe, 0:breite]
    winkel = np.arctan2(yy - cy, xx - cx)
    radius = np.ones_like(winkel)
    for k in rng.choice(np.arange(2, 8), size=3, replace=False):
        radius += rng.uniform(0.03, 0.07) * np.sin(k * winkel + rng.uniform(0, 2 * math.pi))
    radius += 0.004 * np.sin(int(rng.integers(20, 41)) * winkel + rng.uniform(0, 2 * math.pi))
    return np.hypot(xx - cx, yy - cy) < r0 * radius


# --------------------------------------------------------------------------------------
# Formen
# --------------------------------------------------------------------------------------


class _Platz:
    """Wo eine Form hinpasst: weit genug vom Probenrand, vom Bildrand und von den
    anderen Formen."""

    def __init__(self, probe: np.ndarray) -> None:
        self.probe = probe
        self.abstand = cv2.distanceTransform(probe.astype(np.uint8), cv2.DIST_L2, 5)
        self.belegt = np.zeros(probe.shape, dtype=bool)
        rand = probe & ~cv2.erode(probe.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.randpunkte = np.argwhere(rand)

    def innen(self, rng, radius: float) -> tuple[int, int] | None:
        """Ein Mittelpunkt, um den ``radius`` frei in der Probe liegt."""
        hoehe, breite = self.probe.shape
        rand = radius + ABSTAND_PX
        ok = self.abstand >= rand
        ys, xs = np.nonzero(ok)
        if len(xs) == 0:
            return None
        for _ in range(60):
            i = int(rng.integers(len(xs)))
            x, y = int(xs[i]), int(ys[i])
            if rand <= x < breite - rand and rand <= y < hoehe - rand:
                return x, y
        return None

    def auf_rand(self, rng, radius: float) -> tuple[int, int] | None:
        """Ein Mittelpunkt auf dem Probenrand, nicht am Bildrand."""
        hoehe, breite = self.probe.shape
        rand = radius + ABSTAND_PX
        for _ in range(200):
            y, x = (int(v) for v in self.randpunkte[int(rng.integers(len(self.randpunkte)))])
            if rand <= x < breite - rand and rand <= y < hoehe - rand:
                return x, y
        return None

    def frei(self, maske: np.ndarray) -> bool:
        groesser = cv2.dilate(maske.astype(np.uint8),
                              np.ones((2 * ABSTAND_PX + 1,) * 2, np.uint8)).astype(bool)
        return not (groesser & self.belegt).any()

    def belegen(self, maske: np.ndarray) -> None:
        self.belegt |= maske


def _leer(probe_form) -> np.ndarray:
    return np.zeros(probe_form, dtype=np.uint8)


def _kreis(form, x, y, r) -> np.ndarray:
    m = _leer(form)
    cv2.circle(m, (x, y), r, 1, -1, lineType=cv2.LINE_8)
    return m.astype(bool)


def _ellipse(form, x, y, a, b, winkel) -> np.ndarray:
    m = _leer(form)
    cv2.ellipse(m, (x, y), (a, b), winkel, 0, 360, 1, -1, lineType=cv2.LINE_8)
    return m.astype(bool)


def _vieleck(form, punkte) -> np.ndarray:
    m = _leer(form)
    cv2.fillPoly(m, [np.round(np.asarray(punkte)).astype(np.int32)], 1, lineType=cv2.LINE_8)
    return m.astype(bool)


def _saubere_form(rng, platz: _Platz):
    form = platz.probe.shape
    art = str(rng.choice(["kreis", "ellipse", "rechteck", "polygon"]))
    for _ in range(20):
        if art == "kreis":
            r = int(rng.integers(5, 29))
            mitte = platz.innen(rng, r)
            if mitte is None:
                continue
            maske = _kreis(form, *mitte, r)
            eintrag = {"parameter": {"mitte": mitte, "radius": r},
                       "analytisch": {"flaeche_px": math.pi * r * r, "feret_max_px": 2.0 * r}}
        elif art == "ellipse":
            a = int(rng.integers(7, 31))
            b = max(4, int(round(a / rng.uniform(1.0, 2.5))))
            w = float(rng.uniform(0, 180))
            mitte = platz.innen(rng, a)
            if mitte is None:
                continue
            maske = _ellipse(form, *mitte, a, b, w)
            eintrag = {"parameter": {"mitte": mitte, "halbachsen": [a, b], "winkel_grad": w},
                       "analytisch": {"flaeche_px": math.pi * a * b, "feret_max_px": 2.0 * a}}
        elif art == "rechteck":
            lang = float(rng.uniform(10, 46))
            kurz = max(6.0, lang / rng.uniform(1.0, 2.2))
            w = float(rng.uniform(0, 180))
            mitte = platz.innen(rng, math.hypot(lang, kurz) / 2)
            if mitte is None:
                continue
            ecken = cv2.boxPoints(((float(mitte[0]), float(mitte[1])), (lang, kurz), w))
            maske = _vieleck(form, ecken)
            eintrag = {"parameter": {"mitte": mitte, "seiten": [lang, kurz], "winkel_grad": w},
                       "analytisch": {"flaeche_px": lang * kurz,
                                      "feret_max_px": math.hypot(lang, kurz)}}
        else:
            r = float(rng.uniform(8, 29))
            n = int(rng.integers(7, 12))
            mitte = platz.innen(rng, r * 1.3)
            if mitte is None:
                continue
            winkel = np.sort(rng.uniform(0, 2 * math.pi, size=n))
            radien = r * rng.uniform(0.75, 1.25, size=n)
            punkte = np.stack([mitte[0] + radien * np.cos(winkel),
                               mitte[1] + radien * np.sin(winkel)], axis=1)
            maske = _vieleck(form, punkte)
            x, y = punkte[:, 0], punkte[:, 1]
            flaeche = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            feret = float(np.sqrt(((punkte[:, None] - punkte[None]) ** 2).sum(axis=2)).max())
            eintrag = {"parameter": {"mitte": mitte, "ecken": np.round(punkte, 2).tolist()},
                       "analytisch": {"flaeche_px": float(flaeche), "feret_max_px": feret}}
        if not platz.frei(maske):
            continue
        platz.belegen(maske)
        return maske, {"art": art, **eintrag}
    return None


def _randform(rng, platz: _Platz):
    form = platz.probe.shape
    for _ in range(30):
        r = int(rng.integers(8, 22))
        mitte = platz.auf_rand(rng, r)
        if mitte is None:
            return None
        if rng.random() < 0.5:
            maske = _kreis(form, *mitte, r)
            eintrag = {"art": "kreis", "parameter": {"mitte": mitte, "radius": r}}
        else:
            b = max(5, int(r / rng.uniform(1.2, 2.0)))
            w = float(rng.uniform(0, 180))
            maske = _ellipse(form, *mitte, r, b, w)
            eintrag = {"art": "ellipse",
                       "parameter": {"mitte": mitte, "halbachsen": [r, b], "winkel_grad": w}}
        innen = (maske & platz.probe).sum() / maske.sum()
        if not 0.3 <= innen <= 0.8 or not platz.frei(maske):
            continue
        platz.belegen(maske)
        eintrag["anteil_in_probe"] = float(innen)
        return maske, eintrag
    return None


def _paar(rng, platz: _Platz):
    """Zwei Kreise, die sich auf einer kurzen Strecke berühren. Die gemeinsamen Pixel
    gehören dem Kreis, dessen Mitte näher liegt."""
    form = platz.probe.shape
    for _ in range(30):
        r1, r2 = int(rng.integers(8, 20)), int(rng.integers(8, 20))
        mitte = platz.innen(rng, r1 + 2 * r2)
        if mitte is None:
            return None
        w = rng.uniform(0, 2 * math.pi)
        d = r1 + r2 - 1
        m2 = (int(round(mitte[0] + d * math.cos(w))), int(round(mitte[1] + d * math.sin(w))))
        a, b = _kreis(form, *mitte, r1), _kreis(form, *m2, r2)
        if not platz.frei(a | b) or not (platz.probe[a | b]).all():
            continue
        yy, xx = np.nonzero(a & b)
        naeher_a = np.hypot(xx - mitte[0], yy - mitte[1]) <= np.hypot(xx - m2[0], yy - m2[1])
        b[yy[naeher_a], xx[naeher_a]] = False
        a[yy[~naeher_a], xx[~naeher_a]] = False
        if not (a | b).any() or not _beruehren(a, b):
            continue
        platz.belegen(a | b)
        return [(a, {"art": "kreis", "parameter": {"mitte": mitte, "radius": r1},
                     "analytisch": {"flaeche_px": math.pi * r1 * r1, "feret_max_px": 2.0 * r1}}),
                (b, {"art": "kreis", "parameter": {"mitte": m2, "radius": r2},
                     "analytisch": {"flaeche_px": math.pi * r2 * r2, "feret_max_px": 2.0 * r2}})]
    return None


def _beruehren(a: np.ndarray, b: np.ndarray) -> bool:
    return bool((cv2.dilate(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & b).any())


def _winzig(rng, platz: _Platz):
    form = platz.probe.shape
    for _ in range(20):
        mitte = platz.innen(rng, 2)
        if mitte is None:
            return None
        if rng.random() < 0.5:
            maske = _kreis(form, *mitte, 1)                    # 5 Pixel, Kreuz
            eintrag = {"art": "kreis", "parameter": {"mitte": mitte, "radius": 1}}
        else:
            maske = _vieleck(form, [(mitte[0], mitte[1]), (mitte[0] + 1, mitte[1]),
                                    (mitte[0] + 1, mitte[1] + 1), (mitte[0], mitte[1] + 1)])
            eintrag = {"art": "quadrat", "parameter": {"mitte": mitte, "seite": 2}}
        if not platz.frei(maske):
            continue
        platz.belegen(maske)
        return maske, eintrag
    return None


def _kratzer(rng, platz: _Platz):
    form = platz.probe.shape
    for _ in range(30):
        laenge = float(rng.uniform(50, 110))
        dicke = int(rng.integers(2, 4))
        mitte = platz.innen(rng, laenge / 2)
        if mitte is None:
            return None
        w = rng.uniform(0, math.pi)
        dx, dy = laenge / 2 * math.cos(w), laenge / 2 * math.sin(w)
        p1 = (int(round(mitte[0] - dx)), int(round(mitte[1] - dy)))
        p2 = (int(round(mitte[0] + dx)), int(round(mitte[1] + dy)))
        m = _leer(form)
        cv2.line(m, p1, p2, 1, dicke, lineType=cv2.LINE_8)
        maske = m.astype(bool)
        if not platz.frei(maske):
            continue
        platz.belegen(maske)
        return maske, {"art": "linie", "parameter": {"von": p1, "bis": p2, "dicke": dicke}}
    return None


def _png(pfad: Path, bild: np.ndarray) -> None:
    ok, puffer = cv2.imencode(".png", bild)
    if not ok:
        raise OSError(f"{pfad.name} ließ sich nicht kodieren")
    puffer.tofile(str(pfad))
