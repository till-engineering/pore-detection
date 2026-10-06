"""Untergrund schätzen - das, wogegen der Kontrast einer Pore gemessen wird.

Wählbar über ``pore.local_contrast.background.method`` in der einstellungen.yaml:

``closing``      Morphologische Schließung (bisheriges Verfahren). Füllt alles auf, was
                 dunkler und **kleiner als der Kernel** ist. Schnell und lokal genau, aber:
                 eine Pore, die größer als der Kernel ist, wird selbst zum Untergrund und
                 verschwindet aus dem Kontrastbild.
``flaechenfit``  Glatte Polynomfläche über das Gefüge, die dunklen Bereiche ausgespart.
                 Unabhängig von der Porengröße; gleicht großräumige Ausleuchtungsverläufe
                 aus (Vignettierung), aber keine kleinräumigen Helligkeitsunterschiede.
``maskiert``     Glättung nur über die Gefügepixel ("normierte Faltung"); über den
                 dunklen Bereichen wird aus der Umgebung interpoliert. Lokaler als der
                 Flächenfit, und große Poren bleiben trotzdem erhalten.
``aus``          Kein Untergrund: eine Konstante (Median-Helligkeit der Probe). Der
                 Kontrast ist dann einfach "dunkler als das typische Gefüge" - eine
                 globale Schwelle ohne Ausleuchtungskorrektur.

``flaechenfit`` und ``maskiert`` brauchen eine Vormaske der dunklen Bereiche, die beim
Schätzen ausgespart werden: die dunkelste Multi-Otsu-Klasse innerhalb der Probe, etwas
erweitert. Beide rechnen auf einem verkleinerten Bild (``rechen_max_px``) - der
Untergrund ist glatt, volle Auflösung kostet nur Zeit.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..einstellungen import Abschnitt


def estimate_background(gray: np.ndarray, cfg: Abschnitt,
                        specimen: np.ndarray | None = None) -> np.ndarray:
    """Der Untergrund als float32-Bild in Bildgröße. ``cfg`` ist der Abschnitt
    ``pore.local_contrast.background``, ``specimen`` die Probenmaske (optional)."""
    if specimen is None or not specimen.any():
        specimen = np.ones(gray.shape, dtype=bool)

    methode = cfg.method
    if methode == "closing":
        return _closing(gray, cfg)
    if methode == "aus":
        return np.full(gray.shape, float(np.median(gray[specimen])), dtype=np.float32)
    if methode in ("flaechenfit", "maskiert"):
        return _ausgespart(gray, specimen, cfg, methode)
    raise ValueError(f"Unbekanntes Untergrund-Verfahren {methode!r} - "
                     "möglich: closing | flaechenfit | maskiert | aus")


def contrast_image(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Wie viel dunkler ist dieses Pixel als sein Untergrund? Positiv = dunkler."""
    return background - gray.astype(np.float32)


def dunkle_bereiche(gray: np.ndarray, specimen: np.ndarray, klassen: int,
                    stufe: int = 0) -> np.ndarray:
    """Alles in der Probe, was in die dunkelste(n) Multi-Otsu-Klasse(n) fällt.

    ``stufe`` 0 = nur die dunkelste Klasse, 1 = die beiden dunkelsten, ...
    """
    werte = gray[specimen]
    if werte.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    from skimage.filters import threshold_multiotsu

    try:
        grenzen = threshold_multiotsu(werte, classes=max(2, int(klassen)))
        grenze = float(grenzen[min(int(stufe), len(grenzen) - 1)])
    except ValueError:      # zu wenige verschiedene Grauwerte
        grenze, _ = cv2.threshold(werte, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return (gray <= grenze) & specimen


# --------------------------------------------------------------------------------------
# Die Verfahren
# --------------------------------------------------------------------------------------


def _closing(gray: np.ndarray, cfg: Abschnitt) -> np.ndarray:
    """Morphologische Schließung + Glättung. Der Kernel MUSS größer als die größte Pore
    sein, sonst wird die Pore als Untergrund geschätzt."""
    kernel_px = cfg.kernel_px
    if kernel_px <= 0:
        kernel_px = max(cfg.kernel_min_px, min(gray.shape[:2]) // cfg.kernel_divisor)
    kernel_px = max(3, kernel_px | 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(background.astype(np.float32), (0, 0),
                            kernel_px / cfg.smooth_sigma_factor, borderType=cv2.BORDER_REPLICATE)


def _ausgespart(gray: np.ndarray, specimen: np.ndarray, cfg: Abschnitt,
                methode: str) -> np.ndarray:
    """Untergrund nur aus dem Gefüge schätzen - dunkle Bereiche und Nicht-Probe ausgespart."""
    hoehe, breite = gray.shape
    faktor = min(1.0, cfg.rechen_max_px / max(hoehe, breite))
    klein = (max(1, round(breite * faktor)), max(1, round(hoehe * faktor)))
    g = cv2.resize(gray, klein, interpolation=cv2.INTER_AREA).astype(np.float32)
    probe = cv2.resize(specimen.astype(np.uint8), klein, interpolation=cv2.INTER_NEAREST) > 0

    # Gefüge = Probe ohne die dunklen Bereiche (etwas erweitert, damit der weiche
    # Porenrand nicht in die Schätzung eingeht).
    dunkel = dunkle_bereiche(g.astype(np.uint8), probe, cfg.dunkel_klassen)
    rand = max(0, round(cfg.dunkel_erweitern_px * faktor))
    if rand:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rand + 1, 2 * rand + 1))
        dunkel = cv2.dilate(dunkel.astype(np.uint8), kern).astype(bool)
    gefuege = probe & ~dunkel

    rueckfall = float(np.median(g[probe])) if probe.any() else float(np.median(g))
    if gefuege.sum() < 50:          # zu wenig Gefüge, um etwas zu schätzen
        return np.full(gray.shape, rueckfall, dtype=np.float32)

    if methode == "flaechenfit":
        untergrund = _polynom(g, gefuege, int(cfg.polynom_grad))
    else:
        sigma = cfg.maske_sigma_px if cfg.maske_sigma_px > 0 else min(hoehe, breite) / 16
        untergrund = _normierte_faltung(g, gefuege, max(1.0, sigma * faktor), rueckfall)

    return cv2.resize(untergrund.astype(np.float32), (breite, hoehe),
                      interpolation=cv2.INTER_LINEAR)


def _polynom(g: np.ndarray, gefuege: np.ndarray, grad: int) -> np.ndarray:
    """Polynomfläche vom Grad ``grad`` (Kleinste Quadrate) über die Gefügepixel.

    Zweimal gerechnet: im zweiten Durchgang fallen Pixel heraus, die deutlich dunkler
    als die erste Fläche sind - Porenreste, die die Vormaske nicht erwischt hat.
    """
    hoehe, breite = g.shape
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    xx = xx / max(breite - 1, 1) * 2 - 1        # auf [-1, 1] normiert: gut konditioniert
    yy = yy / max(hoehe - 1, 1) * 2 - 1
    terme = [xx ** i * yy ** j for i in range(grad + 1) for j in range(grad + 1 - i)]

    auswahl = gefuege.copy()
    for _ in range(2):
        idx = np.flatnonzero(auswahl)
        if idx.size > 200_000:                  # Stichprobe reicht für eine glatte Fläche
            idx = np.random.default_rng(0).choice(idx, 200_000, replace=False)
        a = np.stack([t.ravel()[idx] for t in terme], axis=1)
        koeff, *_ = np.linalg.lstsq(a, g.ravel()[idx], rcond=None)
        flaeche = sum(k * t for k, t in zip(koeff, terme, strict=True))
        rest = g - flaeche
        streuung = 1.4826 * float(np.median(np.abs(rest[gefuege]))) or 1.0
        auswahl = gefuege & (rest > -3 * streuung)
    return flaeche


def _normierte_faltung(g: np.ndarray, gefuege: np.ndarray, sigma: float,
                       rueckfall: float) -> np.ndarray:
    """Gauß-Glättung nur über die Gefügepixel: Summe(Wert·Gewicht) / Summe(Gewicht).

    Liegt ein Pixel so tief in einer großen Pore, dass im Radius kein Gefüge mehr ist,
    wird mit doppeltem Radius nachgefüllt - bis alles einen Wert hat.
    """
    w = gefuege.astype(np.float32)
    untergrund = np.full(g.shape, np.nan, dtype=np.float32)
    for _ in range(6):
        zaehler = cv2.GaussianBlur(g * w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        nenner = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        frei = np.isnan(untergrund) & (nenner > 1e-3)
        untergrund[frei] = zaehler[frei] / nenner[frei]
        if not np.isnan(untergrund).any():
            return untergrund
        sigma *= 2
    untergrund[np.isnan(untergrund)] = rueckfall
    return untergrund
