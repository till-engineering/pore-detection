"""Hintergrundabzug mit Hysterese. **Das Standardverfahren.**

Zwei Einsichten tragen dieses Verfahren, und beide sind der Grund, warum eine schlichte
Grauwertschwelle an echten Schliffbildern scheitert.

**1. Eine Pore ist nicht dunkel - sie ist dunkler als ihre Umgebung.**

Keine Aufnahme ist gleichmäßig ausgeleuchtet. Eine Pore am abgeschatteten Bildrand hat
denselben Grauwert wie helles Grundgefüge in der Bildmitte; jede globale Schwelle muss
daran scheitern, und zwar auf beide Arten gleichzeitig - sie übersieht die Randpore und
nimmt die Bildmitte mit. Entschieden wird deshalb nicht auf dem Grauwert, sondern auf dem
**Abstand zum geschätzten Untergrund**. Dieser Abstand ist die physikalisch gemeinte
Größe: ein Hohlraum wirft weniger Licht zurück als das Metall *an derselben Stelle*.

Wie der Untergrund geschätzt wird, steht in ``untergrund.py`` und ist in der
einstellungen.yaml wählbar (``background.method``).

**Ergänzung für große Poren** (``grosse_poren``, abschaltbar): Was absolut sehr dunkel ist
(dunkelste Multi-Otsu-Klasse) und eine Mindestfläche hat, zählt zusätzlich als Pore -
auch wenn es im Kontrastbild nicht auffällt, weil der Untergrund es "geschluckt" hat.

**2. Ob etwas eine Pore ist und wie weit sie reicht, sind zwei verschiedene Fragen.**

Poren haben weiche Ränder - die Optik verschleift den Übergang, und beim Polieren wird
die Kante verrundet, sodass der Grauwert zur Wand hin ansteigt. Eine einzelne Schwelle
muss dann zwei gegenläufige Aufgaben auf einmal lösen: streng genug, um die dunkle
Gefügesprenkelung nicht mitzunehmen, und locker genug, um die Pore bis an ihren Rand zu
erfassen. Beides zugleich geht nicht.

Die Hysterese trennt die Fragen. Eine **strenge** Schwelle entscheidet, *ob* an dieser
Stelle eine Pore ist - sie darf ruhig nur den Kern treffen. Eine **lockere** Schwelle
entscheidet, *wie weit* sie reicht, aber nur dort, wo die strenge schon angeschlagen hat.
Gefügesprenkel bleiben draußen, weil sie die strenge Schwelle nie erreichen, und die
erkannten Poren behalten trotzdem ihre volle Fläche.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..einstellungen import Abschnitt
from .base import DetectionInput, DetectionResult
from .untergrund import dunkle_bereiche


class LocalContrastDetector:
    """Findet Poren über ihren Kontrast zum geschätzten Untergrund."""

    name = "local_contrast"

    def detect(self, data: DetectionInput, cfg: Abschnitt) -> DetectionResult:
        local = cfg.local_contrast
        # Die Schwelle kommt aus dem UNGEGLÄTTETEN Bild. Glätten senkt vor allem das
        # Rauschen; eine daraus bestimmte MAD-Schwelle fiele so tief, dass die lockere
        # Schwelle im Grundrauschen läge und ganze Flächen als Pore durchgingen.
        inside = data.inside(data.contrast)
        # Die grobe Suche - ob und wo eine Pore ist - läuft dann auf dem geglätteten
        # Bild. Der Rand wird bei 'half_max' danach wieder am Original bestimmt.
        contrast = _smoothed(data.contrast, local.presmooth_sigma_px)

        strict = _strict_threshold(inside, local)
        strict = max(strict, local.min_contrast, local.strict_floor)
        loose = max(local.min_contrast, strict * local.grow_factor)

        notes: tuple[str, ...] = ()
        if inside.size == 0:
            notes += ("keine Probenfläche - es wurde nichts ausgewertet",)
            return DetectionResult(
                mask=np.zeros(data.shape, dtype=bool), method=self.name, notes=notes
            )

        core = (contrast >= strict) & data.specimen
        extent = (contrast >= loose) & data.specimen
        mask = _hysteresis(core, extent)
        if local.extent_method == "half_max":
            mask = _half_max(mask, data.contrast, data.specimen, strict, local)

        thresholds = {"streng": float(strict), "locker": float(loose)}
        gross = local.grosse_poren
        if gross.enabled:
            zusatz = _grosse_dunkle_bereiche(data, gross)
            if zusatz.any():
                thresholds["grosse_poren_px"] = float(zusatz.sum())
                mask = mask | zusatz
                core = core | zusatz

        if not core.any():
            notes += (
                (f"kein Pixel erreicht die strenge Schwelle ({strict:.1f} Graustufen "
                 f"Kontrast) - in diesem Bild ist keine Pore zu sehen"),
            )

        return DetectionResult(
            mask=mask,
            method=self.name,
            thresholds=thresholds,
            notes=notes,
        )


def _grosse_dunkle_bereiche(data: DetectionInput, cfg: Abschnitt) -> np.ndarray:
    """Zusammenhängende, absolut dunkle Bereiche ab ``min_area_px`` Fläche."""
    dunkel = dunkle_bereiche(data.gray, data.specimen, cfg.multiotsu_classes,
                             cfg.multiotsu_level)
    if not dunkel.any():
        return dunkel
    anzahl, labels, stats, _ = cv2.connectedComponentsWithStats(dunkel.astype(np.uint8), 8)
    gross = np.zeros(anzahl, dtype=bool)
    gross[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_area_px
    return gross[labels]


# --------------------------------------------------------------------------------------
# Schwellen
# --------------------------------------------------------------------------------------


def _strict_threshold(inside: np.ndarray, cfg: Abschnitt) -> float:
    """Die strenge Schwelle, bestimmt **nur** über Pixel innerhalb der Probe."""
    if inside.size == 0:
        return float("inf")

    if cfg.strict_method == "fixed":
        return float(cfg.strict_fixed)
    if cfg.strict_method == "percentile":
        return float(np.percentile(inside, cfg.strict_percentile))
    if cfg.strict_method == "mad":
        # Das Kontrastbild ist kein Zweiklassenproblem. Es ist ein Berg um null - das
        # Grundrauschen des Gefüges - mit einem dünnen Ausläufer nach oben, und dieser
        # Ausläufer sind die Poren. Gefragt ist deshalb nicht, wo man den Berg halbiert,
        # sondern ab wann ein Pixel nicht mehr zum Berg gehört.
        #
        # Median und mittlere absolute Abweichung beschreiben den Berg, ohne sich vom
        # Ausläufer verschieben zu lassen - genau das leisten Mittelwert und
        # Standardabweichung nicht. Der Faktor 1,4826 macht aus der MAD das Gegenstück
        # zur Standardabweichung einer Normalverteilung, damit "8 Sigma" auch acht
        # Sigma heißt.
        median = float(np.median(inside))
        mad = float(np.median(np.abs(inside - median)))
        return median + cfg.mad_sigmas * 1.4826 * mad

    # Otsu rechnet auf 8-Bit-Histogrammen. Der Kontrast wird dafür auf ein robustes
    # Maximum normiert, nicht auf das echte - ein einzelner Ausreißer staucht sonst den
    # ganzen relevanten Bereich in wenige Histogrammstufen.
    values = np.clip(inside, 0, None)
    high = float(np.percentile(values, 99.5))
    if high <= 0:
        return float("inf")
    as_bytes = np.clip(values / high * 255.0, 0, 255).astype(np.uint8)

    if cfg.strict_method == "multiotsu_masked":
        from skimage.filters import threshold_multiotsu

        try:
            levels = threshold_multiotsu(as_bytes, classes=cfg.multiotsu_classes)
        except ValueError:
            level, _ = cv2.threshold(as_bytes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            index = cfg.multiotsu_level
            level = float(levels[index if index >= 0 else len(levels) + index])
    else:
        level, _ = cv2.threshold(as_bytes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return float(level) / 255.0 * high


def _smoothed(contrast: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return contrast
    return cv2.GaussianBlur(contrast, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _half_max(
    mask: np.ndarray,
    contrast: np.ndarray,
    specimen: np.ndarray,
    strict: float,
    cfg: Abschnitt,
) -> np.ndarray:
    """Den Rand jeder Pore aus ihr selbst bestimmen statt aus einer Bildschwelle.

    Eine globale lockere Schwelle gibt jeder Pore denselben absoluten Rand. Für eine
    blasse Pore passt er, eine tiefschwarze bekommt damit den ganzen Übergang zum Metall
    zugeschlagen - der Saum wächst mit der Tiefe der Pore. Hier liegt der Rand dort, wo
    der Kontrast einen festen **Anteil** des Wegs vom lokalen Untergrund zum Kern
    erreicht hat; das ist unabhängig davon, wie tief die Pore ist.

    Gemessen wird am ungeglätteten Kontrastbild. Die Pore kann dabei nur schrumpfen,
    nicht über ihre Fläche aus der Hysterese hinauswachsen, und ein Teilstück bleibt nur,
    wenn es selbst die strenge Schwelle erreicht.
    """
    if not mask.any():
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    hoehe, breite = mask.shape
    ring = cfg.half_max_ring_px
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    out = np.zeros_like(mask)

    for index in range(1, count):
        x, y, w, h = (int(v) for v in stats[index, :4])
        x0, y0 = max(0, x - ring), max(0, y - ring)
        x1, y1 = min(breite, x + w + ring), min(hoehe, y + h + ring)

        pore = labels[y0:y1, x0:x1] == index
        werte = contrast[y0:y1, x0:x1]
        umgebung = (cv2.dilate(pore.astype(np.uint8), kern).astype(bool)
                    & ~pore & ~mask[y0:y1, x0:x1] & specimen[y0:y1, x0:x1])

        spitze = float(werte[pore].max())
        untergrund = float(np.median(werte[umgebung])) if umgebung.any() else 0.0
        stufe = untergrund + cfg.half_max_fraction * (spitze - untergrund)
        stufe = max(stufe, cfg.min_contrast)

        teil = pore & (werte >= stufe)
        if not teil.any():
            continue
        teil = _hysteresis(teil & (werte >= strict), teil)
        out[y0:y1, x0:x1] |= teil

    return out


def _hysteresis(core: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Alle Gebiete aus ``extent`` behalten, die mindestens ein Pixel aus ``core`` enthalten.

    Das ist der ganze Trick: die lockere Schwelle darf großzügig sein, weil sie nichts
    Eigenes ins Ergebnis bringt - sie erweitert nur, was die strenge bereits bestätigt hat.
    """
    if not core.any():
        return np.zeros_like(core)

    count, labels = cv2.connectedComponents(extent.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(core)

    bestaetigt = np.unique(labels[core])
    keep = np.zeros(count, dtype=bool)
    keep[bestaetigt[bestaetigt > 0]] = True
    return keep[labels]
