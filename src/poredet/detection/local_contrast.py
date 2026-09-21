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

Der Untergrund wird morphologisch geschätzt - Schließung mit einem Kernel, der größer ist
als die größte erwartete Pore. Das ist die kritische Bedingung: ist der Kernel zu klein,
wird die Pore selbst als Untergrund geschätzt und verschwindet spurlos aus dem
Kontrastbild.

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

from ..config.schema import BackgroundConfig, LocalContrastConfig, PoreConfig
from .base import DetectionInput, DetectionResult


class LocalContrastDetector:
    """Findet Poren über ihren Kontrast zum geschätzten Untergrund."""

    name = "local_contrast"

    def detect(self, data: DetectionInput, cfg: PoreConfig) -> DetectionResult:
        local = cfg.local_contrast
        contrast = data.contrast
        inside = data.inside(contrast)

        strict = _strict_threshold(inside, local)
        strict = max(strict, local.min_contrast)
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

        if not core.any():
            notes += (
                (f"kein Pixel erreicht die strenge Schwelle ({strict:.1f} Graustufen "
                 f"Kontrast) - in diesem Bild ist keine Pore zu sehen"),
            )

        return DetectionResult(
            mask=mask,
            method=self.name,
            thresholds={"streng": float(strict), "locker": float(loose)},
            notes=notes,
        )


# --------------------------------------------------------------------------------------
# Untergrund
# --------------------------------------------------------------------------------------


def estimate_background(gray: np.ndarray, cfg: BackgroundConfig) -> np.ndarray:
    """Der Untergrund, gegen den der Kontrast gemessen wird.

    Morphologische Schließung: sie füllt alles auf, was **dunkler und kleiner** als der
    Kernel ist - also genau die Poren - und lässt großflächige Helligkeitsverläufe stehen.
    Die anschließende Glättung nimmt die Treppenstufen heraus, die die Schließung an
    Kanten hinterlässt.

    Der Kernel muss größer sein als die größte erwartete Pore. Ist er zu klein, schließt
    sich die Pore nicht, wird als Untergrund geschätzt - und hebt sich damit von sich
    selbst nicht mehr ab.
    """
    kernel_px = cfg.kernel_px
    if kernel_px <= 0:
        kernel_px = max(cfg.kernel_min_px, min(gray.shape[:2]) // cfg.kernel_divisor)
    kernel_px = max(3, kernel_px | 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE
    )
    return cv2.GaussianBlur(
        background.astype(np.float32), (0, 0), kernel_px / cfg.smooth_sigma_factor,
        borderType=cv2.BORDER_REPLICATE,
    )


def contrast_image(gray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Wie viel dunkler ist dieses Pixel als sein Untergrund? Positiv = dunkler."""
    return background - gray.astype(np.float32)


# --------------------------------------------------------------------------------------
# Schwellen
# --------------------------------------------------------------------------------------


def _strict_threshold(inside: np.ndarray, cfg: LocalContrastConfig) -> float:
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
