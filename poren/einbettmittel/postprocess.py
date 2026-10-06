"""Der gemeinsame Abschluss aller Segmentierer.

Was hier steht, hängt nicht am Verfahren, sondern an der Sache:

* **Luftblasen und eingebrannte Overlays gehören zum Harz** - kleine Löcher werden
  gefüllt, große nicht. Umschließt das Harz die Probe als Rahmen, wäre das große Loch die
  Probe selbst.
* **Ein Fund muss sich vom Rest unterscheiden.** Irgendein Bereich sieht immer am ehesten
  nach Saum aus; ohne diese Abnahme meldet jedes Verfahren in Bildern ohne jedes
  Einbettmittel welches.
* **Eingeschlossene Inseln gehören dorthin, wonach sie aussehen.** Eine große Luftblase
  im Harz wird sonst zum eigenen "Probenstück" - und später zur größten Pore des Bildes.

Dass alle Verfahren denselben Abschluss durchlaufen, ist auch die Voraussetzung dafür,
dass ihr Vergleich etwas aussagt: Unterschiede im Ergebnis stammen dann aus der
Segmentierung und nicht aus unterschiedlich gründlichem Aufräumen.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from ..einstellungen import Abschnitt
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals


def finish(
    resin: np.ndarray,
    signals: Signals,
    cfg: Abschnitt,
    method: str,
    notes: tuple[str, ...] = (),
    debug: dict[str, np.ndarray] | None = None,
) -> SpecimenMask:
    """Aus einer rohen Harzmaske in Arbeitsauflösung das fertige Ergebnis machen."""
    lasso = cfg.lasso
    gray = signals.smoothed

    resin, dropped = keep_rim_like(resin, lasso)
    if dropped:
        notes += (
            (f"{dropped} dunkle Fläche(n) nicht als Einbettmittel gewertet - sie sind "
             f"zu klein oder liegen nicht am Bildrand entlang und sind damit Poren"),
        )

    resin = fill_small_holes(resin, lasso.max_hole_frac)
    resin = sig.close(resin, lasso.close_radius_px)

    contrast = material_contrast(resin, gray)
    if resin.any() and contrast < lasso.min_material_contrast:
        notes += (
            (f"Fund verworfen: der dunkle Randbereich hebt sich nur um {contrast:.0f} "
             f"Graustufen von der Probe ab (nötig: {lasso.min_material_contrast:.0f}). "
             f"Das ist dunkleres Gefüge, kein anderer Werkstoff."),
        )
        resin = np.zeros_like(resin)

    specimen, resin, components, more = specimen_from_resin(resin, gray, cfg)
    notes += more

    full_debug = {"work": signals.work, "texture": signals.texture,
                  "dark": signals.is_dark, "smooth": signals.is_smooth,
                  "resin_work": resin}
    if debug:
        full_debug.update(debug)

    specimen_full = sig.upscale(specimen, signals.full_shape)
    if cfg.erode_border_px > 0:
        specimen_full = sig.erode(specimen_full, cfg.erode_border_px)

    return SpecimenMask(
        specimen=specimen_full,
        resin=sig.upscale(resin, signals.full_shape),
        method=method,
        components=components,
        warnings=notes,
        debug=full_debug,
    )


def empty(signals: Signals, method: str, notes: tuple[str, ...] = ()) -> SpecimenMask:
    """Ergebnis "kein Einbettmittel" - das ganze Bild ist Probe."""
    shape = signals.full_shape
    return SpecimenMask(
        specimen=np.ones(shape, dtype=bool),
        resin=np.zeros(shape, dtype=bool),
        method=method,
        components=1,
        warnings=notes,
        debug={"work": signals.work, "texture": signals.texture,
               "dark": signals.is_dark, "smooth": signals.is_smooth,
               "resin_work": np.zeros(signals.shape, dtype=bool)},
    )


# --------------------------------------------------------------------------------------


def keep_rim_like(resin: np.ndarray, cfg) -> tuple[np.ndarray, int]:
    """Nur Harzflächen behalten, die ein Saum sein können.

    Zwei Bedingungen, und beide beschreiben dieselbe Sache aus verschiedenen Richtungen:

    * **am Bildrand entlang.** Das Sichtfeld liegt innerhalb des Einbettlings, Harz läuft
      also über den Rand hinaus - und zwar auf einer nennenswerten Länge. Eine
      angeschnittene Pore ist zwar auch randoffen, berührt den Rand aber nur auf einem
      kurzen Stück.
    * **groß genug.** Ein Harzfeld nimmt einen nennenswerten Teil des Bildes ein.

    Was durchfällt, ist eine **Pore** - und Poren sind das Messobjekt, nicht der Abfall.

    Die Regel steht hier und nicht in einem einzelnen Verfahren, weil sie keinem gehört.
    Die rein erscheinungsbasierten Verfahren brauchen sie besonders dringend: GrabCut
    lernt das Aussehen des Harzes und findet es in jeder Pore wieder. Ohne den ersten
    Teil entfernte es an einem synthetischen Prüfbild 98 % der Porenfläche, ohne den
    zweiten jede angeschnittene Randpore.
    """
    if not resin.any():
        return resin, 0

    height, width = resin.shape
    total = float(height * width)
    perimeter = 2.0 * (height + width)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        resin.astype(np.uint8), 8
    )
    border = np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )
    contact = np.bincount(border, minlength=count)

    out = np.zeros_like(resin)
    kept = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] / total < cfg.min_area_frac:
            continue
        if contact[index] / perimeter < cfg.min_border_contact_frac:
            continue
        out |= labels == index
        kept += 1
    return out, (count - 1) - kept


def fill_small_holes(mask: np.ndarray, max_hole_frac: float) -> np.ndarray:
    """Luftblasen und eingebrannte Overlays gehören zum Harz.

    Gefüllt wird nur, was klein ist. Umschließt das Harz die Probe als Rahmen, ist das
    große "Loch" die Probe selbst - sie zu füllen würde die ganze Trennung aufheben.
    """
    if not mask.any():
        return mask
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask

    limit = max_hole_frac * mask.size
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), 8
    )
    out = mask.copy()
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] <= limit:
            out |= labels == index
    return out


def material_contrast(resin: np.ndarray, gray: np.ndarray) -> float:
    """Wie weit sich das gefundene Harz vom Rest des Bildes abhebt, in Graustufen."""
    rest = ~resin
    if not resin.any() or not rest.any():
        return 0.0
    return float(np.median(gray[rest])) - float(np.median(gray[resin]))


def specimen_from_resin(
    resin: np.ndarray, gray: np.ndarray, cfg: Abschnitt
) -> tuple[np.ndarray, np.ndarray, int, tuple[str, ...]]:
    """Alles, was nicht Harz ist, ist Probe - bis auf zu kleine Fetzen und Fremdkörper.

    Die Zuordnung eingeschlossener Inseln ist eine Nächste-Nachbar-Frage: gleicht die
    Insel in der Helligkeit eher dem Harz oder der Hauptprobe? Sie geht dorthin, wo sie
    hingehört.
    """
    candidate = ~resin
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    if count <= 1:
        return candidate, resin, 0, (
            "kein Probenbereich übrig - alles gilt als Einbettmittel",
        )

    total = float(candidate.size)
    areas = [(index, int(stats[index, cv2.CC_STAT_AREA])) for index in range(1, count)]
    areas.sort(key=lambda item: -item[1])

    main = labels == areas[0][0]
    main_gray = float(np.median(gray[main]))
    resin_gray = float(np.median(gray[resin])) if resin.any() else None

    kept: list[tuple[int, int]] = [areas[0]]
    reassigned = 0
    dropped = 0
    out_resin = resin.copy()

    for index, area in areas[1:]:
        region = labels == index
        if resin_gray is not None:
            island_gray = float(np.median(gray[region]))
            if abs(island_gray - resin_gray) < abs(island_gray - main_gray):
                out_resin |= region
                reassigned += 1
                continue
        if cfg.keep == "largest" or area / total < cfg.min_specimen_frac:
            dropped += 1
            continue
        kept.append((index, area))

    specimen = np.zeros_like(candidate)
    for index, _area in kept:
        specimen |= labels == index

    notes: tuple[str, ...] = ()
    if reassigned:
        notes += (
            (f"{reassigned} eingeschlossene Insel(n) dem Einbettmittel zugeschlagen - "
             f"sie sehen aus wie Harz, nicht wie die Probe (vermutlich Luftblasen)"),
        )
    if dropped:
        notes += (f"{dropped} zu kleine Probenstück(e) verworfen",)
    return specimen, out_resin, len(kept), notes
