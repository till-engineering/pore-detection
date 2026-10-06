"""Morphologie, Löcher füllen, Rauschgrenze - und das Label-Bild daraus.

Der gemeinsame Abschluss aller Detektoren. Was hier steht, hängt nicht am Verfahren,
sondern an der Sache, und gilt deshalb für jeden Detektor gleich - auch für einen, der
später dazukommt.

Zwei Schritte verdienen eine Begründung:

**Löcher füllen.** Eine Pore ist zur Wand hin heller (Kantenverrundung beim Polieren) und
trägt auf dem Boden einen Helligkeitsverlauf aus der schrägen Beleuchtung. Beides reißt
Löcher in die Kandidatenmaske, wenn die Schwelle knapp sitzt. Eine Pore mit einem Loch in
der Mitte ist aber keine zwei Poren mit einem Ring dazwischen - die Fläche wäre zu klein
und die Form unbrauchbar.

**Die Rauschgrenze ist keine fachliche Grenze.** ``min_size_px`` wirft weg, was aus
einzelnen Rauschpixeln besteht. Was eine Pore fachlich *sein darf* - Mindestgröße,
Rundheit, Seitenverhältnis - entscheiden die Filter in ``poren/filter.py``,
und die tun es nachvollziehbar, mit Begründung je verworfenem Objekt.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from ..einstellungen import Abschnitt


def clean(mask: np.ndarray, cfg: Abschnitt) -> np.ndarray:
    """Kandidatenmaske aufräumen."""
    out = mask
    if cfg.open_radius_px > 0:
        out = _morph(out, cv2.MORPH_OPEN, cfg.open_radius_px)
    if cfg.close_radius_px > 0:
        out = _morph(out, cv2.MORPH_CLOSE, cfg.close_radius_px)
    if cfg.fill_holes and out.any():
        out = binary_fill_holes(out)
    return out


def label_image(mask: np.ndarray, cfg: Abschnitt) -> np.ndarray:
    """Aus der Maske ein Label-Bild, ohne die Objekte unterhalb der Rauschgrenze."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        keep[index] = stats[index, cv2.CC_STAT_AREA] >= cfg.min_size_px

    # Neu durchnummerieren, damit die Labels lückenlos bei 1 beginnen.
    mapping = np.zeros(count, dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return mapping[labels]


def split_touching(labels: np.ndarray, cfg: Abschnitt) -> tuple[np.ndarray, int]:
    """Zusammengewachsene Porennester in einzelne Poren trennen.

    Zwei berührende Poren als eine zu zählen verfälscht gleich drei Kennzahlen: die
    Anzahl, die Größenverteilung und die größte Pore - und letztere ist oft das
    eigentliche Abnahmekriterium.

    Das Werkzeug ist die Distanztransformation: in einem Nest aus zwei verschmolzenen
    Poren gibt es zwei lokale Maxima des Abstands zum Rand. Von ihnen aus geflutet
    (Watershed) fällt die Trennlinie genau in die Einschnürung zwischen beiden.

    Eine Sicherung ist nötig, denn Lunker sind von Natur aus zerklüftet und haben viele
    kleine Nebenmaxima. Ein Teilstück unterhalb von ``min_area_ratio`` ist so ein
    Ausfransen und wird zurückgenommen - sonst zerlegt das Verfahren eine einzelne
    verzweigte Pore in ein Dutzend Scherben.
    """
    if not cfg.split.enabled or not labels.any():
        return labels, 0

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    mask = labels > 0
    distance = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(
        distance, min_distance=cfg.split.min_distance_px, labels=mask, exclude_border=False
    )
    if len(peaks) == 0:
        return labels, 0

    markers = np.zeros(labels.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    markers, _ = ndi.label(markers > 0)

    split = watershed(-distance, markers, mask=mask)
    split = _undo_slivers(split, labels, cfg.split.min_area_ratio)

    vorher = int(labels.max())
    nachher = int(split.max())
    return split.astype(np.int32), max(0, nachher - vorher)


def _undo_slivers(split: np.ndarray, original: np.ndarray, min_ratio: float) -> np.ndarray:
    """Zu kleine Teilstücke wieder mit ihrem Nachbarn verschmelzen.

    Geprüft wird gegen das **ursprüngliche** Objekt: ein Teilstück, das weniger als
    ``min_ratio`` davon ausmacht, ist kein zweiter Hohlraum, sondern ein Nebenmaximum der
    Distanztransformation. Gerechnet wird je Objekt nur in dessen umschließendem
    Kasten - nicht über das ganze Bild, das wäre bei vielen Poren sehr langsam.
    """
    from scipy import ndimage as ndi

    out = split.copy()
    for original_label, fenster in enumerate(ndi.find_objects(original), start=1):
        if fenster is None:
            continue
        region = original[fenster] == original_label
        teil_bild = out[fenster]                      # Sicht auf out - Änderungen wirken
        teile, counts = np.unique(teil_bild[region], return_counts=True)
        gueltig = teile > 0
        teile, counts = teile[gueltig], counts[gueltig]
        if len(teile) <= 1:
            continue

        gesamt = float(region.sum())
        grosstes = teile[int(np.argmax(counts))]
        for teil, anzahl in zip(teile, counts, strict=True):
            if anzahl / gesamt < min_ratio:
                teil_bild[region & (teil_bild == teil)] = grosstes

    # Nach dem Verschmelzen neu durchnummerieren.
    werte = np.unique(out)
    werte = werte[werte > 0]
    mapping = np.zeros(int(out.max()) + 1, dtype=np.int32)
    mapping[werte] = np.arange(1, len(werte) + 1, dtype=np.int32)
    return mapping[out]


def _morph(mask: np.ndarray, op: int, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask.astype(np.uint8), op, kernel).astype(bool)
