"""Die gemeinsame Grundlage aller Segmentierer: Helligkeit, Textur, Arbeitsauflösung.

Jedes Verfahren in diesem Paket beantwortet dieselbe Frage auf einem anderen Weg. Damit
ein Vergleich etwas über die **Verfahren** aussagt und nicht über ihre Vorverarbeitung,
rechnen alle auf denselben Signalen:

* dem auf Arbeitsgröße verkleinerten Graubild,
* der lokalen Standardabweichung als Maß für Struktur, **in voller Auflösung gemessen**
  und erst danach verkleinert,
* zwei je Bild bestimmten Schwellen - eine für "dunkel", eine für "strukturlos".

Die Schwellen sind hier kein Ergebnis, sondern Ausgangsmaterial. Was ein Verfahren daraus
macht - hart schneiden, Saaten setzen, eine Energie minimieren -, ist seine Sache.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..einstellungen import Abschnitt


@dataclass(frozen=True)
class Signals:
    """Was jedes Verfahren an Vorarbeit bekommt."""

    work: np.ndarray             # verkleinertes Graubild, uint8
    smoothed: np.ndarray         # dasselbe, entrauscht - Grundlage aller Entscheidungen
    texture: np.ndarray          # lokale Standardabweichung, float32
    dark_threshold: float
    texture_threshold: float
    separable: bool              # ließen sich die dunklen Pixel überhaupt aufteilen?
    full_shape: tuple[int, int]

    @property
    def is_dark(self) -> np.ndarray:
        return self.smoothed <= self.dark_threshold

    @property
    def is_smooth(self) -> np.ndarray:
        return self.texture <= self.texture_threshold

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.work.shape[0]), int(self.work.shape[1]))


def compute(gray: np.ndarray, cfg: Abschnitt) -> Signals:
    """Signale für ein Bild berechnen."""
    lasso = cfg.lasso

    # Die Textur wird in VOLLER Auflösung gemessen und erst danach verkleinert.
    # Andersherum mittelt das Verkleinern die feine Gefügestruktur weg und macht ein
    # dunkles Perlitband künstlich "strukturlos" - genau die Verwechslung, die dieses
    # Signal verhindern soll. An den Testbildern schrumpft der Unterschied Harz/Band
    # dabei von 3,2 : 9,9 auf 4,5 : 6,4 und ist praktisch weg.
    texture_full = local_std(denoise(gray, lasso.denoise_ksize), lasso.texture_ksize_px)

    work = downscale(gray, cfg.work_max_px)
    smoothed = denoise(work, lasso.denoise_ksize)
    texture = downscale_values(texture_full, work.shape[:2])

    dark_threshold = dark_level(smoothed, lasso)
    texture_threshold, separable = texture_level(texture, smoothed <= dark_threshold, lasso)

    return Signals(
        work=work,
        smoothed=smoothed,
        texture=texture,
        dark_threshold=dark_threshold,
        texture_threshold=texture_threshold,
        separable=separable,
        full_shape=(int(gray.shape[0]), int(gray.shape[1])),
    )


# --------------------------------------------------------------------------------------
# Signale
# --------------------------------------------------------------------------------------


def local_std(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Lokale Standardabweichung - das Maß für "hat hier etwas Struktur?".

    Über die Verschiebungsformel var = E[x²] - E[x]², beides als Boxfilter. Das ist um
    Größenordnungen schneller als ein gleitendes Fenster und liefert dasselbe.
    """
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (ksize, ksize))
    mean_of_squares = cv2.blur(values * values, (ksize, ksize))
    return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))


def dark_level(gray: np.ndarray, cfg: Abschnitt) -> float:
    """Helligkeitsschwelle, je Bild neu bestimmt."""
    if cfg.dark_method == "percentile":
        value = float(np.percentile(gray, cfg.dark_percentile))
    elif cfg.dark_method == "multiotsu":
        from skimage.filters import threshold_multiotsu

        try:
            thresholds = threshold_multiotsu(gray, classes=cfg.multiotsu_classes)
        except ValueError:
            # Zu wenige verschiedene Grauwerte für mehrere Klassen.
            value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            level = min(cfg.multiotsu_level, len(thresholds) - 1)
            value = float(thresholds[level])
    else:
        value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(value) + cfg.dark_offset


def texture_level(
    texture: np.ndarray, dark: np.ndarray, cfg: Abschnitt
) -> tuple[float, bool]:
    """Texturschwelle über die dunklen Pixel. Liefert Schwelle und ob sie trennt.

    Über das ganze Bild gerechnet landet Otsu weit im Ausläufer der Verteilung - an einem
    Testbild galten damit 99,5 % als strukturlos, und das Kriterium war wirkungslos. Unter
    den dunklen Pixeln dagegen steht genau die Frage an, um die es geht: Harz oder
    dunkles Gefüge?

    Der zweite Rückgabewert sichert gegen den Fall ab, dass es gar kein dunkles Gefüge
    gibt: dann ist die Verteilung einhügelig, Otsu legt trotzdem irgendwo eine Schwelle
    hinein und würde **das Harz selbst** zerschneiden.
    """
    values = texture[dark] if dark.any() else texture.ravel()
    if values.size == 0:
        return float("inf"), False

    if cfg.texture_method == "percentile":
        return float(np.percentile(values, cfg.texture_percentile)), True

    # Normiert wird auf ein robustes Maximum, nicht auf das echte. Otsu rechnet auf 256
    # Stufen; ein einzelner Ausreißer - eine Kante, ein Kratzer - staucht sonst den ganzen
    # interessanten Bereich in wenige Bins.
    high = float(np.percentile(values, cfg.texture_clip_percentile))
    if high <= 0:
        return float("inf"), False

    as_bytes = np.clip(values / high * 255.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(as_bytes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = float(level) / 255.0 * high

    if separability(as_bytes, int(level)) < cfg.min_texture_separability:
        return float("inf"), False
    return threshold, True


def separability(values: np.ndarray, level: int) -> float:
    """Otsus Trennschärfe: Anteil der Varianz, den die Aufteilung erklärt (0…1)."""
    low = values[values <= level]
    high = values[values > level]
    if low.size == 0 or high.size == 0:
        return 0.0
    total_variance = float(values.var())
    if total_variance <= 0:
        return 0.0
    weight_low = low.size / values.size
    weight_high = high.size / values.size
    between = weight_low * weight_high * (float(low.mean()) - float(high.mean())) ** 2
    return between / total_variance


# --------------------------------------------------------------------------------------
# Größen und Formen
# --------------------------------------------------------------------------------------


def downscale(gray: np.ndarray, max_px: int) -> np.ndarray:
    """Arbeitsauflösung. Die Harzgrenze ist eine grobe Struktur."""
    longest = max(gray.shape[:2])
    if longest <= max_px:
        return gray
    factor = max_px / longest
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def downscale_values(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Eine Messkarte auf die Arbeitsgröße bringen.

    Gemittelt, nicht abgetastet: der Mittelwert lokaler Rauigkeit über einen Block ist
    selbst wieder eine sinnvolle Rauigkeit, ein herausgegriffenes Pixel dagegen Zufall.
    """
    if values.shape[:2] == shape:
        return values
    return cv2.resize(values, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def upscale(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return resized.astype(bool)


def denoise(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return gray
    return cv2.medianBlur(gray, ksize if ksize % 2 else ksize + 1)


def disk(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(radius)).astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), disk(radius)).astype(bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), disk(radius)).astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Nur das größte zusammenhängende Gebiet behalten."""
    if not mask.any():
        return mask
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return mask
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == biggest


def border_band(shape: tuple[int, int], width_px: int) -> np.ndarray:
    """Ein Streifen entlang des Bildrands."""
    height, width = shape
    band = np.zeros((height, width), dtype=bool)
    w = max(1, width_px)
    band[:w, :] = True
    band[-w:, :] = True
    band[:, :w] = True
    band[:, -w:] = True
    return band


def border_distance(height: int, width: int) -> np.ndarray:
    """Für jedes Pixel der Abstand zum nächsten Bildrand."""
    rows = np.minimum(np.arange(height), height - 1 - np.arange(height))
    cols = np.minimum(np.arange(width), width - 1 - np.arange(width))
    return np.minimum(rows[:, None], cols[None, :]).astype(np.float32)
