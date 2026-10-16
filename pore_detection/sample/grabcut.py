"""Embedding medium detection with GrabCut (graph cut with learned color models).

Ported from Pore_Detection_V2 (specimen.method: grabcut), where it was selected on a set of
12 comparison images: it covers the resin rim most completely. Downside: on images WITHOUT
embedding medium it reports some in about 5 of 14 cases.

Steps, all at a reduced working resolution (the resin border is a coarse structure):
1. Signals: smoothed gray image, local texture (measured at FULL resolution, then reduced –
   reducing first averages away fine microstructure and makes dark bands look "smooth"),
   one threshold for "dark" and one for "smooth" per image.
2. Seeds: sure resin = dark + smooth + in a band along the image border (resin surrounds
   the field of view, so a dark band in the middle can never become a seed);
   sure specimen = bright or textured, away from the border.
3. GrabCut learns a color model for resin and specimen from the seeds and cuts the graph.
   A border costs length, so thin dark bands inside the specimen stay specimen.
4. Clean-up: keep only resin areas that run along the image border and are large (anything
   else is a pore), fill small holes (air bubbles), discard the result if the resin is not
   clearly darker than the rest, assign enclosed islands to resin or specimen by gray value.

Self-contained (imports only core). CPU only: OpenCV's GrabCut has no GPU version, and the
method runs on a small working image, so it takes well under a second per image.
"""

import logging

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import RoiDetector

log = logging.getLogger("pore_detection")


@register("grabcut")
class GrabCut(RoiDetector):
    """Separates specimen and embedding medium by graph cut with learned color models."""

    params = {
        "work_max_px": "long edge of the working image; all *_work_px values refer to it",
        "iterations": "GrabCut iterations (1-10)",
        "denoise_ksize": "median filter size against noise",
        "texture_ksize_px": "window of the local texture (full resolution)",
        "dark_offset": "shift of the dark threshold in gray levels 0-255, + = more resin",
        "texture_clip_percentile": "robust maximum for normalising the texture",
        "min_texture_separability": "below it the texture criterion is switched off (0-1)",
        "border_band_frac": "width of the border band for resin seeds (share of short edge)",
        "seed_erode_work_px": "shrink seeds inwards",
        "min_seed_frac": "minimum size of a seed patch (share of image area)",
        "min_area_frac": "minimum size of a resin area (share of image area)",
        "min_border_contact_frac": "minimum share of the image perimeter a resin area touches",
        "max_hole_frac": "fill holes in the resin up to this size (share of image area)",
        "close_radius_work_px": "closes the resin rim over air bubbles",
        "min_material_contrast": "minimum gray difference resin <-> rest (0-255), else no resin",
        "min_specimen_frac": "drop smaller specimen pieces (share of image area)",
        "keep": "largest | all_above_min – which specimen pieces to keep",
        "erode_border_px": "shrink the specimen border (full resolution), 0 = off",
    }

    def detect_roi(self, info, original):
        full_shape = original.shape[:2]
        rgb8 = original if original.dtype == np.uint8 else (original >> 8).astype(np.uint8)
        gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)

        signals = self._signals(gray)
        work_shape = signals["smoothed"].shape
        resin_seed, specimen_seed = self._seeds(signals)
        if not (resin_seed.any() and specimen_seed.any()):
            log.info("%s: no embedding medium found (no usable seed at the border)", info.name)
            return np.ones(full_shape, dtype=bool)

        color = cv2.resize(rgb8, (work_shape[1], work_shape[0]), interpolation=cv2.INTER_AREA)
        mask = np.full(work_shape, cv2.GC_PR_FGD, dtype=np.uint8)  # unknown = "probably specimen"
        mask[specimen_seed] = cv2.GC_FGD
        mask[resin_seed] = cv2.GC_BGD
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(np.ascontiguousarray(color), mask, None, background_model, foreground_model,
                    self.iterations, cv2.GC_INIT_WITH_MASK)
        resin = (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)

        specimen = self._finish(resin, signals["smoothed"], info.name)
        specimen = _upscale(specimen, full_shape)
        if self.erode_border_px > 0:
            specimen = _erode(specimen, self.erode_border_px)
        return specimen

    # -- 1. signals --------------------------------------------------------------------------

    def _signals(self, gray: np.ndarray) -> dict:
        texture_full = _local_std(_denoise(gray, self.denoise_ksize), self.texture_ksize_px)
        work = _downscale(gray, self.work_max_px)
        smoothed = _denoise(work, self.denoise_ksize)
        texture = cv2.resize(texture_full, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_AREA)

        otsu, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dark = smoothed <= float(otsu) + self.dark_offset
        texture_threshold = self._texture_threshold(texture, dark)
        return {"smoothed": smoothed, "dark": dark, "smooth": texture <= texture_threshold}

    def _texture_threshold(self, texture: np.ndarray, dark: np.ndarray) -> float:
        """Otsu over the texture of the DARK pixels: resin or dark microstructure?

        inf (= texture criterion off) if the split does not separate, e.g. when there is no
        dark microstructure at all – Otsu would then cut the resin itself in two.
        """
        values = texture[dark] if dark.any() else texture.ravel()
        high = float(np.percentile(values, self.texture_clip_percentile)) if values.size else 0.0
        if high <= 0:
            return float("inf")
        as_bytes = np.clip(values / high * 255.0, 0, 255).astype(np.uint8)
        level, _ = cv2.threshold(as_bytes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if _separability(as_bytes, int(level)) < self.min_texture_separability:
            return float("inf")
        return float(level) / 255.0 * high

    # -- 2. seeds ----------------------------------------------------------------------------

    def _seeds(self, signals: dict) -> tuple[np.ndarray, np.ndarray]:
        dark, smooth = signals["dark"], signals["smooth"]
        height, width = dark.shape
        band = _border_band(dark.shape, max(2, int(self.border_band_frac * min(height, width))))
        min_area = self.min_seed_frac * height * width

        resin = _drop_small(_erode(dark & smooth & band, self.seed_erode_work_px), min_area)
        # "or": dark microstructure is specimen as long as it has texture
        specimen = _drop_small(_erode((~dark | ~smooth) & ~band, self.seed_erode_work_px), min_area)
        return resin, specimen

    # -- 4. clean-up -------------------------------------------------------------------------

    def _finish(self, resin: np.ndarray, gray: np.ndarray, name: str) -> np.ndarray:
        """Raw resin mask (working resolution) -> specimen mask (working resolution)."""
        resin = self._keep_rim_like(resin)
        resin = _fill_small_holes(resin, self.max_hole_frac)
        resin = _close(resin, self.close_radius_work_px)

        if resin.any() and (~resin).any():
            contrast = float(np.median(gray[~resin])) - float(np.median(gray[resin]))
            if contrast < self.min_material_contrast:
                log.info("%s: dark border area rejected as embedding medium (contrast %.0f < %.0f)",
                         name, contrast, self.min_material_contrast)
                resin = np.zeros_like(resin)
        return self._specimen_from_resin(resin, gray)

    def _keep_rim_like(self, resin: np.ndarray) -> np.ndarray:
        """Keep only resin areas that run along the image border and are large.

        Everything else is a pore – without this GrabCut finds the resin look in every pore.
        """
        if not resin.any():
            return resin
        height, width = resin.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(resin.view(np.uint8), connectivity=8)
        border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        contact = np.bincount(border, minlength=count)
        keep = ((stats[:, cv2.CC_STAT_AREA] / (height * width) >= self.min_area_frac)
                & (contact / (2.0 * (height + width)) >= self.min_border_contact_frac))
        keep[0] = False
        return keep[labels]

    def _specimen_from_resin(self, resin: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Everything that is not resin is specimen – except too small pieces.

        Enclosed islands go to whichever they resemble more in gray value: the resin or the
        largest specimen piece (an air bubble in the resin would otherwise become a "pore").
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats((~resin).view(np.uint8), connectivity=8)
        if count <= 1:
            return np.zeros_like(resin)

        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(-areas) + 1                  # labels, largest first
        medians = _medians_per_label(gray, labels, count)
        main_gray = medians[order[0]]
        resin_gray = float(np.median(gray[resin])) if resin.any() else None

        keep = np.zeros(count, dtype=bool)
        keep[order[0]] = True
        for label in order[1:]:
            if resin_gray is not None and abs(medians[label] - resin_gray) < abs(medians[label] - main_gray):
                continue                                # looks like resin
            if self.keep == "largest" or stats[label, cv2.CC_STAT_AREA] / resin.size < self.min_specimen_frac:
                continue                                # too small
            keep[label] = True
        return keep[labels]


# -- helpers ---------------------------------------------------------------------------------

def _downscale(gray: np.ndarray, max_px: int) -> np.ndarray:
    longest = max(gray.shape[:2])
    if longest <= max_px:
        return gray
    factor = max_px / longest
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def _upscale(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.view(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).view(bool)


def _denoise(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return gray
    return cv2.medianBlur(gray, ksize if ksize % 2 else ksize + 1)


def _local_std(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Local standard deviation via var = E[x²] - E[x]² with box filters."""
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (ksize, ksize))
    mean_of_squares = cv2.blur(values * values, (ksize, ksize))
    return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))


def _separability(values: np.ndarray, level: int) -> float:
    """Otsu's separability: share of the variance explained by the split (0…1)."""
    low, high = values[values <= level], values[values > level]
    total_variance = float(values.var())
    if low.size == 0 or high.size == 0 or total_variance <= 0:
        return 0.0
    between = (low.size / values.size) * (high.size / values.size) * (float(low.mean()) - float(high.mean())) ** 2
    return between / total_variance


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return cv2.erode(mask.view(np.uint8), _disk(radius)).view(bool)


def _close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return cv2.morphologyEx(mask.view(np.uint8), cv2.MORPH_CLOSE, _disk(radius)).view(bool)


def _border_band(shape: tuple[int, int], width_px: int) -> np.ndarray:
    band = np.zeros(shape, dtype=bool)
    band[:width_px, :] = band[-width_px:, :] = True
    band[:, :width_px] = band[:, -width_px:] = True
    return band


def _drop_small(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Drop patches below min_area (mostly noise); keeps the mask if nothing would remain."""
    if not mask.any():
        return mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    result = keep[labels]
    return result if result.any() else mask


def _fill_small_holes(mask: np.ndarray, max_hole_frac: float) -> np.ndarray:
    """Fill holes (areas not connected to the image border) up to a maximum size.

    Large holes are not filled: if the resin frames the specimen, the big "hole" is the
    specimen itself.
    """
    if not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).view(np.uint8), connectivity=4)
    on_border = np.zeros(count, dtype=bool)
    on_border[np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])] = True
    small = ~on_border & (stats[:, cv2.CC_STAT_AREA] <= max_hole_frac * mask.size)
    small[0] = False
    return mask | small[labels]


def _medians_per_label(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """Median of `values` per label 0 … count-1 in one pass (sorted by label, then value)."""
    flat_labels, flat_values = labels.ravel(), values.ravel()
    sorted_values = flat_values[np.lexsort((flat_values, flat_labels))].astype(np.float64)
    sizes = np.bincount(flat_labels, minlength=count)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    low = np.clip(starts + (sizes - 1) // 2, 0, sorted_values.size - 1)
    high = np.clip(starts + sizes // 2, 0, sorted_values.size - 1)
    return (sorted_values[low] + sorted_values[high]) / 2
