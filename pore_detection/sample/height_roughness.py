"""Embedding medium detection in a false color height image (e.g. Keyence VHX depth composition)
by roughness: the polished metal is smooth, the softer resin around it is rough.

A brightness threshold (imagej, grabcut) does not work on false colors: sample and resin show the
same colors. What differs is how much the height scatters within a few pixels.

1. Relative height from the hue (core/imaging.py, false_color_height), as the filter
   false_color_height does.
2. Roughness: mean absolute deviation of the height from its local mean, both in a window of
   texture_window_px, measured at FULL resolution, then reduced to the working image (long edge
   work_max_px; reducing first would average the scatter away). Each deviation is clipped at
   outlier_clip x its median over the image: the gap to the resin and deep pores are far deeper
   than any roughness and would otherwise be the only "rough" part Otsu finds. Colorless pixels
   (no measurement, overlays) count as rough.
3. One threshold per image: Otsu on the logarithm of the roughness splits smooth from rough. The
   scale of the false colors is stretched per image; a ratio of roughnesses does not change with it.
4. Check: the rough part must be clearly rougher than the smooth part (ratio of their medians at
   least min_roughness_ratio). Otherwise there is no resin in the image (e.g. a cut-out from inside
   the sample): the whole image is sample, logged.
5. Clean-up on the working image: opening then closing of the rough part with smooth_radius_work_px
   (single rough spots such as pores inside the sample go, gaps in the resin close), sample = the
   rest, holes filled (pores and porous zones inside the sample belong to it), only the largest part.
6. Back to full size, shrunk by erode_border_px: the polished edge is rounded and the gap to the
   resin is measured with false heights; that band is no sample.

The gap and the edge zone are rough too, so the mask follows the edge only to about half a texture
window; sample.frame edge_lines then fits straight edges through it. A rough porous zone at the edge
cuts a notch, which edge_lines bridges. Self-contained (imports only core), CPU only.
"""

import logging

import cv2
import numpy as np

from ..core import imaging
from ..core.registry import register
from ..core.step import RoiDetector

log = logging.getLogger("pore_detection")


@register("height_roughness")
class HeightRoughness(RoiDetector):
    """Sample = smooth part of a false color height image, resin = rough part (Otsu on the roughness)."""

    params = {
        "hue_deep_deg": "hue of the deepest color of the scale in degrees (blue: 240)",
        "hue_high_deg": "hue of the highest color in degrees (red: 0), reached through decreasing hue",
        "min_saturation": "saturation 0..1; less saturated pixels carry no height (count as rough)",
        "min_value": "value (brightness) 0..1; darker pixels carry no height (count as rough)",
        "texture_window_px": "window for the local roughness in px at full resolution",
        "outlier_clip": "deviations above this many times their median are clipped (gap, pores; e.g. 5)",
        "work_max_px": "long edge of the working image; all *_work_px values refer to it",
        "min_roughness_ratio": "resin must be at least this many times rougher than the sample "
                               "(medians), otherwise there is no resin",
        "smooth_radius_work_px": "opening / closing radius of the rough part on the working image",
        "erode_border_px": "the sample is shrunk by this many px at full resolution (rounded edge, gap)",
    }

    def detect_roi(self, info, original):
        height, colorless = imaging.false_color_height(original, self.hue_deep_deg, self.hue_high_deg,
                                                       self.min_saturation, self.min_value)
        full_shape = height.shape
        factor = max(1.0, max(full_shape) / self.work_max_px)
        rough = imaging.downscale(_roughness(height, self.texture_window_px, self.outlier_clip), factor)
        no_height = imaging.downscale(colorless, factor)
        del height, colorless

        log_rough = np.log(np.maximum(rough, np.float32(1e-6)))
        valid = log_rough[~no_height]
        if valid.size < 2:
            raise ValueError("no colored pixels: not a false color height image (check hue / saturation settings)")
        resin = (log_rough > _otsu(valid)) | no_height

        if not (~resin).any():
            raise ValueError("everything is rough: no smooth sample found")
        measured_resin = resin & ~no_height
        resin_rough = np.median(rough[measured_resin]) if measured_resin.any() else np.inf
        ratio = resin_rough / max(float(np.median(rough[~resin])), 1e-9)
        if ratio < self.min_roughness_ratio:
            log.info("%s: no embedding medium found (resin only %.2fx rougher than the sample, needs %.2fx)",
                     info.name, ratio, self.min_roughness_ratio)
            return np.ones(full_shape, dtype=bool)
        log.debug("%s: resin %.2fx rougher than the sample", info.name, ratio)

        kernel = _disk(self.smooth_radius_work_px)
        resin = cv2.morphologyEx(resin.view(np.uint8), cv2.MORPH_OPEN, kernel)
        resin = cv2.morphologyEx(resin, cv2.MORPH_CLOSE, kernel) > 0
        sample = _largest_part(_fill_holes(~resin))
        if not sample.any():
            raise ValueError("no sample left after the clean-up")

        sample = _upscale(sample, full_shape)
        if self.erode_border_px > 0:
            sample = cv2.erode(sample.view(np.uint8), _disk(self.erode_border_px)) > 0
        return sample


def _roughness(image: np.ndarray, window_px: int, clip: float) -> np.ndarray:
    """Mean |image - local mean| in a window_px box, each deviation clipped at clip x its median
    (median over every 7th pixel: enough values, far faster on large images)."""
    size = (int(window_px), int(window_px))
    deviation = cv2.boxFilter(image, cv2.CV_32F, size, borderType=cv2.BORDER_REFLECT_101)
    np.subtract(image, deviation, out=deviation)
    np.abs(deviation, out=deviation)
    limit = np.float32(clip * np.median(deviation.ravel()[::7]))
    np.minimum(deviation, limit, out=deviation)
    return cv2.boxFilter(deviation, cv2.CV_32F, size, borderType=cv2.BORDER_REFLECT_101)


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold of float values (256 classes between their 1st and 99th percentile)."""
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return float(high)
    levels = np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8).reshape(1, -1)
    level, _ = cv2.threshold(levels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(low + (level + 0.5) / 255 * (high - low))


def _upscale(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Working mask -> full size; linear interpolation then 0.5 gives smooth instead of blocky edges."""
    big = cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return big >= 0.5


def _disk(radius: float) -> np.ndarray:
    size = 2 * max(1, round(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_part(mask: np.ndarray) -> np.ndarray:
    """Only the largest 8-connected part of the mask (empty mask stays empty)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    if count <= 2:                      # background + at most one part
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Background not 4-connected to the image border becomes foreground."""
    padded = cv2.copyMakeBorder((~mask).view(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(padded, None, (0, 0), 2, flags=4)
    return padded[1:-1, 1:-1] != 2
