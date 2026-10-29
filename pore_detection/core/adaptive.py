"""Helpers for methods that adapt to each image: lengths in px or µm, matrix brightness and noise,
growing pores from seeds, spot measure, values per pore, light direction of shaded-relief images.

Only used by methods (no stage of its own); everything works on NumPy arrays with OpenCV.
"""

import logging
import math

import cv2
import numpy as np

from .imaging import downscale

log = logging.getLogger("pore_detection")


def length_px(step, name: str, ctx) -> float:
    """Length parameter of a step given as <name>_px or <name>_um (exactly one in config.yaml) in px.
    _um needs a scale – core skips the image before if it is missing."""
    if hasattr(step, f"{name}_um"):
        return getattr(step, f"{name}_um") / ctx.pixel_size_um
    return getattr(step, f"{name}_px")


def matrix_background(image: np.ndarray, roi: np.ndarray, window_px: float, work_scale: float) -> np.ndarray:
    """Local brightness of the matrix: grey closing (window larger than the largest pore) on a shrunk
    copy, smoothed, back to full size. Outside the sample the typical matrix value is used, so the
    dark resin does not pull the background down at the sample edge (as background_correction)."""
    small = downscale(image, work_scale)
    sample = downscale(roi, work_scale)
    if not sample.any():
        return np.full_like(image, np.float32(np.median(image)))
    small[~sample] = np.median(small[sample])
    size = max(3, round(window_px / work_scale) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT_101)
    small = cv2.GaussianBlur(small, (0, 0), size / 2, borderType=cv2.BORDER_REFLECT_101)
    height, width = image.shape
    background = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.maximum(background, np.float32(1e-3), out=background)


def matrix_level(image: np.ndarray, roi: np.ndarray) -> float:
    """Typical matrix brightness: median of the sample (pores are a small share of it)."""
    values = image[roi]
    return float(np.median(values)) if values.size else float(np.median(image))


def noise_level(image: np.ndarray, roi: np.ndarray) -> float:
    """Pixel noise (standard deviation, robust): spread of image - median filter 3x3 inside the
    sample. Structure larger than a pixel hardly enters, so it measures the camera noise."""
    detail = image - cv2.medianBlur(image, 3)
    values = detail[roi]
    if not values.size:
        return 0.0
    # 1.4826 x MAD = sigma of a normal distribution; / 0.86 corrects for the median filter
    # taking part of the noise away (measured on white gaussian noise: 0.86)
    return float(1.4826 * np.median(np.abs(values - np.median(values))) / 0.86)


def grow(seeds: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Connected areas of candidates (8-connected) that contain at least one seed pixel."""
    candidates = candidates | seeds
    n, labels = cv2.connectedComponents(candidates.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    keep = np.zeros(n, dtype=bool)
    keep[labels[seeds]] = True
    keep[0] = False
    return keep[labels]


def spot_measure(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Scale-normalised smaller eigenvalue of the Hessian (x sigma², gray units): large only where the
    image is a dark spot (curved up in every direction), small on dark lines and edges."""
    border = cv2.BORDER_REFLECT_101
    smooth = cv2.GaussianBlur(image, (0, 0), sigma_px, borderType=border)
    dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3, borderType=border)
    dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3, borderType=border)
    dxy2 = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3, scale=0.5, borderType=border)  # 2 x dxy
    del smooth
    trace = cv2.add(dxx, dyy)
    difference = cv2.magnitude(cv2.subtract(dxx, dyy), dxy2)
    measure = cv2.subtract(trace, difference)        # 2 x smaller eigenvalue
    measure *= np.float32(sigma_px ** 2 / 2)
    return measure


def group_quantile(groups: np.ndarray, values: np.ndarray, count: int, q: float) -> np.ndarray:
    """q-quantile of values per group id 1..count (NaN for empty groups); groups 0 are ignored."""
    keep = groups > 0
    groups, values = groups[keep], values[keep]
    result = np.full(count + 1, np.nan)
    if not groups.size:
        return result
    order = np.lexsort((values, groups))
    groups, values = groups[order], values[order]
    starts = np.searchsorted(groups, np.arange(count + 1), side="left")
    ends = np.searchsorted(groups, np.arange(count + 1), side="right")
    present = ends > starts
    pick = starts + np.floor(q * (ends - starts - 1)).astype(np.int64)
    result[present] = values[pick[present]]
    return result


def group_mean(groups: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Mean of values per group id 1..count (NaN for empty groups)."""
    total = np.bincount(groups.ravel(), weights=values.ravel(), minlength=count + 1)
    n = np.bincount(groups.ravel(), minlength=count + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / n


def disk(radius_px: float) -> np.ndarray:
    size = 2 * max(1, round(radius_px)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def depth_below_surroundings(depth: np.ndarray, radius_px: float) -> np.ndarray:
    """Depth (large = deep) minus its grey opening with a disk of radius_px, at least 0, in place:
    depth below the surroundings. Pores narrower than the disk stay, tilt, slow waves and bumps -> 0.
    The opening on a copy shrunk to about 8 px radius (full size costs ~10 s on 100 MP)."""
    factor = max(1.0, radius_px / 8)
    small = cv2.morphologyEx(downscale(depth, factor), cv2.MORPH_OPEN, disk(radius_px / factor),
                             borderType=cv2.BORDER_REFLECT_101)
    height, width = depth.shape
    depth -= cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.maximum(depth, 0, out=depth)


def pore_rings(pores, gap_px: float, ring_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Index per counted pore (1..count, in the order of pores.ids) for its own pixels and for a ring
    of matrix around it: from gap_px to gap_px + ring_px outside the pore, without any pore pixels
    (counted or rejected). Where rings of close pores overlap, the pixel goes to one of them."""
    lookup = np.zeros(int(pores.labels.max()) + 1, np.int32)
    lookup[pores.ids] = np.arange(1, pores.count + 1, dtype=np.int32)
    own = lookup[pores.labels]
    # dilating the index (float32 holds ids up to 2^24 exactly) spreads it into the ring
    spread = cv2.dilate(own.astype(np.float32), disk(gap_px + ring_px)).astype(np.int32)
    any_pore = pores.labels > 0
    if pores.rejected_labels is not None:
        any_pore |= pores.rejected_labels > 0
    near = cv2.dilate(any_pore.view(np.uint8), disk(gap_px)) > 0 if gap_px >= 0.5 else any_pore
    spread[near] = 0
    return own, spread


def light_angle(step, ctx, image: np.ndarray, level: float, radius_px: float) -> float:
    """Light direction of a shaded-relief image in degrees (0 = to the right, 90 = up): the base
    step.light_angle_deg, or the estimate (see estimate_light_angle, on a copy shrunk by
    step.estimate_work_scale, offsets up to radius_px) if its certainty reaches step.min_certainty
    and it lies more than step.angle_tolerance_deg off the base. The angle used is logged."""
    estimate, certainty = estimate_light_angle(image, level, radius_px, step.estimate_work_scale)
    base = float(step.light_angle_deg)
    if estimate is None or certainty < step.min_certainty:
        log.info("%s: light direction uncertain (certainty %.2f) -> base %.0f°", ctx.info.name, certainty, base)
        return base
    difference = abs((estimate - base + 180) % 360 - 180)
    if difference <= step.angle_tolerance_deg:
        log.debug("%s: light %.0f° estimated (certainty %.2f) -> base %.0f°", ctx.info.name, estimate, certainty, base)
        return base
    log.warning("%s: light %.0f° estimated (certainty %.2f), %.0f° off the base %.0f° -> estimate used",
                ctx.info.name, estimate, certainty, difference, base)
    return estimate


def estimate_light_angle(image: np.ndarray, level: float, radius_px: float,
                         work_scale: float) -> tuple[float | None, float]:
    """(angle in degrees, certainty 0..1) of the light in a shaded-relief image: a depression has its
    shadow behind and its highlight ahead along the light. On a copy shrunk by work_scale the
    cross-correlation of the dark parts (below level) and the bright parts (above) over offsets up to
    radius_px gives the mean offset shadow -> highlight; certainty = its asymmetry (0 = none, e.g.
    pure noise, 1 = all pairs in one direction). (None, 0) if the image has no contrast at all."""
    small = downscale(image, work_scale)
    dark = np.maximum(level - small, 0)
    bright = np.maximum(small - level, 0)
    radius = max(2, round(radius_px / work_scale))
    # zero padding of at least radius: offsets up to radius do not wrap around
    height = cv2.getOptimalDFTSize(small.shape[0] + radius)
    width = cv2.getOptimalDFTSize(small.shape[1] + radius)
    spectra = []
    for part in (bright, dark):
        padded = np.zeros((height, width), np.float32)
        padded[:small.shape[0], :small.shape[1]] = part
        spectra.append(cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT))
    # corr[v] = sum over p of bright(p + v) x dark(p): large where the highlight lies at v from the shadow
    corr = cv2.idft(cv2.mulSpectrums(spectra[0], spectra[1], 0, conjB=True),
                    flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    offsets = np.arange(-radius, radius + 1)
    window = corr[np.ix_(offsets % height, offsets % width)].astype(np.float64)
    vy, vx = np.meshgrid(offsets, offsets, indexing="ij")
    distance = np.hypot(vx, vy)
    inside = (distance > 0) & (distance <= radius)
    asymmetry = (window - window[::-1, ::-1])[inside]         # corr[v] - corr[-v]
    total = (window + window[::-1, ::-1])[inside].sum()
    if total <= 0:
        return None, 0.0
    mx = (asymmetry * vx[inside] / distance[inside]).sum()
    my = (asymmetry * vy[inside] / distance[inside]).sum()
    return math.degrees(math.atan2(-my, mx)), float(math.hypot(mx, my) / total)  # y points down
