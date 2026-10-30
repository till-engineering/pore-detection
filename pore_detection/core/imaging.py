"""Image helpers for core and methods."""

import cv2
import numpy as np


def to_gray(original: np.ndarray, weights, out: np.ndarray | None = None) -> np.ndarray:
    """RGB (H, W, 3) uint8/uint16 -> gray (H, W) float32 0..1 with R, G, B weights.

    `out`: optional (H, W) float32 array to write into (e.g. one band of a larger image).
    """
    scale = 1.0 / np.iinfo(original.dtype).max
    gray = np.multiply(original[..., 0], np.float32(weights[0] * scale), dtype=np.float32, out=out)
    for channel in (1, 2):
        gray += np.multiply(original[..., channel], np.float32(weights[channel] * scale), dtype=np.float32)
    return gray


def false_color_height(original: np.ndarray, hue_deep_deg: float, hue_high_deg: float,
                       min_saturation: float, min_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative height 0 (deep) .. 1 (high) of a height image in false colors, from the hue alone.

    The scale runs from hue_deep_deg to hue_high_deg through decreasing hue (240 blue -> 120 green
    -> 0 red); hues outside that arc (magenta, purple) go to the nearer end. Returns (height, colorless):
    colorless = saturation below min_saturation or value below min_value (gray, white, black: no
    height); their height is meaningless, the caller decides what they get.
    """
    rgb = original.astype(np.float32) * np.float32(1 / np.iinfo(original.dtype).max)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)             # float input: hue 0..360, s and v 0..1
    del rgb
    hue, saturation, value = cv2.split(hsv)
    del hsv
    arc = (hue_deep_deg - hue_high_deg) % 360
    if arc == 0:
        raise ValueError("hue_deep_deg and hue_high_deg must differ")
    # degrees from the deep end towards the high end (decreasing hue), 0..360
    along = np.subtract(np.float32(hue_deep_deg), hue, out=hue)
    np.mod(along, np.float32(360), out=along)
    # outside the arc: the nearer end (beyond high by along - arc, before deep by 360 - along)
    beyond = along > arc
    along[beyond & (along - arc <= 360 - along)] = arc
    along[beyond & (along - arc > 360 - along)] = 0
    height = np.multiply(along, np.float32(1 / arc), out=along)
    colorless = (saturation < min_saturation) | (value < min_value)
    return height, colorless


def downscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Shrink an image by `factor` (e.g. 8 -> 1/8 edge length), area averaging.

    Works for gray, color and bool images; bool masks are averaged and thresholded at 0.5.
    """
    height, width = image.shape[:2]
    size = (max(1, round(width / factor)), max(1, round(height / factor)))  # cv2: (width, height)
    if image.dtype == bool:
        small = cv2.resize(image.view(np.uint8) * np.uint8(255), size, interpolation=cv2.INTER_AREA)
        return small >= 128
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
