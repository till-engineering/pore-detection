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
