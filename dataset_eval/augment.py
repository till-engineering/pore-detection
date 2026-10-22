"""Augmentation of training patches for the U-Net (dataset_eval/train_unet.py): only in training.

Works on one patch: image (H, W, 3) float32 RGB 0..1 (before the normalization) and mask (H, W) bool.
Geometric changes act on both, all others on the image only. No scaling and no distortion: the
network sees the pores in their real size in px, as in the pipeline run.

Settings (train_config.yaml, augment:), each one 0 / null / false = off:
    flip_rotate     random rotation by 0/90/180/270 degrees and mirroring (8 variants)
    probability     each of the following is applied to a patch with this probability
    brightness      +- share of the brightness (0.2 = x 0.8 ... 1.2)
    contrast        +- share of the contrast around the patch mean
    gamma           [min, max] exponent (other camera curve)
    gradient        max. brightness change across the patch (uneven illumination), e.g. 0.3 = +-15 %
    hue             +- shift of the hue (share of the colour circle, 0.03 = +- 11 degrees)
    saturation      +- share of the saturation
    blur_sigma_px   [min, max] gaussian blur in px (focus not perfect)
    noise           max. standard deviation of gaussian noise (gray 0..1)
"""

import cv2
import numpy as np


def apply(image: np.ndarray, mask: np.ndarray, settings: dict, rng: np.random.Generator):
    if settings.get("flip_rotate"):
        turns = int(rng.integers(4))
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)

    p = settings.get("probability", 0.5)
    hue, saturation = settings.get("hue") or 0, settings.get("saturation") or 0
    if (hue or saturation) and rng.random() < p:
        image = _hue_saturation(image, rng.uniform(-hue, hue), 1 + rng.uniform(-saturation, saturation))
    if settings.get("brightness") and rng.random() < p:
        image = image * np.float32(1 + rng.uniform(-settings["brightness"], settings["brightness"]))
    if settings.get("contrast") and rng.random() < p:
        mean = image.mean()
        image = (image - mean) * np.float32(1 + rng.uniform(-settings["contrast"], settings["contrast"])) + mean
    if settings.get("gradient") and rng.random() < p:
        image = image * _gradient(image.shape[:2], settings["gradient"], rng)[..., None]
    if settings.get("gamma") and rng.random() < p:
        image = np.power(np.clip(image, 0, 1), np.float32(rng.uniform(*settings["gamma"])))
    if settings.get("blur_sigma_px") and rng.random() < p:
        sigma = rng.uniform(*settings["blur_sigma_px"])
        if sigma > 0.1:
            image = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    if settings.get("noise") and rng.random() < p:
        image = image + rng.normal(0, rng.uniform(0, settings["noise"]), image.shape).astype(np.float32)
    return np.clip(image, 0, 1).astype(np.float32, copy=False), mask


def _hue_saturation(image: np.ndarray, shift: float, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2HSV)   # float: H 0..360, S and V 0..1
    hsv[..., 0] = (hsv[..., 0] + shift * 360) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _gradient(shape: tuple[int, int], strength: float, rng: np.random.Generator) -> np.ndarray:
    """Linear brightness ramp in a random direction: 1 - strength/2 ... 1 + strength/2."""
    height, width = shape
    angle = rng.uniform(0, 2 * np.pi)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (x - width / 2) * np.cos(angle) + (y - height / 2) * np.sin(angle)
    ramp /= max(np.abs(ramp).max(), 1)
    return (1 + ramp * np.float32(strength * rng.uniform(0, 1) / 2)).astype(np.float32)
