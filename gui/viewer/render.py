"""Drawing helpers: base image + overlays, pore outlines along pixel edges."""

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QLineF, QPointF
from PySide6.QtGui import QImage, QPixmap

PORE_COLOR = np.array([255, 26, 26], dtype=np.float32)       # RGB
REJECTED_COLOR = np.array([184, 114, 43], dtype=np.float32)  # rejected by a filter
REMOVED_COLOR = np.array([214, 51, 112], dtype=np.float32)   # rejected by hand
ADDED_COLOR = np.array([12, 133, 153], dtype=np.float32)     # counted by hand (restored or drawn)
EMBEDDING_COLOR = np.array([124, 92, 214], dtype=np.float32)
EMBEDDING_ALPHA = 0.45
FRAME_ARROW = 0.25          # length of the axis arrows relative to the smaller side of the sample
FRAME_HEAD = 0.18           # arrow head relative to the arrow
PORE_FILL_ALPHA = 0.3       # tint inside pores at full opacity; the border is drawn opaque
BORDER_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class Layers:
    base: str              # "original" (later: "background")
    pores: bool
    embedding: bool
    rejected: bool
    removed: bool          # rejected by hand
    added: bool            # counted by hand
    opacity: float         # of the pore overlays, 0..1


def compose(base: np.ndarray, masks: dict[str, np.ndarray], sample: np.ndarray, layers: Layers) -> np.ndarray:
    """RGB uint8 image with the selected overlays. base: RGB uint8; masks: pores, rejected,
    added, removed (bool, size of base)."""
    rgb = base.copy()
    if layers.embedding:
        _tint(rgb, ~sample, EMBEDDING_COLOR, EMBEDDING_ALPHA)
    if layers.opacity > 0:
        for name, color in (("rejected", REJECTED_COLOR), ("removed", REMOVED_COLOR),
                            ("pores", PORE_COLOR), ("added", ADDED_COLOR)):
            if getattr(layers, name):
                _mark(rgb, masks[name], color, layers.opacity)
    return rgb


def _mark(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, opacity: float) -> None:
    """Light tint inside, fine opaque border: the pore itself stays visible."""
    if not mask.any():
        return
    _tint(rgb, mask, color, PORE_FILL_ALPHA * opacity)
    border = mask & ~cv2.erode(mask.view(np.uint8), BORDER_KERNEL).view(bool)
    _tint(rgb, border, color, opacity)


def _tint(rgb: np.ndarray, where: np.ndarray, color: np.ndarray, alpha: float) -> None:
    rgb[where] = (rgb[where] * (1 - alpha) + color * alpha).astype(np.uint8)


def frame_graphics(frame) -> tuple[list[QLineF], list[QLineF], list[tuple[QPointF, str]]]:
    """Sample frame as vector graphics in full image px: (axes with arrow heads, outline of the
    frame rectangle, text labels with their anchor point)."""
    def point(x: float, z: float) -> QPointF:
        return QPointF(*frame.to_image(x, z))

    length, thickness = frame.length_px, frame.thickness_px
    arrow = FRAME_ARROW * min(length, thickness)
    head = FRAME_HEAD * arrow
    axes = [QLineF(point(0, 0), point(arrow, 0)), QLineF(point(0, 0), point(0, arrow)),
            QLineF(point(arrow, 0), point(arrow - head, head / 2)),
            QLineF(point(arrow, 0), point(arrow - head, -head / 2)),
            QLineF(point(0, arrow), point(head / 2, arrow - head)),
            QLineF(point(0, arrow), point(-head / 2, arrow - head))]
    corners = [point(0, 0), point(length, 0), point(length, thickness), point(0, thickness)]
    outline_lines = [QLineF(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    labels = [(point(arrow + head, 0), "x"), (point(0, arrow + head), "z"),
              (point(length / 2, thickness), "outer"), (point(length / 2, 0), "inner")]
    return axes, outline_lines, labels


def shrink(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to (width, height); bool masks: any covered pixel counts (small pores stay visible)."""
    if image.shape[1] == size[0] and image.shape[0] == size[1]:
        return image
    if image.dtype == bool:
        return cv2.resize(image.view(np.uint8) * np.uint8(255), size, interpolation=cv2.INTER_AREA) > 0
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy())


def outline(mask: np.ndarray, dx: float = 0, dy: float = 0) -> list[QLineF]:
    """Line segments along the pixel edges between True and False (the exact pore border).

    Coordinates in pixels of `mask`, shifted by (dx, dy).
    """
    padded = np.pad(mask, 1)
    lines = []
    ys, xs = np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])        # horizontal edges at y
    lines += [QLineF(x + dx, y + dy, x + 1 + dx, y + dy) for y, x in zip(ys.tolist(), xs.tolist())]
    ys, xs = np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])        # vertical edges at x
    lines += [QLineF(x + dx, y + dy, x + dx, y + 1 + dy) for y, x in zip(ys.tolist(), xs.tolist())]
    return lines
