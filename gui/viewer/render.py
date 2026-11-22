"""Drawing helpers: base image + overlays, pore outlines along pixel edges."""

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QLineF, QPointF
from PySide6.QtGui import QImage, QPixmap, QTransform

PORE_COLOR = np.array([255, 26, 26], dtype=np.float32)       # RGB
REJECTED_COLOR = np.array([184, 114, 43], dtype=np.float32)  # rejected by a filter
REMOVED_COLOR = np.array([255, 230, 0], dtype=np.float32)    # rejected by hand (style.REMOVED)
ADDED_COLOR = np.array([12, 133, 153], dtype=np.float32)     # counted by hand (restored or drawn)
EMBEDDING_COLOR = np.array([124, 92, 214], dtype=np.float32)
EMBEDDING_ALPHA = 0.45
FRAME_ARROW = 0.25          # length of the axis arrows relative to the smaller side of the sample
FRAME_HEAD = 0.18           # arrow head relative to the arrow
PORE_FILL_ALPHA = 0.3       # tint inside pores at full opacity; the border is drawn opaque
BORDER_KERNEL = np.ones((3, 3), np.uint8)
HEAT_COLORMAP = cv2.COLORMAP_TURBO  # stress field layer, as in the Fatigue_crack_labeler
COLORBAR_SIZE = (150, 10)


@dataclass(frozen=True)
class Layers:
    base: str              # "original" (later: "background")
    pores: bool
    embedding: bool
    rejected: bool
    removed: bool          # rejected by hand
    added: bool            # counted by hand
    opacity: float         # of the pore overlays, 0..1
    fill: bool = True      # light tint inside the pores; False = only the fine border (check window)


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
                _mark(rgb, masks[name], color, layers.opacity, layers.fill)
    return rgb


def _mark(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, opacity: float, fill: bool = True) -> None:
    """Light tint inside (if fill), fine opaque border: the pore itself stays visible."""
    if not mask.any():
        return
    if fill:
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


def frame_preview(points: list[tuple[float, float]], reach: float) -> list[QLineF]:
    """Lines of the frame tool while it is clicked (PLAN.md 7.11), full image px. `points` = clicks
    so far plus the mouse position: 1-2 the long edge (line through both, `reach` px to each side),
    3 the parallel edge, 4-5 the left / right edge (perpendicular, between both long edges).
    Only a preview: the frame itself is computed by the correction session."""
    lines: list[QLineF] = []
    if len(points) < 2:
        return lines
    (x1, y1), (x2, y2) = points[0], points[1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return lines
    dx, dy = dx / norm, dy / norm
    nx, ny = -dy, dx                                    # perpendicular to the long edge

    def long_line(x, y) -> QLineF:
        return QLineF(QPointF(x - reach * dx, y - reach * dy), QPointF(x + reach * dx, y + reach * dy))

    lines.append(long_line(x1, y1))
    if len(points) >= 3:
        x3, y3 = points[2]
        lines.append(long_line(x3, y3))
        for x, y in points[3:5]:
            t1 = (x1 - x) * nx + (y1 - y) * ny          # along the perpendicular to both long edges
            t3 = (x3 - x) * nx + (y3 - y) * ny
            lines.append(QLineF(QPointF(x + t1 * nx, y + t1 * ny), QPointF(x + t3 * nx, y + t3 * ny)))
    return lines


def heatmap(frame, values: np.ndarray) -> tuple[QPixmap, QTransform, float, float]:
    """Stress field layer as the labeler draws it: the map of pore_index (row 0 = outer edge,
    column 0 = left edge, one cell = sample rectangle / map size) as pixmap, its transform into full
    image px, and the range (min, max) the colours stand for."""
    height, width = values.shape
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    length, thickness = frame.length_px, frame.thickness_px
    # pixmap (u, v) -> sample (x = u L/W, z = D - v D/H) -> image
    x0, y0 = frame.to_image(0.0, thickness)
    xu, yu = frame.to_image(length / width, thickness)
    xv, yv = frame.to_image(0.0, thickness - thickness / height)
    transform = QTransform(xu - x0, yu - y0, xv - x0, yv - y0, x0, y0)
    return to_pixmap(colorize(values, low, high)), transform, low, high


def colorize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """RGB uint8 of values between low and high (colour map HEAT_COLORMAP)."""
    span = high - low if high > low else 1.0
    gray = np.nan_to_num(np.clip((values - low) / span * 255, 0, 255), nan=0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, HEAT_COLORMAP), cv2.COLOR_BGR2RGB)


def colorbar() -> QPixmap:
    width, height = COLORBAR_SIZE
    ramp = np.tile(np.linspace(0, 1, width)[None, :], (height, 1))
    return to_pixmap(colorize(ramp, 0.0, 1.0))


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
