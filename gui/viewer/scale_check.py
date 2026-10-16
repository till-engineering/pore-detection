"""Optical check of the scale: the scale overlay from the original, the measured bar drawn over it.

The marker spans exactly the bar length the pipeline measured (end ticks at its first and last
pixel column), so a bar measured too short or too long is visible at a glance.
"""

import numpy as np
from PySide6.QtCore import QLineF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .render import to_pixmap

MARGIN_PX = 12              # image px shown around the scale overlay
TICK_SCREEN_PX = 7          # half height of the end ticks on screen


def scale_crop_box(box, bar, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) shown in the check: overlay box (else bar) plus a margin."""
    top, bottom, left, right = box if box is not None else bar
    return (max(0, top - MARGIN_PX), min(shape[0], bottom + MARGIN_PX),
            max(0, left - MARGIN_PX), min(shape[1], right + MARGIN_PX))


class ScaleCheck(QWidget):
    """Crop of the scale overlay, scaled to the widget width, with the measured bar marked."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.pixmap = None
        self.bar = None             # (top, bottom, left, right) in crop px
        self.color = QColor(0, 200, 255)
        self.message = ""

    def show_scale(self, crop_rgb: np.ndarray, bar: tuple[int, int, int, int] | None, color: QColor) -> None:
        self.pixmap, self.bar, self.color, self.message = to_pixmap(crop_rgb), bar, color, ""
        self.updateGeometry()
        self.update()

    def show_message(self, text: str) -> None:
        self.pixmap, self.bar, self.message = None, None, text
        self.updateGeometry()
        self.update()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if self.pixmap is None:
            return 60
        return max(40, round(width * self.pixmap.height() / self.pixmap.width()))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self.pixmap is None:
            painter.drawText(QRectF(self.rect()).adjusted(4, 4, -4, -4),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
            return
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        width, height = self.pixmap.width() * scale, self.pixmap.height() * scale
        target = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        if self.bar is None:
            return
        top, bottom, left, right = self.bar
        y = target.top() + (top + bottom) / 2 * scale
        x0, x1 = target.left() + left * scale, target.left() + right * scale
        pen = QPen(self.color, 2)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        tick = max(TICK_SCREEN_PX, (bottom - top) * scale)
        painter.drawLines([QLineF(x0, y, x1, y), QLineF(x0, y - tick, x0, y + tick), QLineF(x1, y - tick, x1, y + tick)])
