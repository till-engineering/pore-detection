"""Loupe: the selected pore enlarged from the full-resolution original, outline only."""

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .render import outline, to_pixmap

MIN_SIDE_PX = 24            # smallest shown area in image px (tiny pores)


def loupe_box(box: np.ndarray, surround: float, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square (top, bottom, left, right) around a pore box, `surround` x its size, inside the image."""
    top, bottom, left, right = (int(v) for v in box)
    side = max(MIN_SIDE_PX, round(max(bottom - top, right - left) * surround))
    side = min(side, shape[0], shape[1])
    cy, cx = (top + bottom) // 2, (left + right) // 2
    y0 = min(max(0, cy - side // 2), shape[0] - side)
    x0 = min(max(0, cx - side // 2), shape[1] - side)
    return y0, y0 + side, x0, x0 + side


class Loupe(QWidget):
    """Square view: image pixels as sharp squares, pore border as fine lines along pixel edges."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pixmap = None
        self.lines = []
        self.color = QColor(255, 26, 26)
        self.message = ""

    def show_pore(self, crop_rgb: np.ndarray, pore_mask: np.ndarray, color: QColor) -> None:
        self.pixmap = to_pixmap(crop_rgb)
        self.lines = outline(pore_mask)
        self.color = color
        self.message = ""
        self.update()

    def show_message(self, text: str) -> None:
        self.pixmap, self.lines, self.message = None, [], text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        side = min(self.width(), self.height())
        target = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        painter.fillRect(target, self.palette().color(self.backgroundRole()).darker(115))
        if self.pixmap is None:
            painter.drawText(target.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.message)
            return
        painter.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))   # no smoothing: sharp pixels
        scale = side / self.pixmap.width()
        painter.translate(target.topLeft())
        painter.scale(scale, scale)
        pen = QPen(self.color, 1.2)
        pen.setCosmetic(True)               # fine line at every magnification
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawLines(self.lines)
