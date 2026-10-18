"""One place of the plot grid: shows a plot image (PNG from the results) over the full width.

The height follows the plot's aspect ratio (no white bars at the sides). The image is scaled
down once per size with Qt's smooth scaling (area averaging) – drawing it scaled with the
painter would only interpolate between neighbouring pixels and make text grainy.
"""

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

MESSAGE_HEIGHT_PX = 120     # height while there is no plot to show
MARGIN_PX = 10              # space between the plot and the card border


class PlotView(QWidget):
    def __init__(self):
        super().__init__()
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.pixmap: QPixmap | None = None
        self.scaled: QPixmap | None = None      # pixmap scaled to the current size (cache)
        self.message = ""

    def show_png(self, png: bytes) -> None:
        pixmap = QPixmap()
        pixmap.loadFromData(png, "PNG")
        self.pixmap, self.scaled, self.message = pixmap, None, ""
        self.updateGeometry()
        self.update()

    def show_message(self, text: str) -> None:
        self.pixmap, self.scaled, self.message = None, None, text
        self.updateGeometry()
        self.update()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        if self.pixmap is None or self.pixmap.isNull():
            return MESSAGE_HEIGHT_PX
        inner = max(1, width - 2 * MARGIN_PX)
        return round(inner * self.pixmap.height() / self.pixmap.width()) + 2 * MARGIN_PX

    def sizeHint(self) -> QSize:
        return QSize(400, self.heightForWidth(400))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        if self.pixmap is None or self.pixmap.isNull():
            painter.drawText(QRectF(self.rect()).adjusted(6, 6, -6, -6),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
            return
        ratio = self.devicePixelRatioF()
        target = (self.size() - QSize(2 * MARGIN_PX, 2 * MARGIN_PX)) * ratio
        if self.scaled is None or self.scaled.size() != self.pixmap.size().scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio):
            self.scaled = self.pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
            self.scaled.setDevicePixelRatio(ratio)
        size = self.scaled.deviceIndependentSize()
        painter.drawPixmap(round((self.width() - size.width()) / 2), round((self.height() - size.height()) / 2),
                           self.scaled)
