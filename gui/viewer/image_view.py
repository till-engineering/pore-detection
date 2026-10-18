"""Zoomable image: overview for the whole image, full resolution for the visible part when zoomed in.

Scene coordinates are pixels of the full image. The overview is scaled up to that size; once the
view shows more than DETAIL_ZOOM screen px per image px, the visible part is drawn again from the
full-resolution data (`detail_source`), only as large as the screen needs.
"""

import math
from typing import Callable

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView)

from .render import to_pixmap

DETAIL_ZOOM = 0.35          # screen px per image px from which the full resolution is drawn
ZOOM_STEP = 1.25

# (top, bottom, left, right, out_width, out_height) -> RGB uint8 of size out
DetailSource = Callable[[int, int, int, int, int, int], np.ndarray]


class ImageView(QGraphicsView):
    zoom_changed = Signal(float)            # screen px per full-resolution px
    clicked = Signal(int, int)              # full-resolution x, y
    hovered = Signal(int, int)              # mouse over the image (no button pressed): full-resolution x, y
    hover_left = Signal()                   # mouse left the image
    outlined = Signal(list)                 # outline drawn in mode "outline": [(x, y), …] full-resolution px

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.base = QGraphicsPixmapItem()
        self.base.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.detail = QGraphicsPixmapItem()
        self.detail.setZValue(1)
        self.selection = QGraphicsPathItem()
        self.selection.setZValue(2)
        self.hover = QGraphicsPathItem()        # outline of the pore a click would pick
        self.hover.setZValue(1.8)
        self.drawing = QGraphicsPathItem()      # outline while it is drawn
        self.drawing.setZValue(3)
        self.frame_axes = QGraphicsPathItem()   # sample frame: axes (solid) and outline (dashed)
        self.frame_outline = QGraphicsPathItem()
        self.frame_labels: list[QGraphicsSimpleTextItem] = []
        for item in (self.frame_axes, self.frame_outline):
            item.setZValue(1.5)                 # above the image, below the selection
            self.scene().addItem(item)
        self.mode = "select"                    # select | click | outline
        self.points: list[QPointF] = []
        for item in (self.base, self.detail, self.selection, self.hover, self.drawing):
            self.scene().addItem(item)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._pointer = False                   # a pore under the mouse could be picked
        self._hover_xy: tuple[int, int] | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(0, 0, 0, 0))
        self.full_size = (1, 1)             # (width, height)
        self.detail_source: DetailSource | None = None
        self.fitted = True
        self._press: QPointF | None = None
        self._timer = QTimer(self, singleShot=True, interval=40, timeout=self._update_detail)
        self.horizontalScrollBar().valueChanged.connect(self._schedule)
        self.verticalScrollBar().valueChanged.connect(self._schedule)

    # --- content ---

    def set_image(self, overview_rgb: np.ndarray, full_size: tuple[int, int], fit: bool) -> None:
        """Overview image, shown stretched to the full image size (width, height)."""
        self.full_size = full_size
        self.base.setPixmap(to_pixmap(overview_rgb))
        self.base.setScale(full_size[0] / overview_rgb.shape[1])
        self.scene().setSceneRect(QRectF(0, 0, *full_size))
        if fit:
            self.fit()
        self.refresh_detail()

    def clear(self) -> None:
        """Nothing shown (while an image is loading)."""
        self.base.setPixmap(QPixmap())
        self.set_selection([], QColor())
        self.set_frame([], [], [], QColor())
        self.set_detail_source(None)

    def set_detail_source(self, source: DetailSource | None) -> None:
        self.detail_source = source
        self.refresh_detail()

    def refresh_detail(self) -> None:
        self.detail.setVisible(False)
        self._schedule()

    def set_selection(self, lines, color: QColor) -> None:
        """Outline of the selected pore (QLineF list in full-resolution px), [] = none."""
        path = QPainterPath()
        for line in lines:
            path.moveTo(line.p1())
            path.lineTo(line.p2())
        pen = QPen(color, 2)
        pen.setCosmetic(True)               # 2 screen px at every zoom
        self.selection.setPen(pen)
        self.selection.setPath(path)

    def set_frame(self, axes: list[QLineF], outline: list[QLineF], labels: list[tuple[QPointF, str]],
                  color: QColor) -> None:
        """Sample frame drawn over the image (full-resolution px); empty lists = none. Lines and
        text keep their screen size at every zoom."""
        for item, lines, pen_style in ((self.frame_axes, axes, Qt.PenStyle.SolidLine),
                                       (self.frame_outline, outline, Qt.PenStyle.DashLine)):
            path = QPainterPath()
            for line in lines:
                path.moveTo(line.p1())
                path.lineTo(line.p2())
            pen = QPen(color, 2 if pen_style == Qt.PenStyle.SolidLine else 1.5, pen_style)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setPath(path)
        for label in self.frame_labels:
            self.scene().removeItem(label)
        self.frame_labels = []
        font = QFont()
        font.setBold(True)
        for anchor, text in labels:
            label = QGraphicsSimpleTextItem(text)
            label.setFont(font)
            label.setBrush(color)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(anchor)
            label.setZValue(1.5)
            self.scene().addItem(label)
            self.frame_labels.append(label)

    def set_hover(self, lines, color: QColor) -> None:
        """Outline of the pore a click would pick ([] = none); the cursor shows that it can be picked."""
        path = QPainterPath()
        for line in lines:
            path.moveTo(line.p1())
            path.lineTo(line.p2())
        pen = QPen(color, 5)                    # wide soft glow: would be picked (selected = thin line)
        pen.setCosmetic(True)
        self.hover.setPen(pen)
        self.hover.setPath(path)
        self._pointer = bool(lines)
        self._update_cursor()

    def _update_cursor(self) -> None:
        if self.mode == "select":
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor if self._pointer
                                      else Qt.CursorShape.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def set_mode(self, mode: str, color: QColor | None = None) -> None:
        """select: drag pans, click selects · click: click for a tool · outline: drag draws an outline."""
        self.mode = mode
        self.points = []
        self.drawing.setPath(QPainterPath())
        pen = QPen(color or QColor(0, 0, 0), 2)
        pen.setCosmetic(True)
        self.drawing.setPen(pen)
        fill = QColor(color or QColor(0, 0, 0))
        fill.setAlpha(50)
        self.drawing.setBrush(fill)
        self.setDragMode(QGraphicsView.DragMode.NoDrag if mode == "outline"
                         else QGraphicsView.DragMode.ScrollHandDrag)
        self.set_hover([], QColor())

    # --- zoom ---

    def zoom(self) -> float:
        return self.transform().m11()

    def fit(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.fitted = True
        self._zoomed()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fitted:
            self.fit()

    def wheelEvent(self, event) -> None:
        self.zoom_by(ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP)

    def zoom_by(self, factor: float) -> None:
        """Zoom around the mouse (wheel) or the view centre (buttons)."""
        if self.zoom() * factor > 40:       # 40 screen px per image px is enough
            return
        self.scale(factor, factor)
        self.fitted = False
        self._zoomed()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def _zoomed(self) -> None:
        self.zoom_changed.emit(self.zoom())
        self.refresh_detail()

    # --- click (press + release without moving) ---

    def mousePressEvent(self, event) -> None:
        self._press = event.position()
        if self.mode == "outline" and event.button() == Qt.MouseButton.LeftButton:
            self.points = [self.mapToScene(event.position().toPoint())]
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() == Qt.MouseButton.NoButton and self.mode != "outline":
            point = self.mapToScene(event.position().toPoint())
            x, y = int(point.x()), int(point.y())
            if 0 <= x < self.full_size[0] and 0 <= y < self.full_size[1]:
                if (x, y) != self._hover_xy:
                    self._hover_xy = (x, y)
                    self.hovered.emit(x, y)
            elif self._hover_xy is not None:
                self._hover_xy = None
                self.hover_left.emit()
        if self.mode == "outline" and self.points:
            self.points.append(self.mapToScene(event.position().toPoint()))
            path = QPainterPath(self.points[0])
            for point in self.points[1:]:
                path.lineTo(point)
            path.closeSubpath()
            self.drawing.setPath(path)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.mode == "outline" and self.points:
            points = [(p.x(), p.y()) for p in self.points]
            self.points = []
            self.drawing.setPath(QPainterPath())
            if len(points) >= 3:
                self.outlined.emit(points)
            self._press = None
            return
        super().mouseReleaseEvent(event)
        self._update_cursor()                   # dragging set the hand cursor
        if self._press is not None and (event.position() - self._press).manhattanLength() < 4 \
                and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            x, y = int(point.x()), int(point.y())
            if 0 <= x < self.full_size[0] and 0 <= y < self.full_size[1]:
                self.clicked.emit(x, y)
        self._press = None

    def leaveEvent(self, event) -> None:
        self._hover_xy = None
        self.hover_left.emit()
        super().leaveEvent(event)

    # --- full resolution of the visible part ---

    def _schedule(self) -> None:
        self._timer.start()

    def _update_detail(self) -> None:
        zoom = self.zoom()
        if self.detail_source is None or zoom < DETAIL_ZOOM:
            self.detail.setVisible(False)
            return
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        left, top = max(0, math.floor(visible.left())), max(0, math.floor(visible.top()))
        right = min(self.full_size[0], math.ceil(visible.right()) + 1)
        bottom = min(self.full_size[1], math.ceil(visible.bottom()) + 1)
        if right <= left or bottom <= top:
            return
        out_w = min(right - left, math.ceil((right - left) * zoom))   # never more px than the screen needs
        out_h = min(bottom - top, math.ceil((bottom - top) * zoom))
        rgb = self.detail_source(top, bottom, left, right, out_w, out_h)
        self.detail.setPixmap(to_pixmap(rgb))
        self.detail.setPos(left, top)
        self.detail.setTransform(QTransform.fromScale((right - left) / out_w, (bottom - top) / out_h))
        # zoomed far in: show image pixels as sharp squares
        self.detail.setTransformationMode(Qt.TransformationMode.FastTransformation if zoom > 1
                                          else Qt.TransformationMode.SmoothTransformation)
        self.detail.setVisible(True)
