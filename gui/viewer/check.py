"""Check window of the viewer (docs/plan_check_window.md): go through one image in patches of a fixed
size and correct right there.

Modes:
    score   the counted pore with the lowest U-Net score (pores.json unet_score) that is not checked yet
            in the centre; every counted pore lying completely inside the patch counts as checked
    tiles   the image in tiles of the patch size, overlapping, serpentine; tiles without sample are skipped

Modal. Corrections go through a correction session of its own (same pore_detection.Corrections as the
viewer), not live into the viewer: after closing, the viewer loads the image again. What is checked is
kept only while the window is open, separately per mode. No zoom: the patch is always shown whole;
at the image border it is filled with gray, so the pore stays in the centre.

Keys: → / Space next, ← back, Backspace rejects the centre pore (score mode), Delete rejects every
counted pore lying completely inside the patch, Ctrl+Z undo, Esc tool off.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QGraphicsPathItem, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout)

from . import style
from .data import Image
from .icons import ICON_PX
from .image_view import ImageView
from .render import outline, to_pixmap

SCORE_COLUMN = "unet_score"     # Detector.score_column of the U-Net
PAD_GRAY = 128                  # patch area outside the image
OVERVIEW_PX = 260               # long edge of the overview
PICK_RADIUS_SCREEN_PX = 12      # "+" / "−": a click this close to a pore (on screen) picks it
VISITED_COLOR = QColor(40, 190, 70)     # overview: patches already looked at
OPACITY = 100                   # start value of the outline opacity in % (thin vector lines: full by default)
OUTLINE_PX = 1.0                # outlines of the other pores, screen px at every zoom
CENTRE_COLOR = "#39ff14"        # outline of the centre pore (score mode): neon green, as thin as the others
HOVER_ALPHA = 110               # glow of the pore a click would hit (as in the viewer)
OTHER_ALPHA = 0.7               # the other pores a little paler than the centre pore
# correction tools: key -> (icon, tooltip, colour of the drawn outline, view mode)
TOOLS = {
    "restore": ("circle-plus", "count a rejected pore (click on it)", "#0c8599", "click"),
    "remove": ("circle-minus", "reject a counted pore (click on it); a drawn pore is deleted", style.REMOVED, "click"),
    "draw": ("pencil", "draw a missed pore (hold the mouse button and trace its outline)", None, "outline"),
    "area": ("square-dashed", "reject every counted pore lying completely inside the traced area", style.REMOVED, "outline"),
}
MODES = (("score", "By U-Net score"), ("tiles", "Tiles"))
KEYS = ("→ / Space  next\n← back\nBackspace  reject the centre pore\nDelete  reject all pores in the patch\n"
        "Ctrl+Z  undo\nEsc  tool off")


@dataclass
class Boxes:
    """Bounding boxes of the counted pores (bottom / right exclusive), full image px."""

    ids: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @classmethod
    def of(cls, labels: np.ndarray) -> "Boxes":
        """One pass over the pore pixels only (a few % of the image)."""
        flat = np.flatnonzero(labels)
        if not len(flat):
            empty = np.zeros(0, dtype=np.int64)
            return cls(empty, empty, empty, empty, empty)
        ids = labels.ravel()[flat]
        order = np.argsort(ids, kind="stable")
        ids, flat = ids[order], flat[order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ys, xs = np.divmod(flat, labels.shape[1])
        return cls(ids[starts].astype(np.int64), np.minimum.reduceat(ys, starts), np.maximum.reduceat(ys, starts) + 1,
                   np.minimum.reduceat(xs, starts), np.maximum.reduceat(xs, starts) + 1)

    def inside(self, top: int, left: int, size: int) -> np.ndarray:
        """Ids of the pores lying completely inside the square patch."""
        within = ((self.top >= top) & (self.bottom <= top + size) & (self.left >= left)
                  & (self.right <= left + size))
        return self.ids[within]

    def box(self, pore_id: int) -> tuple[int, int, int, int] | None:
        i = np.searchsorted(self.ids, pore_id)
        if i < len(self.ids) and self.ids[i] == pore_id:
            return int(self.top[i]), int(self.bottom[i]), int(self.left[i]), int(self.right[i])
        return None


@dataclass
class _ScoreMode:
    order: list[int]                                    # pores with a score, lowest score first
    scores: dict[int, float]
    checked: set[int] = field(default_factory=set)
    history: list[tuple[int, int, int]] = field(default_factory=list)   # (top, left, centre pore)
    pos: int = -1


@dataclass
class _TileMode:
    tiles: list[tuple[int, int]]                        # (top, left) in serpentine order
    pos: int = -1
    reached: int = -1                                   # furthest tile shown so far (green in the overview)


class _Signals(QObject):
    corrected = Signal(object)          # Future[(State, Boxes)] (emitted from the correction thread)


class PatchView(ImageView):
    """The image view of the viewer without zoom and panning: the patch is always shown whole."""

    def __init__(self):
        super().__init__()
        self.setDragMode(ImageView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.outlines: dict[str, QGraphicsPathItem] = {}     # vector outlines over the patch (not burnt in)
        for z, key in enumerate(("rejected", "added", "pores", "centre")):
            item = QGraphicsPathItem()
            item.setZValue(1.4 + 0.01 * z)      # above the image, below hover, selection and drawing
            self.scene().addItem(item)
            self.outlines[key] = item

    def set_outline(self, key: str, mask: np.ndarray, color: str, width: float, alpha: float = 1.0) -> None:
        """Outline of every pore in `mask` along the pixel edges (patch px), `width` screen px."""
        path = QPainterPath()
        for line in outline(mask):
            path.moveTo(line.p1())
            path.lineTo(line.p2())
        pen_color = QColor(color)
        pen_color.setAlphaF(alpha)
        pen = QPen(pen_color, width)
        pen.setCosmetic(True)
        self.outlines[key].setPen(pen)
        self.outlines[key].setPath(path)

    def set_outline_opacity(self, opacity: float) -> None:
        for item in self.outlines.values():
            item.setOpacity(opacity)

    def set_mode(self, mode: str, color: QColor | None = None) -> None:
        super().set_mode(mode, color)
        self.setDragMode(ImageView.DragMode.NoDrag)

    def wheelEvent(self, event) -> None:
        event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:
        event.ignore()


def tile_starts(length: int, size: int, overlap: float) -> list[int]:
    """Starts of the tiles along one axis: step size * (1 - overlap), the last tile ends at the border."""
    if length <= size:
        return [0]
    step = max(1, int(round(size * (1 - overlap))))
    starts = list(range(0, length - size + 1, step))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


class CheckWindow(QDialog):
    def __init__(self, image: Image, corrections, settings: dict, accent: str, parent=None,
                 colors: dict | None = None):
        """image: the image the viewer shows (not changed here); corrections: pore_detection.Corrections
        of its ID or None (read-only); settings: config.gui (check_patch_px, check_tile_overlap)."""
        super().__init__(parent)
        self.setWindowTitle(f"Check – {image.name}")
        self.setModal(True)
        self.resize(1400, 980)
        self.image, self.store, self.accent = image, corrections, QColor(accent)
        self.colors = colors or style.LIGHT  # theme of the viewer (icon colours)
        self.size_px = int(settings.get("check_patch_px", 1000))
        overlap = float(settings.get("check_tile_overlap", 0.1))
        self.labels, self.rejected_labels = image.labels, image.rejected_labels
        self.added, self.removed = image.added, image.removed
        self.boxes = Boxes.of(self.labels)
        self.changed = False                # a correction was made: the viewer loads the image again
        self.row: dict | None = None        # images.json row after the last correction
        self.session = None                 # correction session (opened on first use, in the worker)
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="check")
        self.signals = _Signals()
        self.signals.corrected.connect(self._on_corrected)
        self.tool: str | None = None
        self.top = self.left = 0            # shown patch
        self.centre = 0                     # centre pore (score mode), 0 = none

        table = image.table
        if SCORE_COLUMN in table.columns:
            scored = table[table[SCORE_COLUMN].notna()].sort_values(SCORE_COLUMN, kind="stable")
            self.score_mode = _ScoreMode([int(i) for i in scored["pore_id"]],
                                         dict(zip(scored["pore_id"].astype(int), scored[SCORE_COLUMN].astype(float))))
        else:
            self.score_mode = None
        height, width = image.shape
        rows, columns = tile_starts(height, self.size_px, overlap), tile_starts(width, self.size_px, overlap)
        tiles = []
        for r, top in enumerate(rows):
            for left in (columns if r % 2 == 0 else columns[::-1]):     # serpentine
                if image.roi[top:top + self.size_px, left:left + self.size_px].any():
                    tiles.append((top, left))
        self.tile_mode = _TileMode(tiles)

        self._build()
        self.mode = "score" if self.score_mode is not None else "tiles"
        self.mode_box.setCurrentIndex(0 if self.mode == "score" else 1)
        self._next()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        head = QHBoxLayout()
        self.mode_box = QComboBox()
        for _key, text in MODES:
            self.mode_box.addItem(text)
        if self.score_mode is None:         # no U-Net score in this run: only tiles
            self.mode_box.model().item(0).setEnabled(False)
            self.mode_box.setToolTip("By U-Net score: no unet_score in this run (not a U-Net run, or an older one)")
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.mode_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        head.addWidget(self.mode_box)
        head.addSpacing(12)
        self.progress = QLabel(objectName="imageName")
        head.addWidget(self.progress)
        head.addSpacing(18)
        self.centre_label = QLabel(objectName="imageName")
        head.addWidget(self.centre_label)
        head.addStretch()
        self.finished = QLabel(objectName="imageName")
        self.finished.setStyleSheet(f"color: {style.ADDED};")
        head.addWidget(self.finished)
        layout.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_tools(), alignment=Qt.AlignmentFlag.AlignTop)
        self.view = PatchView()
        self.view.clicked.connect(self._on_clicked)
        self.view.outlined.connect(self._on_outlined)
        self.view.hovered.connect(self._on_hover)
        self.view.hover_left.connect(self._clear_hover)
        self.hover_key: tuple[int, bool] | None = None      # (pore id, rejected) glowing under the mouse
        body.addWidget(self.view, stretch=1)

        side = QVBoxLayout()
        side.setSpacing(10)
        side.addWidget(QLabel("WHERE IN THE IMAGE", objectName="sectionLabel"))
        self.overview = QLabel()
        self.overview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side.addWidget(self.overview)
        side.addSpacing(8)
        side.addWidget(QLabel("OUTLINE OPACITY", objectName="sectionLabel"))
        row = QHBoxLayout()
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(OPACITY)
        self.opacity.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.opacity_label = QLabel(f"{OPACITY} %", objectName="infoValue")
        self.opacity.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v} %"))
        self.opacity.valueChanged.connect(lambda v: self.view.set_outline_opacity(v / 100))
        row.addWidget(self.opacity, stretch=1)
        row.addWidget(self.opacity_label)
        side.addLayout(row)
        self.rejected_check = QCheckBox("Show rejected pores")
        self.rejected_check.setChecked(True)
        self.rejected_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rejected_check.toggled.connect(lambda _: self._render())
        side.addWidget(self.rejected_check)
        side.addSpacing(8)
        side.addWidget(QLabel("KEYS", objectName="sectionLabel"))
        side.addWidget(QLabel(KEYS, objectName="hint"))
        side.addStretch()
        nav = QHBoxLayout()
        self.back_button = QPushButton("‹ Back")
        self.next_button = QPushButton("Next ›")
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        nav.addWidget(self.back_button)
        nav.addWidget(self.next_button)
        side.addLayout(nav)
        done = QPushButton("Done")
        done.clicked.connect(self.accept)
        side.addWidget(done)
        for b in (self.back_button, self.next_button, done):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)    # Space is "next", never a button press
        holder = QFrame()
        holder.setFixedWidth(OVERVIEW_PX + 20)
        holder.setLayout(side)
        body.addWidget(holder)
        layout.addLayout(body, stretch=1)

        self.info = QLabel(objectName="info")
        layout.addWidget(self.info)

        for keys, slot in (((Qt.Key.Key_Right, Qt.Key.Key_Space), self._next), ((Qt.Key.Key_Left,), self._back),
                           ((Qt.Key.Key_Backspace,), self._reject_centre), ((Qt.Key.Key_Delete,), self._reject_patch),
                           ((Qt.Key.Key_Escape,), lambda: self._set_tool(None))):
            for key in keys:
                QShortcut(QKeySequence(key), self, activated=slot)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=lambda: self._correct("undo"))

    def _build_tools(self) -> QFrame:
        tools = QFrame(objectName="tools")
        column = QVBoxLayout(tools)
        column.setContentsMargins(4, 6, 4, 6)
        column.setSpacing(2)
        self.tool_buttons = {}
        colors = self.colors
        for key, (name, tip, _color, _mode) in TOOLS.items():
            b = QPushButton(objectName="toolButton")
            b.setIcon(style.tool_icon(name, colors, key))
            b.setIconSize(QSize(ICON_PX, ICON_PX))
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setProperty("tool", key)
            b.clicked.connect(lambda _=False, k=key: self._set_tool(k))
            column.addWidget(b)
            self.tool_buttons[key] = b
        column.addWidget(QFrame(objectName="toolLine"))
        undo = QPushButton(objectName="toolButton")
        undo.setIcon(style.tool_icon("undo-2", colors))
        undo.setIconSize(QSize(ICON_PX, ICON_PX))
        undo.setToolTip("undo the last correction of this image (Ctrl+Z)")
        undo.clicked.connect(lambda: self._correct("undo"))
        column.addWidget(undo)
        for b in [*self.tool_buttons.values(), undo]:
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setEnabled(self.store is not None)
        if self.store is None:
            tools.setToolTip("corrections not available")
        return tools

    # ------------------------------------------------------------------ navigation

    def _mode_changed(self, index: int) -> None:
        self.mode = MODES[index][0]
        self._set_tool(None)
        state = self.score_mode if self.mode == "score" else self.tile_mode
        if state.pos < 0:                   # first time in this mode
            self._next()
        else:
            self._show_current()

    def _next(self) -> None:
        if self.mode == "score":
            m = self.score_mode
            if m.pos < len(m.history) - 1:      # forward again after going back
                m.pos += 1
            else:
                centre = next((p for p in m.order if p not in m.checked and self.boxes.box(p) is not None), None)
                if centre is None:
                    self._show_current()
                    self.finished.setText("All pores checked")
                    return
                top, bottom, left, right = self.boxes.box(centre)
                patch_top = (top + bottom) // 2 - self.size_px // 2
                patch_left = (left + right) // 2 - self.size_px // 2
                m.checked.update(int(i) for i in self.boxes.inside(patch_top, patch_left, self.size_px))
                m.checked.add(centre)           # also when it is larger than the patch
                m.history.append((patch_top, patch_left, centre))
                m.pos = len(m.history) - 1
        else:
            m = self.tile_mode
            if m.pos >= len(m.tiles) - 1:
                self._show_current()
                self.finished.setText("All tiles checked")
                return
            m.pos += 1
            m.reached = max(m.reached, m.pos)
        self._show_current()

    def _back(self) -> None:
        m = self.score_mode if self.mode == "score" else self.tile_mode
        if m.pos > 0:
            m.pos -= 1
            self._show_current()

    def _show_current(self) -> None:
        self.finished.setText("")
        if self.mode == "score":
            m = self.score_mode
            self.top, self.left, self.centre = m.history[m.pos] if m.pos >= 0 else (0, 0, 0)
            self.back_button.setEnabled(m.pos > 0)
        else:
            m = self.tile_mode
            self.top, self.left = m.tiles[m.pos] if m.pos >= 0 else (0, 0)
            self.centre = 0
            self.back_button.setEnabled(m.pos > 0)
        self._update_labels()
        self._render(fit=True)
        self._draw_overview()

    def _update_labels(self) -> None:
        if self.mode == "score":
            m = self.score_mode
            present = [p for p in m.order if self.boxes.box(p) is not None]
            done = sum(p in m.checked for p in present)
            share = 100 * done / len(present) if present else 100
            self.progress.setText(f"{done} / {len(present)} pores checked ({share:.0f} %)")
            if self.centre:
                gone = "" if self.boxes.box(self.centre) is not None else "  (rejected)"
                self.centre_label.setText(f"Pore {self.centre} · score {m.scores[self.centre]:.3f}{gone}")
            else:
                self.centre_label.setText("")
        else:
            m = self.tile_mode
            self.progress.setText(f"tile {m.pos + 1} / {len(m.tiles)}")
            self.centre_label.setText("")

    # ------------------------------------------------------------------ drawing

    def _crop(self) -> tuple[slice, slice, slice, slice]:
        """(rows, columns) of the image inside the patch and where they go in the patch."""
        height, width = self.image.shape
        t0, t1 = max(self.top, 0), min(self.top + self.size_px, height)
        l0, l1 = max(self.left, 0), min(self.left + self.size_px, width)
        t1, l1 = max(t1, t0), max(l1, l0)
        return (slice(t0, t1), slice(l0, l1), slice(t0 - self.top, t1 - self.top), slice(l0 - self.left, l1 - self.left))

    def _render(self, fit: bool = False) -> None:
        """The patch as a plain image; the pores as vector outlines above it (thin, on the pixel
        edges, the same on screen at every size of the window), the centre pore stronger."""
        size = self.size_px
        rows, columns, prow, pcol = self._crop()
        base = np.full((size, size, 3), PAD_GRAY, dtype=np.uint8)
        base[prow, pcol] = self.image.original_crop(rows.start, rows.stop, columns.start, columns.stop)
        self.view.set_image(base, (size, size), fit or self.view.fitted)

        def patch(mask: np.ndarray) -> np.ndarray:
            full = np.zeros((size, size), dtype=bool)
            full[prow, pcol] = mask
            return full
        labels = self.labels[rows, columns]
        rejected = self.rejected_labels[rows, columns]
        centre = (labels == self.centre) if self.centre else np.zeros(labels.shape, dtype=bool)
        added = np.isin(labels, list(self.added)) & ~centre if self.added else np.zeros(labels.shape, dtype=bool)
        others = (labels > 0) & ~centre & ~added
        shown_rejected = (rejected > 0) & ~np.isin(rejected, list(self.removed))    # removed by hand: not drawn
        if not self.rejected_check.isChecked():
            shown_rejected[:] = False
        view = self.view
        view.set_outline("rejected", patch(shown_rejected), style.REJECTED, OUTLINE_PX, OTHER_ALPHA)
        view.set_outline("added", patch(added), style.ADDED, OUTLINE_PX, OTHER_ALPHA)
        view.set_outline("pores", patch(others), style.PORE, OUTLINE_PX, OTHER_ALPHA)
        view.set_outline("centre", patch(centre), CENTRE_COLOR, OUTLINE_PX)
        view.set_outline_opacity(self.opacity.value() / 100)
        self._clear_hover()                 # pores or patch changed: shown again on the next mouse move

    def _visited(self) -> list[tuple[int, int]]:
        """(top, left) of every patch shown so far, score mode and tile mode together."""
        visited = list(self.tile_mode.tiles[:self.tile_mode.reached + 1])
        if self.score_mode is not None:
            visited += [(top, left) for top, left, _ in self.score_mode.history]
        return visited

    def _draw_overview(self) -> None:
        overview = self.image.overview
        height, width = self.image.shape
        pixmap = to_pixmap(overview).scaled(OVERVIEW_PX, OVERVIEW_PX, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
        factor = pixmap.width() / width
        painter = QPainter(pixmap)
        side = self.size_px * factor
        painter.setPen(QPen(VISITED_COLOR, 1))          # patches already looked at, from both modes
        for top, left in self._visited():
            painter.drawRect(QRectF(left * factor, top * factor, side, side).adjusted(0, 0, -1, -1))
        pen = QPen(self.accent, 2)
        painter.setPen(pen)
        rect = QRectF(self.left * factor, self.top * factor, side, side)
        painter.drawRect(rect.adjusted(0, 0, -1, -1) if rect.width() > 4 else rect.adjusted(-2, -2, 2, 2))
        painter.end()
        self.overview.setPixmap(pixmap)

    # ------------------------------------------------------------------ corrections

    def _set_tool(self, tool: str | None) -> None:
        if tool == self.tool or self.store is None:
            tool = None
        self.tool = tool
        for key, b in self.tool_buttons.items():
            b.setChecked(key == tool)
        if tool is None:
            self.view.set_mode("select")
            return
        _symbol, _tip, color, mode = TOOLS[tool]
        self.view.set_mode(mode, QColor(color) if color else self.accent)

    def _full(self, x: float, y: float) -> tuple[float, float]:
        """Patch px -> full image px."""
        return x + self.left, y + self.top

    def _on_clicked(self, x: int, y: int) -> None:
        if self.tool not in ("restore", "remove"):
            return
        fx, fy = self._full(x, y)
        target = self._pick(int(fx), int(fy), rejected=self.tool == "restore")
        if target is None:
            self.info.setText("nothing hit – click closer to a pore")
            return
        self._correct(self.tool, *target)

    def _on_hover(self, x: int, y: int) -> None:
        """With "+" / "−": the pore a click would hit glows in the tool's colour, as in the viewer."""
        if self.tool not in ("restore", "remove"):
            self._clear_hover()
            return
        rejected = self.tool == "restore"
        fx, fy = self._full(x, y)
        target = self._pick(int(fx), int(fy), rejected=rejected)
        labels = self.rejected_labels if rejected else self.labels
        pore = int(labels[target[1], target[0]]) if target is not None else 0
        key = (pore, rejected) if pore else None
        if key == self.hover_key:
            return
        self.hover_key = key
        if key is None:
            self.view.set_hover([], QColor())
            return
        rows, columns, prow, pcol = self._crop()
        mask = np.zeros((self.size_px, self.size_px), dtype=bool)
        mask[prow, pcol] = labels[rows, columns] == pore
        color = QColor(TOOLS[self.tool][2] or self.accent)
        color.setAlpha(HOVER_ALPHA)
        self.view.set_hover(outline(mask), color)

    def _clear_hover(self) -> None:
        self.hover_key = None
        self.view.set_hover([], QColor())

    def _on_outlined(self, points: list) -> None:
        if self.tool in ("draw", "area"):
            self._correct(self.tool, [self._full(x, y) for x, y in points])

    def _pick(self, x: int, y: int, rejected: bool) -> tuple[int, int] | None:
        """A pixel of the nearest counted (or rejected) pore within PICK_RADIUS_SCREEN_PX on screen."""
        labels = self.rejected_labels if rejected else self.labels
        height, width = labels.shape
        if not (0 <= x < width and 0 <= y < height):
            return None
        r = max(1, int(np.ceil(PICK_RADIUS_SCREEN_PX / max(self.view.zoom(), 1e-6))))
        top, left = max(0, y - r), max(0, x - r)
        window = labels[top:y + r + 1, left:x + r + 1]
        ys, xs = np.nonzero(window)
        if not len(ys):
            return None
        distance = (ys + top - y) ** 2 + (xs + left - x) ** 2
        i = int(np.argmin(distance))
        return (int(xs[i] + left), int(ys[i] + top)) if distance[i] <= r * r else None

    def _reject_centre(self) -> None:
        """Backspace: only the centre pore (score mode)."""
        if self.mode != "score" or not self.centre or self.store is None:
            return
        box = self.boxes.box(self.centre)
        if box is None:
            return
        top, bottom, left, right = box
        ys, xs = np.nonzero(self.labels[top:bottom, left:right] == self.centre)
        self._correct("remove", int(xs[0] + left), int(ys[0] + top))

    def _reject_patch(self) -> None:
        """Delete: every counted pore lying completely inside the patch (tool "area" with the patch)."""
        if self.store is None:
            return
        rows, columns, _, _ = self._crop()
        if rows.stop <= rows.start or columns.stop <= columns.start:
            return
        t, b, l, r = rows.start, rows.stop, columns.start, columns.stop
        self._correct("area", [(l, t), (r, t), (r, b), (l, b)])

    def _correct(self, action: str, *args) -> None:
        """One action of the correction session in the background (session opened on first use)."""
        if self.store is None:
            return
        self.info.setText("correcting …" if self.session is not None else "preparing corrections …")
        name = self.image.name

        def job():
            if self.session is None:
                self.session = self.store.session(name)
                self.session.defer_plots = True     # only pores and values per click; plots once at the end
            state = getattr(self.session, action)(*args)
            return state, Boxes.of(state.labels)
        future = self.worker.submit(job)
        future.add_done_callback(self.signals.corrected.emit)

    def _on_corrected(self, future: Future) -> None:
        try:
            state, boxes = future.result()
        except ValueError as error:
            self.info.setText(f"not applied: {error}")
            return
        except Exception as error:
            self.info.setText(f"correction failed: {type(error).__name__}: {error}")
            return
        self.labels, self.rejected_labels, self.boxes = state.labels, state.rejected_labels, boxes
        self.added, self.removed = state.added, state.removed
        self.row = state.row
        self.changed = True                 # pores drawn by hand have no score: never in the score list
        self._update_labels()
        self._render()
        self.info.setText("" if state.hit else "nothing hit – click closer to a pore" if self.tool else
                          "nothing to undo")

    # ------------------------------------------------------------------ closing

    def done(self, result: int) -> None:
        """Done and X alike: wait for a running correction; the corrections stay."""
        self.worker.shutdown(wait=True)
        super().done(result)
