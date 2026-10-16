"""Viewer: shows the results of one run. Layout and behaviour: PLAN.md, sections 7.2 and 7.3.

Opened by the launcher right after a run. Reads only the output files (see data.py). Manual
corrections go through the correction object the launcher hands over (pore_detection.Corrections,
not imported here); without it the viewer is read-only.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDialog, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QPushButton,
                               QScrollArea, QSlider, QTableWidget, QTableWidgetItem, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import style
from .data import MANUAL, Image, Run, apply_state, load_image
from .image_view import ImageView
from .loupe import Loupe, loupe_box
from .render import Layers, compose, outline, shrink
from .scale_check import ScaleCheck, scale_crop_box

# key figures shown as tiles: (column of images.json, label, unit) – fixed on purpose (PLAN.md 7.2)
KEY_FIGURES = [
    ("porosity_percent", "Porosity", "%"),
    ("pore_count", "Pores", ""),
    ("pore_area_px2", "Pore area", "px²"),
    ("sample_area_px2", "Sample area", "px²"),
]
RIGHT_COLUMN_PX = 330
LOUPE_COLUMN_PX = 300
PLOT_HEIGHT_PX = 300
NO_SELECTION = "Click a pore in the image or in the list."
# correction tools: key -> (symbol, tooltip, colour of the active button / drawn outline, view mode)
TOOLS = {
    "restore": ("+", "count a rejected pore (click on it)", "#0c8599", "click"),
    "remove": ("−", "reject a counted pore (click on it); a drawn pore is deleted", "#d63370", "click"),
    "draw": ("✎", "draw a missed pore (hold the mouse button and trace its outline)", None, "outline"),
    "area": ("⬚", "reject every counted pore lying completely inside the traced area", "#d63370", "outline"),
}
ADDED = style.ADDED


def format_number(value, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "–"
    if isinstance(value, float) and abs(value) < 1000:
        text = f"{value:.4g}"
    else:
        text = f"{value:,.0f}".replace(",", " ")
    return f"{text} {unit}".strip()


UNIT_SUFFIXES = (("_px2", "px²"), ("_um2", "µm²"), ("_px", "px"), ("_um", "µm"), ("_percent", "%"))


def column_label(column: str) -> tuple[str, str]:
    """"area_px2" -> ("Area", "px²"): readable name and unit of a result column."""
    for suffix, unit in UNIT_SUFFIXES:
        if column.endswith(suffix):
            return column[:-len(suffix)].replace("_", " ").capitalize(), unit
    return column.replace("_", " ").capitalize(), ""


def card(title: str, body: QWidget | None = None) -> QFrame:
    frame = QFrame(objectName="card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(QLabel(title.upper(), objectName="cardTitle"))
    if body is not None:
        layout.addWidget(body, stretch=1)
    return frame


class LayerRow(QFrame):
    """One overlay: checkbox, color patch, name, count; highlighted while on, click anywhere toggles."""

    def __init__(self, name: str, color: str, checked: bool):
        super().__init__(objectName="layerRow")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(8)
        self.check = QCheckBox()
        self.check.setChecked(checked)
        patch = QLabel()
        patch.setFixedSize(10, 10)
        patch.setStyleSheet(f"background: {color}; border-radius: 2px;")
        self.count = QLabel("", objectName="layerCount")
        row.addWidget(self.check)
        row.addWidget(patch)
        row.addWidget(QLabel(name), stretch=1)
        row.addWidget(self.count)
        self.check.toggled.connect(self._restyle)
        self._restyle(checked)

    def mousePressEvent(self, event) -> None:
        self.check.toggle()

    def _restyle(self, on: bool) -> None:
        self.setProperty("on", on)
        self.style().unpolish(self)
        self.style().polish(self)


def dot_icon(color: str) -> QIcon:
    pixmap = QPixmap(10, 10)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, 8, 8)
    painter.end()
    return QIcon(pixmap)


class _Signals(QObject):
    loaded = Signal(int, object)            # image index, Future[Image] (emitted from the loader thread)
    corrected = Signal(int, object)         # image index, Future[State] (emitted from the correction thread)


class ViewerWindow(QMainWindow):
    def __init__(self, results_dir: Path, gui_settings: dict, corrections=None, parent=None):
        super().__init__(parent)
        self.corrections = corrections      # pore_detection.Corrections or None (read-only)
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="corrections")   # one action at a time
        self.session = None                 # correction session of the shown image (opened on first use)
        self.tool: str | None = None
        self.selected_rejected = False      # the selected pore is a rejected one
        self.run = Run.load(results_dir)
        self.setWindowTitle(f"Pore Detection – Viewer – {self.run.results_dir.name}")
        self.resize(1500, 950)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.index = 0                      # shown image (may still be loading)
        self.image: Image | None = None     # None while loading
        self.selected = 0                   # pore id, 0 = none
        self.loader = ThreadPoolExecutor(1, thread_name_prefix="viewer")
        self.loading: dict[int, Future] = {}    # shown image and its neighbours (preloaded)
        self.signals = _Signals()
        self.signals.loaded.connect(self._on_loaded)
        self.signals.corrected.connect(self._on_corrected)

        # theme: the toggle only lasts while the viewer is open
        self.hints = QApplication.styleHints()
        self.scheme_before = self.hints.colorScheme()
        theme = gui_settings.get("theme", "system")
        self.dark = theme == "dark" or (theme == "system" and self.scheme_before == Qt.ColorScheme.Dark)

        page = QScrollArea(objectName="page")
        page.setWidgetResizable(True)
        content = QWidget(objectName="pageContent")
        grid = QGridLayout(content)
        grid.setContentsMargins(16, 16, 16, 24)
        grid.setSpacing(16)
        self.stage = self._build_stage()
        grid.addWidget(self.stage, 0, 0)
        grid.addWidget(self._build_right_column(), 0, 1)
        grid.addWidget(self._build_plots(), 1, 0)
        grid.addWidget(self._build_pore_list(), 1, 1, Qt.AlignmentFlag.AlignTop)
        grid.setColumnStretch(0, 1)
        page.setWidget(content)
        self.page = page
        page.viewport().installEventFilter(self)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(page, stretch=1)
        self.setCentralWidget(central)
        self._apply_theme()

        for key, step in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1)):
            QShortcut(QKeySequence(key), self, activated=lambda s=step: self.show_image(self._index() + s))
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=lambda: self._set_tool(None))

        if len(self.run.images):
            self.show_image(0)
        else:
            self.name_label.setText("no image succeeded – see Errors")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    # ------------------------------------------------------------------ layout

    def _build_header(self) -> QWidget:
        header = QWidget(objectName="header")
        row = QHBoxLayout(header)
        row.setContentsMargins(18, 10, 18, 10)
        row.setSpacing(6)
        row.addWidget(QLabel("Pore Detection", objectName="title"))
        row.addSpacing(18)
        self.prev_button = QPushButton("‹", objectName="navButton")
        self.next_button = QPushButton("›", objectName="navButton")
        self.prev_button.setToolTip("previous image (arrow left)")
        self.next_button.setToolTip("next image (arrow right)")
        self.prev_button.clicked.connect(lambda: self.show_image(self._index() - 1))
        self.next_button.clicked.connect(lambda: self.show_image(self._index() + 1))
        self.position_label = QLabel(objectName="position")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setMinimumWidth(56)
        self.name_label = QLabel(objectName="imageName")
        for widget in (self.prev_button, self.position_label, self.next_button):
            row.addWidget(widget)
        row.addSpacing(14)
        row.addWidget(self.name_label)
        row.addStretch()

        for i, (title, entry) in enumerate(self.run.chain()):
            if i:
                row.addWidget(QLabel("→", objectName="arrow"))
            chip = QLabel(str(entry.get("name", "?")), objectName="chip")
            params = "\n".join(f"{k}: {v}" for k, v in entry.items() if k != "name") or "no parameters"
            chip.setToolTip(f"{title}\n{params}")
            row.addWidget(chip)
        row.addSpacing(16)
        buttons = [("ID values", self._show_id_values), (f"Errors ({len(self.run.errors)})", self._show_errors),
                   ("Settings", self._show_settings), ("Light / dark", self._toggle_theme)]
        for text, slot in buttons:
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        return header

    def _build_stage(self) -> QFrame:
        """Value strip on top; tools | image | optical check; info bar with opacity below."""
        stage = QFrame(objectName="card")
        outer = QVBoxLayout(stage)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        strip = QWidget(objectName="valueStrip")
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(16, 9, 16, 9)
        self.pore_values = QLabel()
        self.pore_values.setTextFormat(Qt.TextFormat.RichText)
        strip_layout.addWidget(self.pore_values)
        outer.addWidget(strip)

        body = QWidget(objectName="stageBody")
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        image_area = QHBoxLayout()
        image_area.setContentsMargins(12, 12, 12, 12)
        image_area.setSpacing(12)
        image_area.addWidget(self._build_tools(), alignment=Qt.AlignmentFlag.AlignTop)
        self.view = ImageView()
        self.view.zoom_changed.connect(lambda _: self._update_info())
        self.view.clicked.connect(self._on_image_clicked)
        self.view.outlined.connect(self._on_outlined)
        image_area.addWidget(self.view, stretch=1)
        row.addLayout(image_area, stretch=1)
        row.addWidget(self._build_check_column())
        outer.addWidget(body, stretch=1)

        bar = QWidget(objectName="infoBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(16, 7, 16, 7)
        self.info_label = QLabel(objectName="info")
        bar_layout.addWidget(self.info_label)
        bar_layout.addStretch()
        bar_layout.addWidget(QLabel("Opacity", objectName="hint"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.setFixedWidth(110)
        self.opacity_label = QLabel("100 %", objectName="infoValue")
        self.opacity_label.setMinimumWidth(40)
        self.opacity.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v} %"))
        self.opacity.valueChanged.connect(self._layers_changed)
        bar_layout.addWidget(self.opacity)
        bar_layout.addWidget(self.opacity_label)
        outer.addWidget(bar)
        return stage

    def _build_tools(self) -> QWidget:
        """Floating tool card: zoom, correction tools (PLAN.md 7.3), undo, reset."""
        tools = QFrame(objectName="tools")
        layout = QVBoxLayout(tools)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(2)

        def button(text, tip, slot=None, tool=None):
            b = QPushButton(text, objectName="toolButton")
            b.setToolTip(tip)
            if tool:
                b.setCheckable(True)
                b.setProperty("tool", tool)
            if slot:
                b.clicked.connect(slot)
            layout.addWidget(b)
            return b

        button("⊕", "zoom in", lambda: self.view.zoom_by(1.25))
        button("⊖", "zoom out", lambda: self.view.zoom_by(1 / 1.25))
        button("⤢", "fit the whole image (double click)", lambda: self.view.fit())
        layout.addWidget(QFrame(objectName="toolLine"))
        self.tool_buttons = {}
        for key, (symbol, tip, _color, _mode) in TOOLS.items():
            self.tool_buttons[key] = button(symbol, tip, lambda _=False, k=key: self._set_tool(k), key)
        layout.addWidget(QFrame(objectName="toolLine"))
        self.undo_button = button("↶", "undo the last correction of this image (Ctrl+Z)", self._undo)
        self.reset_button = button("🗑", "discard all corrections of this image", self._reset)
        editable = self.corrections is not None
        for b in [*self.tool_buttons.values(), self.undo_button, self.reset_button]:
            b.setEnabled(editable)
        if not editable:
            tools.setToolTip("corrections not available")
        return tools

    def _build_check_column(self) -> QWidget:
        """Optical check of the scale (top) and of the selected pore."""
        column = QWidget(objectName="loupeColumn")
        column.setFixedWidth(LOUPE_COLUMN_PX)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(QLabel("SCALE – OPTICAL CHECK", objectName="cardTitle"))
        scale_box = QVBoxLayout()
        scale_box.setContentsMargins(14, 12, 14, 14)
        scale_box.setSpacing(8)
        self.scale_check = ScaleCheck()
        self.scale_check.setToolTip("scale overlay from the original; the marker shows the bar length that was measured")
        scale_box.addWidget(self.scale_check)
        self.scale_values = QLabel(objectName="mono")
        self.scale_values.setWordWrap(True)
        scale_box.addWidget(self.scale_values)
        layout.addLayout(scale_box)

        layout.addWidget(QLabel("PORE – OPTICAL CHECK", objectName="cardTitle"))
        pore_box = QVBoxLayout()
        pore_box.setContentsMargins(14, 12, 14, 14)
        pore_box.setSpacing(8)
        self.loupe = Loupe()
        self.loupe.setFixedSize(LOUPE_COLUMN_PX - 28, LOUPE_COLUMN_PX - 28)
        self.loupe.show_message(NO_SELECTION)
        pore_box.addWidget(self.loupe)
        surround_row = QHBoxLayout()
        surround_row.addWidget(QLabel("Surround", objectName="hint"))
        self.surround = QSlider(Qt.Orientation.Horizontal)
        self.surround.setRange(3, 24)       # half steps: 1.5x … 12x the pore size
        self.surround.setValue(6)
        self.surround.valueChanged.connect(self._update_loupe)
        self.surround_label = QLabel("3.0×", objectName="infoValue")
        surround_row.addWidget(self.surround, stretch=1)
        surround_row.addWidget(self.surround_label)
        pore_box.addLayout(surround_row)
        layout.addLayout(pore_box)
        layout.addStretch()
        return column

    def _build_right_column(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(RIGHT_COLUMN_PX)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        tiles = QWidget(objectName="tileGrid")
        tiles_grid = QGridLayout(tiles)
        tiles_grid.setContentsMargins(0, 0, 0, 0)
        tiles_grid.setSpacing(1)
        self.tile_values = {}
        for i, (key, label, _unit) in enumerate(KEY_FIGURES):
            tile = QFrame(objectName="tile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(14, 12, 14, 12)
            tile_layout.setSpacing(2)
            value = QLabel("–", objectName="tileValue")
            tile_layout.addWidget(value)
            tile_layout.addWidget(QLabel(label, objectName="tileLabel"))
            tiles_grid.addWidget(tile, i // 2, i % 2)
            self.tile_values[key] = value
        layout.addWidget(card("Key figures", tiles))

        layers = QWidget()
        layers_layout = QVBoxLayout(layers)
        layers_layout.setContentsMargins(10, 12, 10, 12)
        layers_layout.setSpacing(4)
        layers_layout.addWidget(QLabel("BASE IMAGE", objectName="sectionLabel"))
        base_row = QHBoxLayout()
        base_row.setSpacing(6)
        self.base_group = QButtonGroup(self)
        for key, text, tip in (("original", "Original", "the micrograph"),
                               ("background", "Background", "estimated background – needs a background "
                                                            "correction filter (follows later)")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setChecked(key == "original")
            button.setEnabled(key != "background")
            button.setToolTip(tip)
            button.setProperty("base", key)
            self.base_group.addButton(button)
            base_row.addWidget(button)
        self.base_group.buttonClicked.connect(self._layers_changed)
        layers_layout.addLayout(base_row)
        layers_layout.addSpacing(10)
        layers_layout.addWidget(QLabel("OVERLAY", objectName="sectionLabel"))
        self.embedding_row = LayerRow("Embedding medium", style.EMBEDDING, checked=False)
        self.rejected_row = LayerRow("Rejected", style.REJECTED, checked=True)
        self.removed_row = LayerRow("Removed by hand", style.REMOVED, checked=True)
        self.added_row = LayerRow("Added by hand", ADDED, checked=True)
        self.pores_row = LayerRow("Pores", style.PORE, checked=True)
        for layer in (self.embedding_row, self.rejected_row, self.removed_row, self.added_row, self.pores_row):
            layer.check.toggled.connect(self._layers_changed)
            layers_layout.addWidget(layer)
        self.embedding_check, self.pores_check = self.embedding_row.check, self.pores_row.check
        layers_layout.addStretch()
        layout.addWidget(card("Layers", layers), stretch=1)
        return column

    def _build_plots(self) -> QWidget:
        """Two-column grid for plots; empty for now (PLAN.md 7.2)."""
        plots = QWidget()
        grid = QGridLayout(plots)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(16)
        for i in range(2):
            hint = QLabel("plots follow later", objectName="hint")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            frame = card("Plot", hint)
            frame.setFixedHeight(PLOT_HEIGHT_PX)
            grid.addWidget(frame, i // 2, i % 2)
        return plots

    def _build_pore_list(self) -> QWidget:
        self.pore_list = QTreeWidget()
        self.pore_list.setColumnCount(2)
        self.pore_list.setHeaderHidden(True)
        self.pore_list.setRootIsDecorated(False)
        self.pore_list.setUniformRowHeights(True)
        self.pore_list.setIndentation(0)
        self.pore_list.itemSelectionChanged.connect(self._on_list_selection)
        frame = card("Pores by area", self.pore_list)
        frame.setFixedSize(RIGHT_COLUMN_PX, PLOT_HEIGHT_PX)
        return frame

    def eventFilter(self, watched, event) -> bool:
        # the top block (image + right column) fills the visible height; plots follow below
        if watched is self.page.viewport() and event.type() == QEvent.Type.Resize:
            self.stage.setFixedHeight(max(480, event.size().height() - 40))
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------ images

    def _index(self) -> int:
        return self.index

    def show_image(self, index: int) -> None:
        if not 0 <= index < len(self.run.images):
            return
        self.index, self.image, self.selected, self.selected_rejected = index, None, 0, False
        self.session = None                 # opened again on first use (the store keeps the corrections)
        row = self.run.row(index)
        self.position_label.setText(f"{index + 1} / {len(self.run.images)}")
        self.name_label.setText(row["image"])
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(index < len(self.run.images) - 1)
        for key, _label, unit in KEY_FIGURES:
            self.tile_values[key].setText(format_number(row.get(key), unit))

        # keep only the shown image and its neighbours; others are dropped (a running load just finishes)
        for i in [i for i in self.loading if abs(i - index) > 1]:
            self.loading.pop(i).cancel()
        future = self._load(index)
        if future.done():
            self._on_loaded(index, future)
        else:
            self.view.clear()
            self.pore_list.clear()
            self.loupe.show_message("loading …")
            self._show_pore_values(0)
            self.scale_check.show_message("loading …")
            self.scale_values.setText("")
            self.info_label.setText(f"loading {row['image']} …")

    def _load(self, index: int) -> Future:
        """Load an image in the background (once); the result arrives via signals.loaded."""
        if index not in self.loading:
            def job():
                if self.corrections is not None:
                    self.corrections.flush()    # never read files that are still being written
                return load_image(self.run, index)
            future = self.loader.submit(job)
            future.add_done_callback(lambda f, i=index: self.signals.loaded.emit(i, f))
            self.loading[index] = future
        return self.loading[index]

    def _on_loaded(self, index: int, future: Future) -> None:
        if index != self.index or self.image is not None or future.cancelled():
            return
        try:
            image = future.result()
        except Exception as error:
            self.info_label.setText(f"cannot show image {index + 1}: {error}")
            return
        self.image = image
        self._update_counts()
        self._fill_pore_list()
        self.view.set_selection([], QColor())
        self._render(fit=True)
        self._update_detail_source()
        self.loupe.show_message(NO_SELECTION)
        self._show_pore_values(0)
        self._update_scale_check()
        self._update_info()
        for neighbour in (index + 1, index - 1):    # preload: next first, then previous
            if 0 <= neighbour < len(self.run.images):
                self._load(neighbour)

    def _layers(self) -> Layers:
        return Layers(base=self.base_group.checkedButton().property("base"), pores=self.pores_check.isChecked(),
                      embedding=self.embedding_check.isChecked(), rejected=self.rejected_row.check.isChecked(),
                      removed=self.removed_row.check.isChecked(), added=self.added_row.check.isChecked(),
                      opacity=self.opacity.value() / 100)

    def _layers_changed(self, *_) -> None:
        self._render(fit=False)
        self._update_detail_source()

    def _render(self, fit: bool) -> None:
        """Whole-image view from the overview."""
        image = self.image
        if image is None:
            return
        rgb = compose(image.overview, image.overview_masks, image.overview_roi, self._layers())
        self.view.set_image(rgb, (image.shape[1], image.shape[0]), fit)

    def _update_detail_source(self) -> None:
        """Full resolution of the visible part while zoomed in."""
        image, layers = self.image, self._layers()
        if image is None:
            self.view.set_detail_source(None)
            return

        def detail(top, bottom, left, right, out_w, out_h):
            size = (out_w, out_h)
            masks = {name: shrink(mask, size) for name, mask in image.status_masks(top, bottom, left, right).items()}
            return compose(shrink(image.original_crop(top, bottom, left, right), size), masks,
                           shrink(image.roi[top:bottom, left:right], size), layers)
        self.view.set_detail_source(detail)

    def _update_info(self) -> None:
        image = self.image
        if image is None:
            return
        um = image.scale.um_per_px
        parts = [f"{image.shape[1]} × {image.shape[0]} px",
                 f"{um:.6g} µm/px" if um else "no scale – values in pixels",
                 f"zoom {self.view.zoom() * 100:.0f} %"]
        if image.message:
            parts.append(image.message)
        self.info_label.setText("   ·   ".join(parts))

    def _update_scale_check(self) -> None:
        """Scale overlay with the measured bar; below it what was read and what follows from it."""
        image = self.image
        scale = image.scale
        if scale.bar is None:
            self.scale_check.show_message("no scale overlay found")
            self.scale_values.setText(f"Scale: {scale.um_per_px:.6g} µm/px (not from an overlay)"
                                      if scale.um_per_px else "No scale – values in pixels.")
            return
        top, bottom, left, right = scale_crop_box(scale.box, scale.bar, image.shape)
        bar = (scale.bar[0] - top, scale.bar[1] - top, scale.bar[2] - left, scale.bar[3] - left)
        accent = QColor((style.DARK if self.dark else style.LIGHT)["accent"])
        if image.original is None:
            self.scale_check.show_message(image.message)
        else:
            self.scale_check.show_scale(image.original_crop(top, bottom, left, right), bar, accent)
        lines = [f"Read: {scale.label}" if scale.label else "Read: nothing (label not readable)",
                 f"Bar: {scale.bar[3] - scale.bar[2]} px"]
        if scale.um_per_px:
            lines.append(f"Scale: {scale.um_per_px:.6g} µm/px")
        self.scale_values.setText("\n".join(lines))

    # ------------------------------------------------------------------ pores

    def _fill_pore_list(self) -> None:
        table = self.image.table
        use_um = self.image.scale.um_per_px is not None and "area_um2" in table
        column, unit = ("area_um2", "µm²") if use_um else ("area_px2", "px²")
        if column in table:
            table = table.sort_values(column, ascending=False, kind="stable")
            values = table[column].tolist()
        else:
            values = [None] * len(table)
        items = []
        dot, added_dot = dot_icon(style.PORE), dot_icon(ADDED)
        mono = QFont(["IBM Plex Mono", "Cascadia Mono", "Consolas"], 9)
        for pore_id, value in zip(table["pore_id"].tolist(), values):
            item = QTreeWidgetItem([f"Pore {pore_id}", format_number(value, unit)])
            item.setIcon(0, added_dot if pore_id in self.image.added else dot)
            if pore_id in self.image.added:
                item.setToolTip(0, "added by hand")
            item.setFont(1, mono)
            item.setData(0, Qt.ItemDataRole.UserRole, int(pore_id))
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            items.append(item)
        self.pore_list.blockSignals(True)
        self.pore_list.clear()
        self.pore_list.addTopLevelItems(items)
        self.pore_list.setColumnWidth(0, 130)
        self.pore_list.blockSignals(False)
        self.list_items = {item.data(0, Qt.ItemDataRole.UserRole): item for item in items}

    def _on_list_selection(self) -> None:
        items = self.pore_list.selectedItems()
        if items:
            self._select(items[0].data(0, Qt.ItemDataRole.UserRole), center=True)

    def _on_image_clicked(self, x: int, y: int) -> None:
        image = self.image
        if image is None:
            return
        if self.tool in ("restore", "remove"):
            self._correct(self.tool, x, y)
            return
        pore = int(image.labels[y, x])
        if pore == 0 and int(image.rejected_labels[y, x]):
            self._select(int(image.rejected_labels[y, x]), rejected=True)
        else:
            self._select(pore)

    def _on_outlined(self, points: list) -> None:
        if self.tool in ("draw", "area") and self.image is not None:
            self._correct(self.tool, points)

    # ------------------------------------------------------------------ corrections

    def _set_tool(self, tool: str | None) -> None:
        """Activate a correction tool; the active one again (or Esc) = back to selecting."""
        if tool == self.tool or self.corrections is None:
            tool = None
        self.tool = tool
        for key, b in self.tool_buttons.items():
            b.setChecked(key == tool)
        if tool is None:
            self.view.set_mode("select")
            return
        _symbol, _tip, color, mode = TOOLS[tool]
        accent = (style.DARK if self.dark else style.LIGHT)["accent"]
        self.view.set_mode(mode, QColor(color or accent))

    def _correct(self, action: str, *args) -> None:
        """Run one action of the correction session in the background (session opened on first use)."""
        if self.corrections is None or self.image is None:
            return
        index, name = self.index, self.image.name
        self.info_label.setText("correcting …" if self.session is not None else "preparing corrections …")

        def job():
            if self.session is None or self.session.image != name:
                self.session = self.corrections.session(name)
            return getattr(self.session, action)(*args)
        future = self.worker.submit(job)
        future.add_done_callback(lambda f: self.signals.corrected.emit(index, f))

    def _undo(self) -> None:
        self._correct("undo")

    def _reset(self) -> None:
        self._correct("reset")

    def _on_corrected(self, index: int, future: Future) -> None:
        try:
            state = future.result()
        except Exception as error:
            self.info_label.setText(f"correction failed: {type(error).__name__}: {error}")
            return
        self.run.set_row(index, state.row)
        cached = self.loading.get(index)
        image = self.image if index == self.index and self.image is not None else (
            cached.result() if cached is not None and cached.done() and not cached.exception() else None)
        if image is None:
            return
        apply_state(image, state)
        if image is not self.image:
            return
        for key, _label, unit in KEY_FIGURES:
            self.tile_values[key].setText(format_number(state.row.get(key), unit))
        self._update_counts()
        self._fill_pore_list()
        self._render(fit=False)
        self._update_detail_source()
        gone = self.selected and not (image.pore_labels(self.selected_rejected) == self.selected).any()
        self._select(0 if gone else self.selected, rejected=self.selected_rejected)
        self._update_info()
        if not state.hit:
            self.info_label.setText("nothing hit – click closer to a pore" if self.tool else "nothing to undo")

    def _update_counts(self) -> None:
        image = self.image
        manual = image.rejected["filter"] == MANUAL
        self.pores_row.count.setText(str(len(image.table)))
        self.rejected_row.count.setText(str(int((~manual).sum())))
        self.removed_row.count.setText(str(len(image.removed)))
        self.added_row.count.setText(str(len(image.added)))
        corrected = int(image.row.get("manual_corrections") or 0) > 0
        if self.corrections is not None:
            self.undo_button.setEnabled(corrected)
            self.reset_button.setEnabled(corrected)

    def _select(self, pore_id: int, center: bool = False, rejected: bool = False) -> None:
        self.selected, self.selected_rejected = pore_id, rejected and pore_id != 0
        self._sync_list(0 if rejected else pore_id)
        self._show_pore_values(pore_id)
        if pore_id == 0:
            self.view.set_selection([], QColor())
            self.loupe.show_message(NO_SELECTION)
            return
        image = self.image
        labels = image.pore_labels(self.selected_rejected)
        top, bottom, left, right = image.pore_box(pore_id, self.selected_rejected)
        accent = QColor((style.DARK if self.dark else style.LIGHT)["accent"])
        self.view.set_selection(outline(labels[top:bottom, left:right] == pore_id, left, top), accent)
        if center:
            self.view.centerOn((left + right) / 2, (top + bottom) / 2)
        self._update_loupe()

    def _update_loupe(self) -> None:
        self.surround_label.setText(f"{self.surround.value() / 2:.1f}×")
        image, pore_id = self.image, self.selected
        if image is None or pore_id == 0:
            return
        if image.original is None:
            self.loupe.show_message(image.message)
            return
        rejected = self.selected_rejected
        top, bottom, left, right = loupe_box(image.pore_box(pore_id, rejected), self.surround.value() / 2, image.shape)
        color = style.REJECTED if rejected else ADDED if pore_id in image.added else style.PORE
        self.loupe.show_pore(image.original[top:bottom, left:right],
                             image.pore_labels(rejected)[top:bottom, left:right] == pore_id, QColor(color))

    def _show_pore_values(self, pore_id: int) -> None:
        """Value strip above the image: all pores.json columns of the selected pore and its status
        (added by hand, or rejected with filter and reason)."""
        c = style.DARK if self.dark else style.LIGHT
        image = self.image
        table = image.table if image is not None else None
        columns = [k for k in table.columns if k != "pore_id"] if table is not None else []

        def part(name, value):
            return (f"<span style='color:{c['muted']}'>{name}</span>&nbsp;&nbsp;"
                    f"<span style='color:{c['ink']}; font-family:Consolas'>{value}</span>")
        gap = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        if image is not None and pore_id and self.selected_rejected:
            filter_name, reason = image.reason(pore_id) or ("?", "")
            status = "removed by hand" if filter_name == MANUAL else f"rejected by {filter_name}"
            self.pore_values.setText(gap.join([part("Pore", pore_id), part(status, reason)]))
            return
        row = table[table["pore_id"] == pore_id] if table is not None and pore_id else None
        values = row.iloc[0] if row is not None and len(row) else None
        parts = [part("Pore", pore_id if values is not None else "–")]
        for k in columns:
            name, unit = column_label(k)
            parts.append(part(name, format_number(values[k], unit) if values is not None else "–"))
        if values is not None and pore_id in image.added:
            parts.append(f"<span style='color:{ADDED}'>added by hand</span>")
        self.pore_values.setText(gap.join(parts))

    def _sync_list(self, pore_id: int) -> None:
        self.pore_list.blockSignals(True)
        self.pore_list.clearSelection()
        item = self.list_items.get(pore_id)
        if item is not None:
            item.setSelected(True)
            self.pore_list.scrollToItem(item)
        self.pore_list.blockSignals(False)

    # ------------------------------------------------------------------ header dialogs

    def _dialog(self, title: str, body: QWidget) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(700, 450)
        QVBoxLayout(dialog).addWidget(body)
        dialog.show()

    def _show_id_values(self) -> None:
        body = QWidget()
        form = QFormLayout(body)
        if self.run.id_values is None:
            form.addRow(QLabel("no ID values (run cancelled or no image succeeded)"))
        for key, value in (self.run.id_values or {}).items():
            form.addRow(key, QLabel(format_number(value)))
        self._dialog("ID values", body)

    def _show_errors(self) -> None:
        errors = self.run.errors
        table = QTableWidget(len(errors), len(errors.columns))
        table.setHorizontalHeaderLabels(list(errors.columns))
        for r, values in enumerate(errors.itertuples(index=False)):
            for c, value in enumerate(values):
                table.setItem(r, c, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        self._dialog(f"Errors ({len(errors)})", table)

    def _show_settings(self) -> None:
        text = QPlainTextEdit(self.run.config_text)
        text.setReadOnly(True)
        self._dialog("Settings used (config_used.yaml)", text)

    # ------------------------------------------------------------------ theme, closing

    def _apply_theme(self) -> None:
        self.hints.setColorScheme(Qt.ColorScheme.Dark if self.dark else Qt.ColorScheme.Light)
        self.setStyleSheet(style.stylesheet(style.DARK if self.dark else style.LIGHT))
        self._show_pore_values(self.selected)
        if self.image is not None:
            self._update_scale_check()          # accent color of the marker
            if self.selected:
                self._select(self.selected)

    def _toggle_theme(self) -> None:
        self.dark = not self.dark
        self._apply_theme()

    def closeEvent(self, event) -> None:
        self.loader.shutdown(wait=False, cancel_futures=True)
        self.worker.shutdown(wait=True)             # finish a running correction
        if self.corrections is not None:
            self.corrections.close()                # wait for the files
        self.hints.setColorScheme(self.scheme_before)
        super().closeEvent(event)
