"""Inputs per image (Step.image_params) in the box of an evaluation module, e.g. test data of pore_index.

A compact, scrollable table: one row per image of the ID folder, one column per parameter. It
starts empty for every folder; an empty field = the value of config.yaml. In front of every image:
green tick = all fields filled, yellow dot = some filled, nothing = none.

"Same values for all images": one row above the list applies to every image (the list is greyed
out); unticking copies its filled fields into every row, so only the exceptions are changed then
(an empty common field leaves the values of the images as they were).
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHeaderView, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

COMPLETE, PARTIAL = ("✓", "#2f9e44"), ("●", "#f08c00")    # mark and its color
LIST_HEIGHT_PX = 190            # visible height of the image list (scrolls beyond)
MARK, NAME = 0, 1               # first columns, the parameters follow


class ImageInputs(QWidget):
    changed = Signal()

    def __init__(self, params: list[str], defaults: dict, tooltips: dict[str, str]):
        super().__init__()
        self.params, self.defaults = params, defaults
        self.images: list[str] = []
        self.same = QCheckBox("Same values for all images")
        self.same.toggled.connect(self._same_toggled)
        self.common = self._table(1)
        self.common.setItem(0, NAME, _fixed("All images"))
        self.common.setFixedHeight(self.common.horizontalHeader().sizeHint().height()
                                   + self.common.verticalHeader().defaultSectionSize() + 4)
        self.common.setVisible(False)
        self.list = self._table(0)
        self.list.setFixedHeight(LIST_HEIGHT_PX)
        for table in (self.common, self.list):
            for column, param in enumerate(params, start=2):
                table.horizontalHeaderItem(column).setToolTip(tooltips.get(param, ""))
            table.itemChanged.connect(self._item_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.same)
        layout.addWidget(self.common)
        layout.addWidget(self.list)
        self._fit_names()
        self._update_marks()

    def _table(self, rows: int) -> QTableWidget:
        table = QTableWidget(rows, 2 + len(self.params))
        table.setHorizontalHeaderLabels(["", "Image", *self.params])
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(22)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = table.horizontalHeader()                  # names in full, the inputs share the rest
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(MARK, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(NAME, QHeaderView.ResizeMode.Fixed)        # set by _fit_names
        for row in range(rows):
            self._init_row(table, row, "")
        return table

    def _init_row(self, table: QTableWidget, row: int, name: str) -> None:
        table.setItem(row, MARK, _fixed("", center=True))
        table.setItem(row, NAME, _fixed(name))
        for column in range(2, 2 + len(self.params)):
            table.setItem(row, column, QTableWidgetItem(""))

    # --- filling ---

    def set_images(self, images: list[str]) -> None:
        """Image file names of the ID folder; a different list starts empty again."""
        if images == self.images:
            return
        self.images = list(images)
        self.list.blockSignals(True)
        self.list.setRowCount(len(images))
        for row, name in enumerate(images):
            self._init_row(self.list, row, name)
        self.list.blockSignals(False)
        self._fit_names()
        self._update_marks()

    def _fit_names(self) -> None:
        """Image column as wide as the longest name, the same in both tables (columns line up)."""
        width = max(table.sizeHintForColumn(NAME) for table in (self.common, self.list))
        width = max(width, self.list.horizontalHeader().sectionSizeHint(NAME)) + 12
        for table in (self.common, self.list):
            table.horizontalHeader().resizeSection(NAME, width)

    def _same_toggled(self, same: bool) -> None:
        self.common.setVisible(same)
        self.list.setEnabled(not same)
        if not same:                                # common values as start for every image
            self.list.blockSignals(True)
            for column in range(2, 2 + len(self.params)):
                text = self.common.item(0, column).text()
                if not text.strip():                # empty common field: the image values stay
                    continue
                for row in range(self.list.rowCount()):
                    self.list.item(row, column).setText(text)
            self.list.blockSignals(False)
        self._update_marks()

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() >= 2:
            self._update_marks()

    def _update_marks(self) -> None:
        common = self._texts(self.common, 0)
        for table, rows in ((self.common, 1), (self.list, self.list.rowCount())):
            table.blockSignals(True)
            for row in range(rows):
                texts = common if self.same.isChecked() else self._texts(table, row)
                filled = sum(bool(text) for text in texts)
                mark, color = COMPLETE if filled == len(texts) else PARTIAL if filled else ("", "#000000")
                item = table.item(row, MARK)
                item.setText(mark)
                item.setForeground(QColor(color))
            table.blockSignals(False)
        self.changed.emit()

    def _texts(self, table: QTableWidget, row: int) -> list[str]:
        return [table.item(row, column).text().strip() for column in range(2, 2 + len(self.params))]

    # --- run ---

    def per_image(self) -> dict[str, dict]:
        """{image file name: {parameter: value}} of the filled fields; raises ValueError for an
        input that is not a number."""
        result = {}
        for row, image in enumerate(self.images):
            texts = self._texts(self.common, 0) if self.same.isChecked() else self._texts(self.list, row)
            values = {}
            for param, text in zip(self.params, texts):
                if text:
                    try:
                        values[param] = _value(text, self.defaults.get(param))
                    except ValueError as error:
                        raise ValueError(f"{image}, {param}: {error}") from None
            if values:
                result[image] = values
        return result


def _value(text: str, default):
    """Number of the field: whole number if config.yaml has one (e.g. crack origins), else any
    number; a decimal comma is fine."""
    text = text.replace(",", ".")
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a whole number") from None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None


def _fixed(text: str, center: bool = False) -> QTableWidgetItem:
    """Cell that cannot be edited (mark, image name)."""
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    if center:
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item
