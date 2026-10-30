"""Launcher tab "Evaluations": inputs of the evaluation modules for the next run.

Built from config.yaml, generically: every evaluation module with parameters gets a box, every
parameter a field (number, list, yes/no, selection if the module declares `choices`). A new
module appears here without any GUI code.

config.yaml stays the only source of the defaults and is never written. Only fields the user
changed override the config for the next run (passed to run() like the paths, recorded in
config_used.yaml); unchanged fields follow config.yaml, also when it changes while the launcher
is open.

Parameters a module may set per image (Step.image_params, e.g. the test data of pore_index) get no
field but a table with one row per image of the ID folder (image_inputs.py); they are passed as
per_image of the module.
"""

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from gui.launcher.image_inputs import ImageInputs
from pore_detection import list_methods

INT_RANGE = 1_000_000_000


class _Field:
    """One parameter: widget, its value as in config.yaml (default) and whether it was changed."""

    def __init__(self, value, choices: list[str] | None, on_change):
        self.default = value
        self.kind = _kind(value, choices)
        if self.kind == "bool":
            self.widget = QCheckBox()
            self.widget.toggled.connect(on_change)
        elif self.kind == "int":
            self.widget = QSpinBox()
            self.widget.setRange(-INT_RANGE, INT_RANGE)
            self.widget.valueChanged.connect(on_change)
        elif self.kind == "float":
            self.widget = QDoubleSpinBox()
            self.widget.setRange(-1e12, 1e12)
            self.widget.setDecimals(6)
            self.widget.valueChanged.connect(on_change)
        elif self.kind == "choice":
            self.widget = QComboBox()
            self.widget.addItems(choices)
            self.widget.currentTextChanged.connect(on_change)
        elif self.kind in ("list", "text"):
            self.widget = QLineEdit()
            if self.kind == "list":
                self.widget.setPlaceholderText("numbers, separated by commas")
            self.widget.textChanged.connect(on_change)
        else:                                       # nested values: only in config.yaml
            self.widget = QLabel(f"{value!r} (change in the config)")
        self.set(value)

    def set(self, value) -> None:
        widget = self.widget
        widget.blockSignals(True)
        if self.kind == "bool":
            widget.setChecked(bool(value))
        elif self.kind in ("int", "float"):
            widget.setValue(value)
        elif self.kind == "choice":
            widget.setCurrentText(str(value))
        elif self.kind == "list":
            widget.setText(", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value))
        elif self.kind == "text":
            widget.setText("" if value is None else str(value))
        widget.blockSignals(False)

    def value(self):
        """Current value; raises ValueError for a list that is not a list of numbers."""
        if self.kind == "bool":
            return self.widget.isChecked()
        if self.kind in ("int", "float"):
            return self.widget.value()
        if self.kind == "choice":
            return self.widget.currentText()
        if self.kind == "list":
            return [_number(part) for part in self.widget.text().split(",") if part.strip()]
        if self.kind == "text":
            text = self.widget.text()
            return None if text == "" and self.default is None else text
        return self.default

    def changed(self) -> bool:
        try:
            return self.value() != self.default
        except ValueError:
            return True


class EvaluationsTab(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self._structure = None                      # (module name, parameter names) of the shown form
        self._fields: dict[tuple[int, str], _Field] = {}
        self._labels: dict[tuple[int, str], QLabel] = {}
        self._names: dict[int, str] = {}
        self._tables: dict[int, ImageInputs] = {}  # module index -> inputs per image
        self._images: list[str] = []                # file names (with input.views: recordings) in the ID folder
        self.reset_button = QPushButton("Reset to config")
        self.reset_button.clicked.connect(self.reset)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(self.hint, stretch=1)
        top.addWidget(self.reset_button)
        self._form = QWidget()
        self._form_layout = QVBoxLayout(self._form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(scroll, stretch=1)

    # --- filling ---

    def refresh(self, config: dict) -> None:
        """Defaults from config.yaml; changed fields keep the user's value. A different set of
        modules / parameters rebuilds the form."""
        entries = config.get("evaluation") or []
        structure = [(e["name"], tuple(k for k in e if k != "name")) for e in entries]
        if structure != self._structure:
            self._build(entries)
            self._structure = structure
            return
        for index, entry in enumerate(entries):
            for key, value in entry.items():
                if (index, key) not in self._fields:    # name, per-image parameters
                    continue
                field = self._fields[(index, key)]
                was_changed = field.changed()
                field.default = value
                if not was_changed:
                    field.set(value)
        self._update_marks()

    def reset(self) -> None:
        for field in self._fields.values():
            field.set(field.default)
        self._update_marks()

    def _build(self, entries: list[dict]) -> None:
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._fields.clear()
        self._labels.clear()
        self._names.clear()
        self._tables.clear()
        methods = {info.name: info for info in list_methods()["evaluation"]}
        shown = 0
        for index, entry in enumerate(entries):
            name = entry["name"]
            self._names[index] = name
            params = {k: v for k, v in entry.items() if k != "name"}
            if not params:
                continue
            info = methods.get(name)
            box = QGroupBox(f"{name} – {info.description}" if info else name)
            form = QFormLayout(box)
            per_image = [key for key in params if info and key in info.image_params]
            for key, value in params.items():
                if key in per_image:
                    continue
                field = _Field(value, info.choices.get(key) if info else None, self._update_marks)
                label = QLabel(key)
                if info and key in info.params:
                    label.setToolTip(info.params[key])
                    field.widget.setToolTip(info.params[key])
                self._fields[(index, key)] = field
                self._labels[(index, key)] = label
                form.addRow(label, field.widget)
            if per_image:
                table = ImageInputs(per_image, params, info.params)
                table.set_images(self._images)
                self._tables[index] = table
                form.addRow(QLabel("Per image (empty = config):"))
                form.addRow(table)
            self._form_layout.addWidget(box)
            shown += 1
        if not shown:
            self._form_layout.addWidget(QLabel("No evaluation module with inputs in the config."))
        self._form_layout.addStretch()
        self._update_marks()

    def _update_marks(self, *_) -> None:
        changed = 0
        for key, field in self._fields.items():
            font = QFont(self._labels[key].font())
            font.setBold(field.changed())
            self._labels[key].setFont(font)
            changed += field.changed()
        self.hint.setText(
            "Defaults from the config. Changed values (bold) apply to the next run only and are "
            "recorded in config_used.yaml; the config file is not changed."
            + (f"  ·  {changed} changed" if changed else ""))
        self.reset_button.setEnabled(changed > 0)
        self.changed.emit()

    def set_images(self, images: list[str]) -> None:
        """File names (with input.views: recording names) of the chosen ID folder (rows of the per-image tables)."""
        self._images = list(images)
        for table in self._tables.values():
            table.set_images(self._images)

    # --- run ---

    def apply(self, config: dict) -> tuple[dict, list[str]]:
        """config (dict of config.yaml) with the changed values; texts of the changes for the log.
        Raises ValueError if an input is not valid."""
        entries = config.get("evaluation") or []
        if [(e["name"], tuple(k for k in e if k != "name")) for e in entries] != self._structure:
            self.refresh(config)            # config.yaml has other modules now: nothing changed yet
            return config, []
        changes = []
        for (index, key), field in self._fields.items():
            if not field.changed():
                continue
            try:
                value = field.value()
            except ValueError as error:
                raise ValueError(f"{self._names[index]}.{key}: {error}") from None
            entries[index][key] = value
            changes.append(f"{self._names[index]}.{key} = {value!r} (config.yaml: {field.default!r})")
        for index, table in self._tables.items():
            try:
                per_image = table.per_image()
            except ValueError as error:
                raise ValueError(f"{self._names[index]}: {error}") from None
            if per_image:
                entries[index]["per_image"] = per_image
            changes.append(f"{self._names[index]}: inputs for {len(per_image)} of {len(table.images)} images"
                           + (" (others: config)" if len(per_image) < len(table.images) else ""))
        return config, changes


def _kind(value, choices) -> str:
    if choices:
        return "choice"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return "list"
    if value is None or isinstance(value, str):
        return "text"
    return "other"


def _number(text: str):
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a number") from None
