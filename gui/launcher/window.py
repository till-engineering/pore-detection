"""Launcher window: choose paths, show the config, start / cancel a run, follow progress and log.

Uses only the public API of pore_detection. config.yaml is read again at every start, so a
change in the file applies to the next run without restarting the launcher. The launcher never
writes the config.
"""

import logging
import traceback
from pathlib import Path
from threading import Event

from PySide6.QtCore import QFileSystemWatcher, QObject, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from gui.viewer.window import ViewerWindow
from pore_detection import (CONFIG_PATH, LOG_FORMAT, Corrections, RunAborted, RunResult, load_config,
                            resolve_path, run)

log = logging.getLogger("pore_detection")


class _LogEmitter(QObject):
    line = Signal(str)


class _GuiLogHandler(logging.Handler):
    """Forwards log records of the pipeline (any thread) to the log view (GUI thread)."""

    def __init__(self):
        super().__init__()
        self.emitter = _LogEmitter()
        self.setFormatter(LOG_FORMAT)

    def emit(self, record: logging.LogRecord) -> None:
        self.emitter.line.emit(self.format(record))


class _RunWorker(QObject):
    """Runs the pipeline in a background thread."""

    progress = Signal(int, int, str, str)
    finished = Signal(object)        # RunResult
    failed = Signal(str)

    def __init__(self, config, input_path: Path, output_dir: Path):
        super().__init__()
        self.config, self.input_path, self.output_dir = config, input_path, output_dir
        self.cancel = Event()

    @Slot()
    def run(self) -> None:
        try:
            result = run(self.config, self.input_path, self.output_dir,
                         progress=self.progress.emit, cancel=self.cancel)
        except RunAborted as error:
            self.failed.emit(f"run aborted: {error}")
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pore Detection – Launcher")
        self.resize(900, 650)
        self._thread: QThread | None = None
        self._worker: _RunWorker | None = None
        self._last_result: RunResult | None = None
        self._total = 0                   # images in the ID folder, from the progress reports
        self._gui_settings: dict = {}     # config.gui of the last run, passed to the viewer
        self._viewer: ViewerWindow | None = None

        # paths, prefilled from config.yaml
        self.input_edit, self.output_edit = QLineEdit(), QLineEdit()
        try:
            config = load_config()
            self.input_edit.setText(str(resolve_path(config.input.path)))
            self.output_edit.setText(str(resolve_path(config.output.dir)))
        except Exception as error:  # broken config: the launcher still opens and shows it
            self.input_edit.setPlaceholderText(f"config.yaml not readable: {error}")
        paths = QGroupBox("Paths")
        grid = QGridLayout(paths)
        for row, (label, edit, title) in enumerate([
                ("ID folder (input)", self.input_edit, "Choose ID folder"),
                ("Output folder", self.output_edit, "Choose output folder")]):
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, e=edit, t=title: self._browse(e, t))
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(browse, row, 2)

        # run controls
        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        self.viewer_button = QPushButton("Open viewer")
        self.cancel_button.setEnabled(False)
        self.viewer_button.setEnabled(False)
        self.viewer_button.setToolTip("show the results of the last run")
        self.viewer_button.clicked.connect(self._open_viewer)
        self.start_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        buttons = QHBoxLayout()
        for button in (self.start_button, self.cancel_button):
            buttons.addWidget(button)
        buttons.addStretch()
        self.open_viewer_check = QCheckBox("Open viewer after the run")
        self.open_viewer_check.setChecked(True)
        buttons.addWidget(self.open_viewer_check)
        buttons.addWidget(self.viewer_button)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("ready")

        # log and config view
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.log_view, self.config_view = QPlainTextEdit(), QPlainTextEdit()
        for view in (self.log_view, self.config_view):
            view.setReadOnly(True)
            view.setFont(mono)
        self.log_view.setMaximumBlockCount(10000)
        tabs = QTabWidget()
        tabs.addTab(self.log_view, "Log")
        tabs.addTab(self.config_view, "Config (config.yaml, read-only)")

        layout = QVBoxLayout()
        layout.addWidget(paths)
        layout.addLayout(buttons)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(tabs, stretch=1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # pipeline log -> log view
        self._log_handler = _GuiLogHandler()
        self._log_handler.emitter.line.connect(self.log_view.appendPlainText)
        log.addHandler(self._log_handler)

        # config view always shows the current file content
        self._watcher = QFileSystemWatcher([str(CONFIG_PATH)])
        self._watcher.fileChanged.connect(self._show_config)
        self._show_config()

    # --- config ---

    @Slot()
    def _show_config(self) -> None:
        try:
            text = CONFIG_PATH.read_text(encoding="utf-8")
        except OSError as error:
            text = f"cannot read {CONFIG_PATH}: {error}"
        self.config_view.setPlainText(text)
        # editors often replace the file on save, which ends the watch: watch it again
        if str(CONFIG_PATH) not in self._watcher.files() and CONFIG_PATH.exists():
            self._watcher.addPath(str(CONFIG_PATH))

    # --- run ---

    def _browse(self, edit: QLineEdit, title: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, title, edit.text())
        if folder:
            edit.setText(str(Path(folder)))

    @Slot()
    def _start(self) -> None:
        self._show_config()
        try:
            config = load_config()  # read again at every start
        except Exception as error:
            self._message(f"config.yaml not readable – run not started:\n{error}")
            return
        self._gui_settings = config.to_dict()["gui"]
        input_path = Path(self.input_edit.text().strip())
        output_dir = Path(self.output_edit.text().strip())

        self._last_result = None
        self._total = 0
        self.viewer_button.setEnabled(False)
        self.log_view.clear()
        self.progress_bar.setRange(0, 0)  # busy until the first progress report
        self.status_label.setText("starting…")
        self._set_running(True)

        self._thread = QThread(self)
        self._worker = _RunWorker(config, input_path, output_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        for signal in (self._worker.finished, self._worker.failed):
            signal.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    @Slot()
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel.set()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("cancelling after the current image…")

    @Slot(int, int, str, str)
    def _on_progress(self, index: int, total: int, stage: str, name: str) -> None:
        self._total = total
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(index)
        if self._worker is not None and self._worker.cancel.is_set():
            return  # keep the "cancelling" message
        self.status_label.setText(f"image {min(index + 1, total)} of {total}: {name} – {stage}" if name else stage)

    @Slot(object)
    def _on_finished(self, result: RunResult) -> None:
        self._last_result = result
        total = max(self._total, result.succeeded + result.failed)
        state = "cancelled" if result.cancelled else "done"
        self.status_label.setText(f"{state}: {result.succeeded} of {total} images succeeded, "
                                  f"{result.failed} failed ({result.duration_s:.1f} s) → {result.results_dir}")
        if not result.cancelled:
            self.progress_bar.setValue(self.progress_bar.maximum())
        for error in result.errors:
            self.log_view.appendPlainText(f"FAILED {error.image}: {error.stage} / {error.method}: {error.message}")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(message.strip().splitlines()[-1])
        self.log_view.appendPlainText(message)

    @Slot()
    def _on_thread_done(self) -> None:
        self._thread.deleteLater()
        self._worker.deleteLater()
        self._thread = self._worker = None
        self._set_running(False)
        if self._last_result is not None:
            self.viewer_button.setEnabled(True)
            if self.open_viewer_check.isChecked():
                self._open_viewer()

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self._viewer is None)
        self.start_button.setToolTip("close the viewer first – a new run overwrites its results"
                                     if self._viewer is not None else "")
        self.cancel_button.setEnabled(running)
        self.input_edit.setEnabled(not running)
        self.output_edit.setEnabled(not running)

    # --- viewer ---

    @Slot()
    def _open_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.raise_()
            self._viewer.activateWindow()
            return
        try:
            results_dir = self._last_result.results_dir
            try:
                corrections = Corrections(results_dir)     # manual corrections (PLAN.md 7.3)
            except Exception:
                corrections = None
                log.warning("corrections not available: %s", traceback.format_exc())
            self._viewer = ViewerWindow(results_dir, self._gui_settings, corrections, parent=self)
        except Exception:
            self._message(f"cannot open the viewer:\n{traceback.format_exc()}")
            return
        self._viewer.destroyed.connect(self._on_viewer_closed)
        self._viewer.show()
        self._set_running(False)  # Start is locked while the viewer is open

    @Slot()
    def _on_viewer_closed(self) -> None:
        self._viewer = None
        self._set_running(self._thread is not None)

    def _message(self, text: str) -> None:
        QMessageBox.warning(self, "Pore Detection", text)

    # --- closing ---

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None:
            answer = QMessageBox.question(self, "Pore Detection",
                                          "A run is in progress. Cancel it after the current image and close?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.cancel.set()
            self.status_label.setText("cancelling after the current image…")
            self._thread.wait()  # finished images stay on disk
        log.removeHandler(self._log_handler)
        super().closeEvent(event)
