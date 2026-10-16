"""GUI: python gui_main.py – opens the launcher (the viewer is opened from there)."""

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.launcher.window import LauncherWindow
from pore_detection import load_config

COLOR_SCHEMES = {"system": Qt.ColorScheme.Unknown, "light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}


def main() -> int:
    app = QApplication(sys.argv)
    try:
        app.styleHints().setColorScheme(COLOR_SCHEMES[load_config().gui.theme])
    except Exception:
        pass  # broken config: the launcher opens anyway and shows it
    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
