"""Tool icons of the viewer and the check window: Lucide line icons (lucide-static 0.460.0, ISC,
icons/LICENSE_lucide.txt), coloured at runtime so they follow the theme and the active tool."""

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICON_DIR = Path(__file__).parent / "icons"
ICON_PX = 18                    # drawn size inside the 30 px tool button
RENDER_SCALE = 3                # rendered larger, Qt scales down: sharp on high-DPI screens too


def _pixmap(name: str, color: str) -> QPixmap:
    svg = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8").replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    size = ICON_PX * RENDER_SCALE
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pixmap


def icon(name: str, color: str, disabled: str, checked: str | None = None) -> QIcon:
    """Lucide icon `name` (file in icons/) in `color`, greyed when disabled, `checked` while the
    button is checked (on the coloured background of the active tool)."""
    result = QIcon()
    result.addPixmap(_pixmap(name, color), QIcon.Mode.Normal, QIcon.State.Off)
    result.addPixmap(_pixmap(name, disabled), QIcon.Mode.Disabled, QIcon.State.Off)
    result.addPixmap(_pixmap(name, disabled), QIcon.Mode.Disabled, QIcon.State.On)
    result.addPixmap(_pixmap(name, checked or color), QIcon.Mode.Normal, QIcon.State.On)
    return result
