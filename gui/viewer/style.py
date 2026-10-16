"""Look of the viewer (colors, cards and type after the Pore_Detection_V2 viewer), light and dark."""

LIGHT = {
    "ground": "#eceff3", "surface": "#ffffff", "surface2": "#f5f7f9", "sunken": "#e3e7ec",
    "ink": "#141820", "muted": "#5d6775", "faint": "#8a94a2", "line": "#d5dae1",
    "accent": "#0f6d8c", "accent_soft": "#d9edf4",
}
DARK = {
    "ground": "#10131a", "surface": "#181d26", "surface2": "#1f2530", "sunken": "#0b0e13",
    "ink": "#e7ecf2", "muted": "#96a1b0", "faint": "#6b7684", "line": "#2c3543",
    "accent": "#48aed0", "accent_soft": "#11323f",
}
PORE = "#ff1a1a"
EMBEDDING = "#7c5cd6"
REJECTED = "#b8722b"
REMOVED = "#d63370"
ADDED = "#0c8599"

# first installed family wins
SANS = '"IBM Plex Sans", "Segoe UI", sans-serif'
MONO = '"IBM Plex Mono", "Cascadia Mono", Consolas, monospace'


def stylesheet(c: dict) -> str:
    return f"""
    QWidget {{ color: {c['ink']}; font-family: {SANS}; font-size: 10pt; }}
    #page, #pageContent, QMainWindow {{ background: {c['ground']}; border: none; }}

    #header {{ background: {c['surface']}; border-bottom: 1px solid {c['line']}; }}
    #title {{ font-weight: 600; font-size: 11.5pt; }}
    #imageName, #position {{ color: {c['muted']}; font-family: {MONO}; font-size: 9.5pt; }}
    #navButton {{ min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px; padding: 0;
                  font-size: 12pt; color: {c['muted']}; }}
    #chip {{ border: 1px solid {c['line']}; border-radius: 11px; padding: 2px 10px;
             background: {c['surface2']}; font-size: 9pt; }}
    #arrow {{ color: {c['faint']}; }}

    #card {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 10px; }}
    #cardTitle {{ color: {c['muted']}; font-size: 8pt; font-weight: 600; letter-spacing: 1px;
                  padding: 10px 14px; border-bottom: 1px solid {c['line']}; }}
    #sectionLabel {{ color: {c['faint']}; font-size: 7.5pt; font-weight: 600; letter-spacing: 1px; }}
    #hint {{ color: {c['muted']}; font-size: 8.5pt; }}
    #mono {{ font-family: {MONO}; font-size: 9pt; }}

    #valueStrip {{ background: {c['surface']}; border-bottom: 1px solid {c['line']}; }}
    #valueStrip QLabel {{ color: {c['muted']}; font-size: 9pt; }}
    #stageBody, QGraphicsView {{ background: {c['surface']}; border: none; }}
    #tools {{ background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 9px; }}
    #toolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px;
                   min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; padding: 0;
                   font-size: 12pt; color: {c['ink']}; }}
    #toolButton:hover {{ background: {c['surface2']}; border-color: {c['line']}; color: {c['accent']}; }}
    #toolButton:disabled {{ color: {c['faint']}; background: transparent; border-color: transparent; }}
    #toolButton:checked {{ color: #ffffff; border-color: transparent; font-weight: 600; }}
    #toolButton[tool="restore"]:checked {{ background: {ADDED}; }}
    #toolButton[tool="remove"]:checked, #toolButton[tool="area"]:checked {{ background: {REMOVED}; }}
    #toolButton[tool="draw"]:checked {{ background: {c['accent']}; }}
    #toolLine {{ background: {c['line']}; min-height: 1px; max-height: 1px; }}
    #loupeColumn {{ background: {c['surface']}; border-left: 1px solid {c['line']}; }}
    #infoBar {{ background: {c['surface2']}; border-top: 1px solid {c['line']}; }}
    #info, #infoValue {{ color: {c['muted']}; font-family: {MONO}; font-size: 8.5pt; }}

    #tileGrid {{ background: {c['line']}; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; }}
    #tile {{ background: {c['surface']}; }}
    #tileValue {{ font-family: {MONO}; font-size: 13pt; font-weight: 500; }}
    #tileLabel {{ color: {c['muted']}; font-size: 8.5pt; }}

    #layerRow {{ border-radius: 6px; padding: 0; }}
    #layerRow[on="true"] {{ background: {c['accent_soft']}; }}
    #layerCount {{ color: {c['faint']}; font-family: {MONO}; font-size: 8.5pt; }}

    QPushButton {{ background: {c['surface2']}; border: 1px solid {c['line']}; border-radius: 6px;
                   padding: 4px 11px; }}
    QPushButton:hover {{ border-color: {c['accent']}; color: {c['accent']}; }}
    QPushButton:disabled {{ color: {c['faint']}; border-color: {c['line']}; }}
    QPushButton:checked {{ background: {c['accent_soft']}; border-color: {c['accent']}; color: {c['accent']};
                           font-weight: 600; }}
    QCheckBox {{ spacing: 0; }}
    QTreeWidget, QTableView, QPlainTextEdit {{ background: {c['surface']}; border: none;
                   alternate-background-color: {c['surface2']}; }}
    QTreeWidget {{ border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; }}
    QTreeWidget::item {{ padding: 6px 4px; border-bottom: 1px solid {c['line']}; }}
    QTreeWidget::item:selected, QTableView::item:selected {{ background: {c['accent_soft']}; color: {c['ink']}; }}
    QHeaderView::section {{ background: {c['surface2']}; color: {c['muted']}; border: none;
                            border-bottom: 1px solid {c['line']}; padding: 4px; }}
    QSlider::groove:horizontal {{ height: 4px; background: {c['line']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {c['accent']}; width: 12px; margin: -5px 0; border-radius: 6px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {c['line']}; border-radius: 3px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['faint']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {c['line']}; border-radius: 3px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QToolTip {{ background: {c['surface']}; color: {c['ink']}; border: 1px solid {c['line']}; padding: 4px; }}
    QDialog {{ background: {c['surface']}; }}
    """
