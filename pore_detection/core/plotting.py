"""Plots: draw the plot modules (stage "plots") into PNG images – fixed size and look for all.

A plot module only draws into the matplotlib Axes it gets (see core/step.py, Plot). Everything
else is done here: figure, look, checking the needed values, saving as PNG. matplotlib is
imported only when a plot is drawn, so runs without plots do not need it.

A plot that cannot be drawn (values missing, error in the plot) is skipped and reported; the
image itself does not fail – plots only show values that are already computed.
"""

import io
import logging

import pandas as pd

from .step import Plot

log = logging.getLogger("pore_detection")

SIZE_IN = (6.4, 4.0)        # figure size in inch; with DPI -> 960 x 600 px (sharp when scaled down)
DPI = 150
PAD_IN = 0.04               # white border around the plot when saved (tight)
# colors for plot modules: first = main series (accent of the viewer), then further series
COLORS = ("#0f6d8c", "#d9480f", "#5c940d", "#7c5cd6", "#868e96")
# scientific look: black frame on all four sides, no grid, ticks inside on all sides
STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal", "axes.labelsize": 9,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.color": "black", "ytick.color": "black", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.fontsize": 8, "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "figure.facecolor": "white", "savefig.facecolor": "white",
}


class MissingData(Exception):
    """Raised by a plot when values it needs are missing: the plot is skipped (not the image)."""


def available() -> bool:
    """matplotlib installed (needed as soon as plots are configured)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def render(plots: list[Plot], image: dict, pores: pd.DataFrame) -> dict[str, bytes]:
    """PNG bytes per plot name, in the order of `plots`; skipped plots are logged and left out."""
    if not plots:
        return {}
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure     # object API: no global pyplot state, safe in threads

    result = {}
    for step in plots:
        try:
            missing = [c for c in step.columns if _empty(image.get(c))]
            missing += [c for c in step.pore_columns if c not in pores.columns]
            if missing:
                raise MissingData(f"values missing: {', '.join(missing)}")
            with matplotlib.rc_context(STYLE):
                figure = Figure(figsize=SIZE_IN, dpi=DPI)
                ax = figure.add_subplot()
                step.plot(ax, image, pores)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=PAD_IN)
            result[step.name] = buffer.getvalue()
        except MissingData as reason:
            log.warning("%s: plot %s skipped – %s", image.get("image"), step.name, reason)
        except Exception as error:
            log.warning("%s: plot %s skipped – %s: %s", image.get("image"), step.name, type(error).__name__, error)
    return result


def _empty(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)
