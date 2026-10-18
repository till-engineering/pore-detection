"""Pore area distribution: number of pores per area class – look and computation as the histogram of
Pore_Detection_V2 (<image>_histogramm.png), so it deliberately overrides parts of core.plotting.STYLE
(open frame, horizontal grid, ticks outside, legend without frame).

Pore sizes are right-skewed (few large cavities, very many small pores), hence:
- logarithmic classes on smooth quarter decades; the number of classes follows Freedman-Diaconis on
  the logarithms (Sturges assumes a normal distribution and gives too few classes for skewed data)
- a kernel density curve (Silverman bandwidth, on the logarithms) on the same count axis as the
  bars: it shows what is structure and what is only the choice of class edges
- D50 / D90 (quantiles, robust against a single large cavity) and the mean as lines
Area in µm² with a scale, else px². Variant "normalized": share of the pores in % instead of the
number (the bars add up to 100 %), for comparing images with different pore counts.
"""

import math

import numpy as np

from ..core.registry import register
from ..core.step import Plot

# colors of V2
RED, INK, ACCENT, MUTED, GRID = "#ff1a1a", "#141820", "#0f6d8c", "#5d6775", "#d5dae1"
MIN_CLASSES, MAX_CLASSES = 5, 24
CURVE_POINTS = 96
RUG_HEIGHT = 0.03           # tick per pore at the bottom, share of the highest bar


@register("pore_areas")
class PoreAreas(Plot):
    """Histogram of the pore areas (log classes) with density curve, D50, D90 and mean (as in V2)."""

    params = {}
    pore_columns = ["area_px2"]
    variant = "normalized"

    def plot(self, ax, image, pores, variant=""):
        use_um = "area_um2" in pores.columns and pores["area_um2"].notna().any()
        area = pores["area_um2" if use_um else "area_px2"].to_numpy(dtype=float)
        area = np.sort(area[area > 0])
        unit = "µm²" if use_um else "px²"
        _v2_look(ax)
        ax.set_title(f"Pore size distribution – {image.get('image', '')}", fontsize=10)
        if len(area) == 0:
            ax.text(0.5, 0.5, "No pores to evaluate", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.minorticks_off()
            return

        edges = class_edges(area, class_count(area))
        counts, _ = np.histogram(area, bins=edges)
        scale = 100.0 / len(area) if variant else 1.0       # normalized: share of the pores in %
        heights = counts * scale
        top = heights.max()
        ax.bar(edges[:-1], heights, width=np.diff(edges), align="edge", color=RED, edgecolor="white",
               linewidth=0.6, label="counted", zorder=2)
        curve = density_curve(area, edges)
        if curve is not None:
            ax.plot(curve[0], curve[1] * scale, color=INK, alpha=0.72, linewidth=1.6, label="density", zorder=3)
        ax.vlines(area, 0, top * RUG_HEIGHT, color=INK, alpha=0.38, linewidth=0.7, zorder=3)

        d10, d50, d90 = np.percentile(area, [10, 50, 90])
        mean = float(area.mean())
        for name, value, style, color in (("D50", d50, (0, (3, 3)), INK), ("D90", d90, (0, (3, 3)), INK),
                                          ("Mean", mean, "-", ACCENT)):
            ax.axvline(value, color=color, linestyle=style, linewidth=1.2, alpha=0.8, zorder=4)
            ax.annotate(name, (value, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                        textcoords="offset points", fontsize=8, color=color)

        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0, top * 1.08)
        if not variant:
            ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel(f"Pore area A [{unit}]")
        ax.set_ylabel("Share of pores [%]" if variant else "Number n")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
        ax.annotate(f"n = {len(area)}   Mean {mean:.6g}   D10 {d10:.6g}   D50 {d50:.6g}   D90 {d90:.6g} {unit}",
                    (0.5, 0), xycoords="axes fraction", xytext=(0, -34), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color=MUTED)


def _v2_look(ax) -> None:
    """matplotlib defaults as V2 used them, on top of core.plotting.STYLE."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.yaxis.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def class_count(values: np.ndarray) -> int:
    """Number of log classes after Freedman-Diaconis (width 2 IQR n^(-1/3) on the logarithms)."""
    if len(values) < 4:
        return max(1, len(values))
    logs = np.log10(values)
    span = logs.max() - logs.min()
    if span <= 0:
        return 1
    q1, q3 = np.percentile(logs, [25, 75])
    if q3 - q1 <= 0:                # more than half of the values equal: square root rule
        return max(4, min(20, round(math.sqrt(len(logs)))))
    width = 2.0 * (q3 - q1) * len(logs) ** (-1 / 3)
    return max(MIN_CLASSES, min(MAX_CLASSES, math.ceil(span / width)))


def class_edges(values: np.ndarray, classes: int) -> np.ndarray:
    """Log class edges, widened to smooth quarter decades (round axis labels)."""
    low, high = values.min(), values.max()
    if high <= low:
        high = low * 1.05
    lo = math.floor(math.log10(low) * 4) / 4
    hi = math.ceil(math.log10(high) * 4) / 4
    return 10 ** np.linspace(lo, hi, classes + 1)


def density_curve(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Gaussian kernel density on the logarithms, scaled to the count axis (n · class width).

    None with fewer than 5 values or no spread.
    """
    logs = np.log10(values)
    n = len(logs)
    if n < 5 or len(edges) < 2:
        return None
    spread = logs.std(ddof=1)
    q1, q3 = np.percentile(logs, [25, 75])
    robust = (q3 - q1) / 1.349
    sigma = min(spread, robust) if robust > 0 else spread
    if sigma <= 0:
        return None
    h = 0.9 * sigma * n ** (-1 / 5)                 # Silverman
    lo, hi = math.log10(edges[0]), math.log10(edges[-1])
    x = np.linspace(lo, hi, CURVE_POINTS)
    # density 1/(n h sqrt(2 pi)) sum exp(-u²/2), times n · class width; n cancels
    factor = (hi - lo) / (len(edges) - 1) / (h * math.sqrt(2 * math.pi))
    height = np.empty(CURVE_POINTS)
    for i, point in enumerate(x):
        u = (point - logs) / h
        u = u[np.abs(u) < 5.0]                       # beyond that the contribution is negligible
        height[i] = factor * np.exp(-0.5 * u * u).sum()
    return 10 ** x, height
