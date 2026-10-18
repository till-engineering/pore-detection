"""Pore size distribution: number of pores per equivalent diameter (logarithmic classes).

Equivalent diameter = diameter of a circle with the pore's area; in µm with a scale, else px.
"""

import numpy as np

from ..core.plotting import COLORS
from ..core.registry import register
from ..core.step import Plot

BINS_PER_DECADE = 8


@register("pore_sizes")
class PoreSizes(Plot):
    """Histogram of the equivalent pore diameters."""

    params = {}
    pore_columns = ["area_px2"]

    def plot(self, ax, image, pores):
        use_um = "area_um2" in pores.columns and pores["area_um2"].notna().any()
        area = pores["area_um2" if use_um else "area_px2"].to_numpy(dtype=float)
        area = area[area > 0]
        unit = "µm" if use_um else "px"
        if len(area):
            diameter = np.sqrt(4 * area / np.pi)
            low, high = np.log10(diameter.min()), np.log10(diameter.max())
            bins = np.logspace(low, max(high, low + 0.1), max(2, int(np.ceil((high - low) * BINS_PER_DECADE)) + 1))
            ax.hist(diameter, bins=bins, color=COLORS[0], edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(f"equivalent diameter ({unit})")
        ax.set_ylabel("number of pores")
        ax.set_title(f"Pore sizes ({len(area)} pores)")
