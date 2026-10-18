"""Pore size distribution: number of pores per equivalent diameter (logarithmic classes).

Equivalent diameter = diameter of a circle with the pore's area; in µm with a scale, else px.

Over the whole ID (plot_id): only the density curves (pores per decade, the area under a curve is
its number of pores), one per image, over the same range (from all pores of the ID) – directly
comparable.
"""

import numpy as np

from ..core.plotting import COLORS, class_count, class_edges, density_curve
from ..core.registry import register
from ..core.step import Plot

BINS_PER_DECADE = 8


@register("pore_sizes")
class PoreSizes(Plot):
    """Histogram of the equivalent pore diameters."""

    params = {}
    pore_columns = ["area_px2"]

    def plot(self, ax, image, pores):
        diameter, unit = _diameters(pores)
        if len(diameter):
            low, high = np.log10(diameter.min()), np.log10(diameter.max())
            bins = np.logspace(low, max(high, low + 0.1), max(2, int(np.ceil((high - low) * BINS_PER_DECADE)) + 1))
            ax.hist(diameter, bins=bins, color=COLORS[0], edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(f"equivalent diameter ({unit})")
        ax.set_ylabel("number of pores")
        ax.set_title(f"Pore sizes ({len(diameter)} pores)")

    def plot_id(self, ax, images, pores):
        """Density curve of every image on the same log classes (from all pores of the ID)."""
        everything, unit = _diameters(pores)
        ax.set_title(f"Pore sizes – {len(images)} images")
        if not len(everything):
            ax.text(0.5, 0.5, "No pores to evaluate", ha="center", va="center", transform=ax.transAxes)
            return
        edges = class_edges(everything, class_count(everything))
        for i, name in enumerate(images["image"]):
            diameter, _ = _diameters(pores[pores["image"] == name])
            curve = density_curve(diameter, edges)
            if curve is not None:
                ax.plot(curve[0], curve[1] * len(diameter), color=COLORS[i % len(COLORS)], linewidth=1.5,
                        label=f"{name} (n = {len(diameter)})")
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"equivalent diameter ({unit})")
        ax.set_ylabel("density (pores per decade)")
        ax.legend(fontsize=7.5, loc="upper right")


def _diameters(pores) -> tuple[np.ndarray, str]:
    """Equivalent diameters (> 0) of the pores and their unit (µm with a scale, else px)."""
    use_um = "area_um2" in pores.columns and pores["area_um2"].notna().any()
    area = pores["area_um2" if use_um else "area_px2"].to_numpy(dtype=float)
    area = area[area > 0]
    return np.sqrt(4 * area / np.pi), "µm" if use_um else "px"
