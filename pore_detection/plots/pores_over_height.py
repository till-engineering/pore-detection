"""Pores over the height of the sample: relative stress at each pore over its height zeta.

Like the stress diagram of the Porenkonzept: zeta upwards (0 = inner, 1 = outer edge), stress to
the right. Every pore is a point at its centroid, the size shows its area; the points lie on the
stress profile, so the pores in the highly stressed parts stand out.
"""

import numpy as np

from ..core.plotting import COLORS
from ..core.registry import register
from ..core.step import Plot

MARKER_PX2 = (6, 300)       # marker area of the smallest / largest pore (points²)


@register("pores_over_height")
class PoresOverHeight(Plot):
    """Stress at the pore centroid over the relative height, marker size = pore area."""

    params = {}
    pore_columns = ["zeta", "stress_centroid", "area_px2"]

    def plot(self, ax, image, pores):
        area = pores["area_px2"].to_numpy(dtype=float)
        if len(area):
            span = np.log(area.max()) - np.log(area.min()) or 1.0
            size = MARKER_PX2[0] + (MARKER_PX2[1] - MARKER_PX2[0]) * (np.log(area) - np.log(area.min())) / span
            ax.scatter(pores["stress_centroid"], pores["zeta"], s=size, color=COLORS[0], alpha=0.45,
                       edgecolors="none", label="pore (size = area)")
        ax.axhline(0.5, color=COLORS[4], linestyle="--", linewidth=1)
        ax.text(1.0, 0.75, "outer zone", ha="right", va="center", color=COLORS[4])
        ax.text(1.0, 0.25, "inner zone", ha="right", va="center", color=COLORS[4])
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1)
        ax.set_xlabel("relative stress g at the pore centroid")
        ax.set_ylabel("relative height ζ (0 inner, 1 outer)")
        ax.set_title(f"Pores over the height ({len(area)} pores)")
