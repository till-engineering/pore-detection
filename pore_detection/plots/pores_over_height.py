"""Pores over the height of the sample: stress (field of pore_index) at each pore over its height zeta.

Like the stress diagram of the Porenkonzept: zeta upwards (0 = inner, 1 = outer edge), stress to
the right. Every pore is a point at its centroid, the size shows its area; the points lie on the
stress field (for a field that also depends on xi: a band), so the pores in the highly stressed
parts stand out. The stress axis runs from 0 to the highest stress of the pores (field values as
they are, e.g. relative or MPa).

Over the whole ID (plot_id): the pores of all images, one color per image (zeta is relative, so the
sections fit on one axis); marker sizes on one scale for all.
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
            ax.scatter(pores["stress_centroid"], pores["zeta"], s=_marker_sizes(area, area), color=COLORS[0],
                       alpha=0.45, edgecolors="none", label="pore (size = area)")
        _axes(ax, pores["stress_centroid"], _unit(image))
        ax.set_title(f"Pores over the height ({len(area)} pores)")

    def plot_id(self, ax, images, pores):
        everything = pores["area_px2"].to_numpy(dtype=float)
        for i, name in enumerate(images["image"]):
            part = pores[pores["image"] == name]
            if len(part):
                ax.scatter(part["stress_centroid"], part["zeta"],
                           s=_marker_sizes(part["area_px2"].to_numpy(dtype=float), everything),
                           color=COLORS[i % len(COLORS)], alpha=0.45, edgecolors="none", label=name)
        _axes(ax, pores["stress_centroid"], _unit(images.iloc[0].to_dict()) if len(images) else "")
        ax.set_title(f"Pores over the height – {len(images)} images, {len(pores)} pores")
        ax.legend(fontsize=7, loc="lower left", markerscale=0.6)


def _marker_sizes(area: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Marker area from the pore area, logarithmic between the smallest and largest of `reference`."""
    reference = reference[reference > 0]
    if not len(area) or not len(reference):
        return np.full(len(area), MARKER_PX2[0])
    low, high = np.log(reference.min()), np.log(reference.max())
    span = high - low or 1.0
    return MARKER_PX2[0] + (MARKER_PX2[1] - MARKER_PX2[0]) * (np.log(np.maximum(area, 1e-12)) - low) / span


def _unit(row: dict) -> str:
    """Field and unit of pore_index for the axis label ("" for results of older runs)."""
    field, unit = row.get("stress_field"), row.get("stress_unit")
    if not isinstance(field, str):
        return ""
    return f" – {field}" + (f" [{unit}]" if isinstance(unit, str) and unit not in ("", "-") else "")


def _axes(ax, stress, unit: str) -> None:
    ax.axhline(0.5, color=COLORS[4], linestyle="--", linewidth=1)
    ax.text(0.99, 0.75, "outer zone", transform=ax.get_yaxis_transform(), ha="right", va="center", color=COLORS[4])
    ax.text(0.99, 0.25, "inner zone", transform=ax.get_yaxis_transform(), ha="right", va="center", color=COLORS[4])
    stress = np.asarray(stress, dtype=float)
    stress = stress[np.isfinite(stress)]
    high = float(stress.max()) if len(stress) and stress.max() > 0 else 1.0
    ax.set_xlim(0, 1.02 * high)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"stress g at the pore centroid{unit}")
    ax.set_ylabel("relative height ζ (0 inner, 1 outer)")
