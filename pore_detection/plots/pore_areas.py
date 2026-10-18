"""Pore area distribution: number of pores per area class – computation as the histogram of
Pore_Detection_V2 (<image>_histogramm.png), look as all plots (core.plotting.STYLE: black frame,
no grid, ticks inside).

Pore sizes are right-skewed (few large cavities, very many small pores), hence:
- logarithmic classes on smooth quarter decades; the number of classes follows Freedman-Diaconis on
  the logarithms (Sturges assumes a normal distribution and gives too few classes for skewed data)
- a kernel density curve (Silverman bandwidth, on the logarithms) on a second axis on the right: a
  true density per decade (pores per decade, normalized: % of the pores per decade; the area under
  the curve is n resp. 100 %) – it shows what is structure and what is only the choice of class edges
- D50 / D90 (quantiles, robust against a single large cavity) and the mean as lines
Area in µm² with a scale, else px². Variant "normalized": share of the pores in % instead of the
number (the bars add up to 100 %), for comparing images with different pore counts.

Over the whole ID (plot_id): only the density curves (per decade), one per image, over the same
classes (computed from all pores of the ID) – so the curves are directly comparable.
"""

import numpy as np

from ..core.plotting import COLORS, class_count, class_edges, density_curve
from ..core.registry import register
from ..core.step import Plot

# colors of V2
RED, INK, ACCENT, MUTED = "#ff1a1a", "#141820", "#0f6d8c", "#5d6775"
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
        handles = ax.get_legend_handles_labels()
        if curve is not None:
            # true density (per decade) has another unit than the bars: own axis on the right
            density = ax.twinx()
            factor = 100.0 if variant else len(area)        # % of the pores / pores per decade
            density.plot(curve[0], curve[1] * factor, color=INK, alpha=0.72, linewidth=1.6, label="density")
            density.set_ylim(0, (curve[1] * factor).max() * 1.08)
            density.set_ylabel("Density [% of pores per decade]" if variant else "Density [pores per decade]",
                               color=INK)
            ax.tick_params(axis="y", which="both", right=False)     # the right side belongs to the density
            density.tick_params(axis="y", which="both", direction="in", right=True, left=False)
            handles = [h + d for h, d in zip(handles, density.get_legend_handles_labels())]
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
        ax.legend(*handles, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
        ax.annotate(f"n = {len(area)}   Mean {mean:.6g}   D10 {d10:.6g}   D50 {d50:.6g}   D90 {d90:.6g} {unit}",
                    (0.5, 0), xycoords="axes fraction", xytext=(0, -34), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color=MUTED)


    def plot_id(self, ax, images, pores, variant=""):
        """Density curve of every image on the same log classes (from all pores of the ID)."""
        use_um = "area_um2" in pores.columns and pores["area_um2"].notna().any()
        column = "area_um2" if use_um else "area_px2"
        unit = "µm²" if use_um else "px²"
        everything = pores[column].to_numpy(dtype=float)
        everything = everything[everything > 0]
        ax.set_title(f"Pore size distribution – {len(images)} images", fontsize=10)
        if len(everything) == 0:
            ax.text(0.5, 0.5, "No pores to evaluate", ha="center", va="center", transform=ax.transAxes, color=MUTED)
            return
        edges = class_edges(everything, class_count(everything))       # the same classes for every image
        for i, name in enumerate(images["image"]):
            area = pores.loc[pores["image"] == name, column].to_numpy(dtype=float)
            area = area[area > 0]
            curve = density_curve(area, edges)
            if curve is None:
                continue
            factor = 100.0 if variant else len(area)        # % of the pores / pores per decade
            ax.plot(curve[0], curve[1] * factor, color=COLORS[i % len(COLORS)], linewidth=1.5,
                    label=f"{name} (n = {len(area)})")
        ax.set_xscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"Pore area A [{unit}]")
        ax.set_ylabel("Density [% of pores per decade]" if variant else "Density [pores per decade]")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.annotate(f"same range for all images, area under a curve = {'100 %' if variant else 'n'}", (0.5, 0), xycoords="axes fraction",
                    xytext=(0, -34), textcoords="offset points", ha="center", va="top", fontsize=8.5, color=MUTED)

