"""Critical pores, split: the Pareto of critical_pores, every bar split into area and stress share.

Same bars as critical_pores (height = share of the pore in Σ A g^q, same order, same cumulative
line, same sample picture on the right). In log space the contribution of a pore to phi_q is a sum
of two terms, both <= 0:

    log10(A g^q / A_Mess) = log10(A / A_Mess) + log10(g^q)
                            area term            stress term

A in px², A_Mess = sample area of the image in px² (pore_index: outer + inner zone), g^q = A g^q / A
(the effective stress of the pore for the pore_stress of pore_index, also for pores split at the
zone boundary). Both terms say by how many decades the contribution lies below 1 (the whole sample
one pore at the surface); the bar is split in the ratio of their sizes:

    lower part (area)    |log10(A / A_Mess)|   large = the pore is small compared to the sample
    upper part (stress)  |log10(g^q)|          large = the pore lies far from the surface

The zone (outer / inner) is the frame around the bar. Over the whole ID: every pore with the sample
area of its own image.
"""

import numpy as np

from ..core.registry import register
from .critical_pores import INNER, OUTER, CriticalPores

AREA_FILL, STRESS_FILL = "#ced4da", "#495057"   # light / dark gray: the zone colors stay for the frames
FRAME_LW = 1.4
MEASURING = ("sample_area_outer_px2", "sample_area_inner_px2")


@register("critical_pores_split")
class CriticalPoresSplit(CriticalPores):
    """Pareto of A g^q as critical_pores, every bar split in log space into area and stress share."""

    columns = list(MEASURING)
    pore_columns = ["pore_id", "zeta", "area_px2"]

    def plot(self, ax, image, pores, picture=None):
        weighted = pores[self._column(pores)].to_numpy(dtype=float)
        labels = [str(int(i)) for i in pores["pore_id"]]
        zeta = pores["zeta"].to_numpy(dtype=float)
        measuring = np.full(len(pores), float(image[MEASURING[0]]) + float(image[MEASURING[1]]))
        if picture is not None:
            grid = ax.figure.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.32)
            ax.set_subplotspec(grid[0])
            self._sample(ax.figure.add_subplot(grid[1]), picture, pores, weighted, zeta)
        summary = self._draw_split(ax, weighted, pores["area_px2"].to_numpy(dtype=float), measuring,
                                   labels, zeta, "pore")
        ax.set_title(f"Critical pores for Φ_q, q = {self.q:g}, area | stress ({len(weighted)} pores){summary}")

    def plot_id(self, ax, images, pores):
        weighted = pores[self._column(pores)].to_numpy(dtype=float)
        short = {name: name.split("_")[-1] for name in images["image"]}
        labels = [f"{short.get(name, name)}/{int(i)}" for name, i in zip(pores["image"], pores["pore_id"])]
        sample_area = dict(zip(images["image"], images[MEASURING[0]] + images[MEASURING[1]]))
        measuring = pores["image"].map(sample_area).to_numpy(dtype=float)
        summary = self._draw_split(ax, weighted, pores["area_px2"].to_numpy(dtype=float), measuring,
                                   labels, pores["zeta"].to_numpy(dtype=float), "image/pore")
        ax.set_title(f"Critical pores for Φ_q, q = {self.q:g}, area | stress – {len(images)} images, "
                     f"{len(weighted)} pores{summary}")

    def _draw_split(self, ax, weighted, area, measuring, labels, zeta, what) -> str:
        """Bars (split, zone as frame) + cumulative line; returns the summary line for the title."""
        total = weighted.sum()
        if len(weighted) == 0 or total <= 0:
            ax.text(0.5, 0.5, "No pore contributes", ha="center", va="center", transform=ax.transAxes)
            return ""
        order = np.argsort(weighted)[::-1]
        shown = order[:self.top]
        share = 100 * weighted[shown] / total
        x = np.arange(len(shown))

        # log space: both terms <= 0, the bar is split in the ratio of their sizes
        area_term = -np.log10(area[shown] / measuring[shown])
        stress_term = np.maximum(-np.log10(weighted[shown] / area[shown]), 0.0)    # float noise at g = 1
        lower = share * area_term / np.maximum(area_term + stress_term, 1e-12)
        ax.bar(x, lower, color=AREA_FILL, linewidth=0, label="area |log A/A_Mess|")
        ax.bar(x, share - lower, bottom=lower, color=STRESS_FILL, linewidth=0, label="stress |log g^q|")
        frames = [OUTER if zeta[i] >= 0.5 else INNER for i in shown]
        ax.bar(x, share, fill=False, edgecolor=frames, linewidth=FRAME_LW)
        ax.bar([0], [0], fill=False, edgecolor=OUTER, linewidth=FRAME_LW, label="outer zone (frame)")
        ax.bar([0], [0], fill=False, edgecolor=INNER, linewidth=FRAME_LW, label="inner zone (frame)")
        ax.set_xticks(x, [labels[i] for i in shown], rotation=90, fontsize=7)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.set_xlim(-0.6, len(shown) - 0.4)
        ax.set_xlabel(f"{what} (sorted by contribution)")
        ax.set_ylabel("contribution to Σ A g^q (%)")
        ax.tick_params(axis="y", which="both", right=False)

        cumulative = np.cumsum(100 * weighted[order] / total)[:len(shown)]
        line = ax.twinx()
        line.plot(x, cumulative, "o-", color="black", markersize=3, linewidth=1.2, label="cumulative")
        line.set_ylim(0, 105)
        line.set_ylabel("cumulative share (%)")
        line.tick_params(axis="y", which="both", direction="in", right=True, left=False)
        line.tick_params(axis="x", which="both", top=False)
        handles = [h + l for h, l in zip(ax.get_legend_handles_labels(), line.get_legend_handles_labels())]
        ax.legend(*handles, fontsize=7.5, loc="center right")
        five = min(5, len(order))
        return f"\ntop {five} = {cumulative[five - 1]:.0f} %   ·   top {len(shown)} = {cumulative[-1]:.0f} %"
