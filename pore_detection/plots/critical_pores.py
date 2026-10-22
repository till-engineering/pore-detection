"""Critical pores (Pareto): the pores sorted by their contribution A g^q to the pore index phi_q.

Bars: the `top` pores with the largest contribution (pore number as in the viewer's pore list),
colored by zone (outer / inner); line on a second axis: cumulative share of the sum over all pores
in %. Shows whether a few large pores near the surface decide phi_q or many small ones – and which
pores to check in the viewer (artefacts count as well).

Right of it the sample (straightened, outer surface on top, a little embedding resin around it)
with the same pores numbered (thin line to the pore, size by contribution, the 3 largest red) – so one sees where the pores that matter lie. Needs the
sample frame; without it, and over the whole ID, only the Pareto part is drawn.

The contributions come from the evaluation module pore_index (weighted_area_q…_px2, computed with
its pore_stress); q must be one of its q_values. Over the whole ID: the critical pores of all
images together, labelled image/pore.

all_q: true (optional) also saves the plot for every q of pore_index (every weighted_area_q…
column) in plots/all_q/<name>_q<q>.png – only saved, the viewer shows the plot with q.
"""

import re

import numpy as np

from ..core.plotting import COLORS, MissingData
from ..core.registry import register
from ..core.step import Plot

OUTER, INNER = COLORS[1], COLORS[0]
LABEL, LABEL_TOP = "#ffff00", "#ff1a1a"     # pore numbers on the sample: bright yellow, the 3 largest red
LABEL_PT = (6.0, 9.0)                       # font size of the smallest / largest contribution
LEADER_PT = (12, 20, 30)                    # tried lengths of the line from the number to the pore
LEADER_LW = 0.6
TURNS_DEG = (0, 30, -30, 60, -60, 90, -90, 135, -135, 180)     # tried directions, relative to "outwards"
LABEL_ROOM = 0.07                           # free border around the sample picture, share of its long side
ALL_Q_DIR = "all_q"                         # subfolder of plots/ for the plots of every q (all_q: true)


@register("critical_pores")
class CriticalPores(Plot):
    """Pareto of the pore contributions A g^q to phi_q, colored by zone."""

    params = {"q": "exponent q (one of the q_values of pore_index)",
              "top": "number of pores shown as bars",
              "all_q": "true = also save the plot for every q of pore_index in plots/all_q/ (not shown)"}
    pore_columns = ["pore_id", "zeta"]
    needs_picture = True
    size_in = (11.0, 4.4)           # room for the sample on the right

    def plot(self, ax, image, pores, picture=None):
        weighted = pores[self._column(pores)].to_numpy(dtype=float)
        labels = [str(int(i)) for i in pores["pore_id"]]
        zeta = pores["zeta"].to_numpy(dtype=float)
        if picture is not None:
            grid = ax.figure.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.32)
            ax.set_subplotspec(grid[0])
            self._sample(ax.figure.add_subplot(grid[1]), picture, pores, weighted, zeta)
        summary = self._draw(ax, weighted, labels, zeta, "pore")
        ax.set_title(f"Critical pores for Φ_q, q = {self.q:g} ({len(weighted)} pores){summary}")

    def _sample(self, ax, picture, pores, weighted, zeta) -> None:
        """The straightened sample, the top pores numbered next to them with a thin line to the pore
        (the pore itself stays free): number size by contribution, the 3 largest in red. Each number
        takes the first place (direction, line length) where it overlaps no number placed before and
        no numbered pore – the largest contributions first."""
        ax.imshow(picture.rgb, interpolation="antialiased")
        height, width = picture.rgb.shape[:2]
        pad = LABEL_ROOM * max(width, height)                           # room for the numbers at the edges
        ax.set_xlim(-pad, width + pad)
        ax.set_ylim(height + pad, -pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.tick_params(which="both", top=False, right=False, bottom=False, left=False)
        ax.set_title("where they lie (outer surface on top)")
        position = picture.pores.set_index("pore_id")
        centre = np.array(picture.rgb.shape[1::-1]) / 2
        if len(weighted) == 0:                                          # no pores: only the sample
            return
        top_share = weighted.max()
        shown = [(rank, i, int(pores["pore_id"].iloc[i]))
                 for rank, i in enumerate(np.argsort(weighted)[::-1][:self.top])]
        shown = [(rank, i, pore_id) for rank, i, pore_id in shown
                 if weighted[i] > 0 and pore_id in position.index]

        # placement in points of the figure (y up): the aspect of the picture is fixed first
        ax.apply_aspect()
        per_pt = ax.figure.dpi / 72
        to_pt = lambda xy: ax.transData.transform(xy) / per_pt         # noqa: E731
        taken = [_box(to_pt(position.loc[pore_id, ["x", "y"]].to_numpy(dtype=float)), 4, 4)
                 for _, _, pore_id in shown]                            # the numbered pores stay free
        for rank, i, pore_id in shown:
            x, y = position.loc[pore_id, ["x", "y"]]
            size = LABEL_PT[0] + (LABEL_PT[1] - LABEL_PT[0]) * np.sqrt(weighted[i] / top_share)
            text = str(pore_id)
            half = (0.31 * size * len(text) + 1, 0.4 * size + 1)          # half width / height of the number
            away = np.array([x, y]) - centre                           # labels point outwards (y down)
            away = away / (np.linalg.norm(away) or 1.0) * (1, -1)
            pore = to_pt((x, y))
            offset = None
            for length in LEADER_PT:
                for turn in np.radians(TURNS_DEG):
                    c, s = np.cos(turn), np.sin(turn)
                    candidate = length * np.array([c * away[0] - s * away[1], s * away[0] + c * away[1]])
                    box = _box(pore + candidate, *half)
                    if not any(_overlap(box, other) for other in taken):
                        offset = candidate
                        break
                if offset is not None:
                    break
            if offset is None:                                         # no free place: outwards anyway
                offset = LEADER_PT[0] * away
            taken.append(_box(pore + offset, *half))
            color = LABEL_TOP if rank < 3 else LABEL
            ax.annotate(text, (x, y), xytext=tuple(offset), textcoords="offset points",
                        ha="center", va="center", fontsize=size, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=LEADER_LW, shrinkA=1, shrinkB=2))

    def plot_id(self, ax, images, pores):
        weighted = pores[self._column(pores)].to_numpy(dtype=float)
        short = {name: name.split("_")[-1] for name in images["image"]}
        labels = [f"{short.get(name, name)}/{int(i)}" for name, i in zip(pores["image"], pores["pore_id"])]
        summary = self._draw(ax, weighted, labels, pores["zeta"].to_numpy(dtype=float), "image/pore")
        ax.set_title(f"Critical pores for Φ_q, q = {self.q:g} – {len(images)} images, {len(weighted)} pores"
                     f"{summary}")

    def extra_plots(self, pores) -> dict:
        """all_q: this plot for every q of pore_index (found from the weighted_area_q…_px2 columns)."""
        if not getattr(self, "all_q", False):
            return {}
        found = re.findall(r"^weighted_area_q([0-9_]+)_px2$", "\n".join(pores.columns), flags=re.MULTILINE)
        result = {}
        for text in sorted(found, key=lambda t: float(t.replace("_", "."))):
            q = float(text.replace("_", "."))
            result[f"{ALL_Q_DIR}/{self.name}_q{text}"] = type(self)(**{**self._params, "q": q, "all_q": False})
        return result

    def _column(self, pores) -> str:
        column = "weighted_area_q" + f"{self.q:g}".replace(".", "_") + "_px2"
        if column not in pores.columns:
            raise MissingData(f"{column} missing (pore_index not active, or q = {self.q:g} not in its q_values)")
        return column

    def _draw(self, ax, weighted, labels, zeta, what) -> str:
        """Bars + cumulative line; returns the summary line for the title ("top 5 = 72 % · top 20 = 98 %")."""
        total = weighted.sum()
        if len(weighted) == 0 or total <= 0:
            ax.text(0.5, 0.5, "No pore contributes", ha="center", va="center", transform=ax.transAxes)
            return ""
        order = np.argsort(weighted)[::-1]
        shown = order[:self.top]
        share = 100 * weighted[shown] / total
        x = np.arange(len(shown))
        colors = [OUTER if zeta[i] >= 0.5 else INNER for i in shown]
        ax.bar(x, share, color=colors, edgecolor="white", linewidth=0.5)
        ax.bar([0], [0], color=OUTER, label="outer zone")              # legend entries only
        ax.bar([0], [0], color=INNER, label="inner zone")
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


def _box(centre, half_width, half_height) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) around a point."""
    x, y = centre
    return x - half_width, y - half_height, x + half_width, y + half_height


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
