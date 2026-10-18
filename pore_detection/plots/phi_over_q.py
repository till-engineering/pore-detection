"""Pore index over the exponent q: phi_q and phi_eff_q of the image (module pore_index).

Over the whole ID (plot_id): one color per image, phi_q solid and phi_eff_q dashed.
"""

import re

from ..core.plotting import COLORS, MissingData
from ..core.registry import register
from ..core.step import Plot

PHI = re.compile(r"^phi_q(\d+(?:_\d+)?)$")      # phi_q1, phi_q1_5, …


@register("phi_over_q")
class PhiOverQ(Plot):
    """phi_q and phi_eff_q over q, porosity as reference line."""

    params = {}

    def plot_id(self, ax, images, pores):
        """One color per image: phi_q solid, phi_eff_q dashed (no phi over the whole ID yet)."""
        qs = _q_columns(images.columns)
        if not qs:
            raise MissingData("no phi_q values (evaluation module pore_index not active?)")
        q = [value for value, _ in qs]
        for i, row in enumerate(images.to_dict("records")):
            color = COLORS[i % len(COLORS)]
            ax.plot(q, [100 * row[f"phi_q{name}"] for _, name in qs], "o-", color=color, markersize=4,
                    label=row["image"])
            ax.plot(q, [100 * row[f"phi_eff_q{name}"] for _, name in qs], "s--", color=color, markersize=3)
        ax.set_xscale("log")
        ax.set_xticks(q, [f"{v:g}" for v in q])
        ax.minorticks_off()
        ax.set_xlabel("q")
        ax.set_ylabel("pore index (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Pore index over q – {len(images)} images (solid Φ_q, dashed Φ_eff,q)")
        ax.legend(fontsize=7.5)

    def plot(self, ax, image, pores):
        qs = _q_columns(name for name in image if image[name] is not None)
        if not qs:
            raise MissingData("no phi_q values (evaluation module pore_index not active?)")
        q = [value for value, _ in qs]
        phi = [100 * image[f"phi_q{name}"] for _, name in qs]
        phi_eff = [100 * image[f"phi_eff_q{name}"] for _, name in qs]
        ax.plot(q, phi, "o-", color=COLORS[0], label="Φ_q")
        ax.plot(q, phi_eff, "s--", color=COLORS[1], label="Φ_eff,q")
        if image.get("porosity_percent") is not None:
            ax.axhline(image["porosity_percent"], color=COLORS[4], linestyle=":", label="porosity")
        ax.set_xscale("log")
        ax.set_xticks(q, [f"{v:g}" for v in q])
        ax.minorticks_off()
        ax.set_xlabel("q")
        ax.set_ylabel("pore index (%)")
        ax.set_ylim(bottom=0)
        ax.set_title("Pore index over q")
        ax.legend()


def _q_columns(columns) -> list[tuple[float, str]]:
    """(q, name part) of all phi_q columns, sorted by q: phi_q1_5 -> (1.5, "1_5")."""
    return sorted((float(match.group(1).replace("_", ".")), match.group(1))
                  for match in map(PHI.match, columns) if match)
