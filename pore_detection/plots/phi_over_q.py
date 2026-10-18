"""Pore index over the exponent q: phi_q and phi_eff_q of the image (module pore_index)."""

import re

from ..core.plotting import COLORS, MissingData
from ..core.registry import register
from ..core.step import Plot

PHI = re.compile(r"^phi_q(\d+(?:_\d+)?)$")      # phi_q1, phi_q1_5, …


@register("phi_over_q")
class PhiOverQ(Plot):
    """phi_q and phi_eff_q over q, porosity as reference line."""

    params = {}

    def plot(self, ax, image, pores):
        qs = sorted((float(match.group(1).replace("_", ".")), match.group(1))
                    for match in map(PHI.match, image) if match and image[match.group(0)] is not None)
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
