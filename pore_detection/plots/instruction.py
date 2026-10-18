"""INSTRUCTION – template for a plot (stage "plots", interface: core/step.py Plot).

Not active: @register below is commented out, so nothing of this file is used.
To make your own plot: copy this file (e.g. plots/my_plot.py), rename the class, uncomment
@register("...") and add it to config.yaml (the viewer shows the plots in this order):

  plots:
    - name: my_plot
      column: area_um2          # every key besides "name" becomes an attribute (self.column)

A plot only draws finished values – it knows column names, never an evaluation module:
  plot(ax, image, pores)
    ax:    matplotlib Axes to draw into (figure size, look, saving as PNG: core/plotting.py)
    image: dict = the row of this image in images.json (core columns + per_image values of all modules)
    pores: DataFrame = pores.json of this image (pore_id + per_pore columns of all modules)
  Saved as <image folder>/plots/<name>.png, shown in the viewer, drawn again after every correction.
  A plot never makes the image fail: missing values -> raise MissingData(reason) -> plot skipped, logged.
"""

import numpy as np

from ..core.plotting import COLORS, MissingData
from ..core.registry import register  # noqa: F401 – needed once @register is active
from ..core.step import Plot


# @register("example_plot")
class ExamplePlot(Plot):
    """Example: histogram of one per-pore column, with the porosity in the title."""

    params = {"column": "per-pore column to show, e.g. area_um2 or zeta"}
    columns = ["porosity_percent"]      # per-image values needed (missing -> plot skipped, logged)
    pore_columns = ["pore_id"]          # per-pore values needed (self.column is checked in plot())
    variant = "normalized"              # optional second view (+ checkbox in the viewer); "" = none
    needs_picture = True                # optional: plot() also gets the straightened sample picture
    size_in = None                      # optional: own figure size in inch, e.g. (11.0, 4.4); None = default

    def plot(self, ax, image, pores, variant="", picture=None):
        # variant: "" = normal view, else the variant ("normalized") – only passed if `variant` is set
        # picture: only passed with needs_picture (None without a sample frame), see _picture_values
        row = _image_values(image)
        if self.column not in pores.columns:
            raise MissingData(f"column {self.column} missing (module not active?)")
        values = pores[self.column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        weights = np.full(len(values), 100.0 / len(values)) if variant and len(values) else None
        ax.hist(values, bins=20, weights=weights, color=COLORS[0], edgecolor="white", linewidth=0.5)
        ax.set_xlabel(self.column)
        ax.set_ylabel("share of pores (%)" if variant else "number of pores")
        ax.set_title(f"{row['image']}: {len(values)} pores, porosity {row['porosity_percent']:.2f} %")
        if picture is not None:
            info = _picture_values(picture)
            ax.annotate(f"{len(info['pores'])} pores on the sample picture", (1, 1), xycoords="axes fraction",
                        xytext=(-4, -4), textcoords="offset points", ha="right", va="top", fontsize=7)

    def plot_id(self, ax, images, pores, variant=""):
        """Optional: the same plot over the whole ID -> <ID>_results/plots/<name>.png (complete runs only).
        images: DataFrame = images.json (one row per image), pores: all pores.json + column "image"
        (delete the method if not needed)."""
        if self.column not in pores.columns:
            raise MissingData(f"column {self.column} missing")
        for i, name in enumerate(images["image"]):
            values = pores.loc[pores["image"] == name, self.column].dropna().to_numpy(dtype=float)
            ax.hist(values, bins=20, histtype="step", color=COLORS[i % len(COLORS)], label=name)
        ax.set_xlabel(self.column)
        ax.set_title(f"{len(images)} images")
        ax.legend(fontsize=7.5)


def _image_values(image: dict) -> dict:
    """The core columns of images.json (docs/output_format.md 4) – the per_image values of the
    evaluation modules are in `image` too, under their names (e.g. image["phi_q10"])."""
    return {
        "image": image["image"],                            # <ID>_<image>
        "path": image.get("path"),                          # source file
        "pixel_size_um": image.get("pixel_size_um"),        # µm per px, None = no scale
        "scale_label": image.get("scale_label"),            # text read from the scale label
        "frame_origin_px": image.get("frame_origin_px"),    # sample frame, None without frame
        "frame_angle_deg": image.get("frame_angle_deg"),
        "frame_thickness_px": image.get("frame_thickness_px"),
        "frame_thickness_um": image.get("frame_thickness_um"),
        "frame_length_px": image.get("frame_length_px"),
        "frame_wedge_deg": image.get("frame_wedge_deg"),
        "manual_corrections": image.get("manual_corrections"),
        "porosity_percent": image.get("porosity_percent"),  # example of a module value (porosity)
    }


def _picture_values(picture) -> dict:
    """core.plotting.SamplePicture: the sample straightened along its frame (outer surface on top)."""
    return {
        "rgb": picture.rgb,         # (H, W, 3) uint8 – ax.imshow(picture.rgb)
        "pores": picture.pores,     # DataFrame pore_id, x, y: centroid of every counted pore in picture px
        "scale": picture.scale,     # picture px per image px
    }
