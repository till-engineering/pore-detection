"""Base classes of all pipeline stages. Binding specification: docs/interfaces.md, section 3.

A method subclasses one base class, implements its CPU path and is registered by name:

    @register("gauss")
    class GaussFilter(Filter):
        params = {"sigma_px": "standard deviation of the gaussian kernel"}
        tileable = True
        tile_overlap_px = 8

        def process(self, ctx, image):
            ...  # self.sigma_px comes from config.yaml

Optional, checked by core with hasattr():
    process_gpu(ctxs, items) / detect_gpu(ctxs, items)  GPU path of tileable steps (batch of tiles)
    Evaluator.finalize(per_image, per_pore)              values over the whole ID
"""

import numpy as np
import pandas as pd

from .types import EvaluationResult, ImageContext, ImageInfo, LabeledPores, SampleFrame, ScaleResult


class Step:
    name: str = ""                  # registered name, set by @register
    params: dict[str, str] = {}     # parameter name -> description (for list_methods)
    choices: dict[str, list[str]] = {}  # parameter name -> allowed values (launcher shows a selection)
    tileable: bool = False          # local operation, may run on tiles
    tile_overlap_px: int = 0        # required border around each tile
    image_params: set[str] = set()  # parameters that may differ per image (entered per image in the launcher)

    def __init__(self, **params):
        """Parameters exactly as in config.yaml; each one becomes an attribute (self.sigma_px).

        Optional key per_image: {<image key>: {parameter: value}} – values of image_params
        for single images (passed by the launcher), see for_image()."""
        per_image = params.pop("per_image", None) or {}
        for image, values in per_image.items():
            unknown = set(values) - self.image_params
            if unknown:
                raise ValueError(f"{self.name}: {', '.join(sorted(unknown))} cannot be set per image "
                                 f"({image}); per image: {', '.join(sorted(self.image_params)) or 'nothing'}")
        self._params = params
        self._per_image = per_image
        for key, value in params.items():
            setattr(self, key, value)

    def for_image(self, image: str) -> "Step":
        """This step with the per-image values of one image (key: file name, e.g. "img_01.tif"; with
        input.views the recording, e.g. "img_01"); itself if there are none. A new instance, so values derived in __init__ follow too."""
        values = self._per_image.get(image)
        if not values:
            return self
        return type(self)(**{**self._params, **values})

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


# --- sample: always on the original image, whole image, not tileable ---

class RoiDetector(Step):
    def detect_roi(self, info: ImageInfo, original: np.ndarray) -> np.ndarray:
        """(H, W) bool, True = sample."""
        raise NotImplementedError


class ScaleDetector(Step):
    def detect_scale(self, info: ImageInfo, original: np.ndarray) -> ScaleResult:
        """Micrometres per pixel (None = unknown) and optionally the mask of the scale overlay;
        core removes that mask from the roi_mask."""
        raise NotImplementedError


class FrameDetector(Step):
    def detect_frame(self, info: ImageInfo, roi_mask: np.ndarray) -> SampleFrame | None:
        """Coordinate system of the sample from its mask (full image, without the scale overlay);
        None = no frame (method "none")."""
        raise NotImplementedError


# --- preprocessing: optional filters on the gray working image ---

class Filter(Step):
    def process(self, ctx: ImageContext, image: np.ndarray) -> np.ndarray:
        """(H, W) float32 0..1 in and out."""
        raise NotImplementedError


# --- detection: exactly one, delivers everything that may be a pore ---

class Detector(Step):
    def detect(self, ctx: ImageContext, image: np.ndarray) -> np.ndarray:
        """Gray (H, W) float32 in, (H, W) bool out."""
        raise NotImplementedError


# --- postprocessing ---

class MaskStep(Step):
    def process(self, ctx: ImageContext, mask: np.ndarray) -> np.ndarray:
        """(H, W) bool in and out."""
        raise NotImplementedError


class Labeler(Step):
    def label(self, ctx: ImageContext, mask: np.ndarray) -> LabeledPores:
        """properties of the result contain only pore_id (1 … count)."""
        raise NotImplementedError


class LabelStep(Step):
    # True: changes which pores exist (merge, split, filters …). False: only adds values per pore;
    # such steps are run again after a manual correction (see core/corrections.py)
    changes_pores: bool = True

    def process(self, ctx: ImageContext, pores: LabeledPores) -> LabeledPores:
        raise NotImplementedError


# --- evaluation: whole image, modules of one image run in parallel ---

class Evaluator(Step):
    excel_sheet: str | None = None  # sheet of the results Excel for its per-image values (None = module name)
    excel_hidden: set[str] = set()  # per-image values written to JSON only, not to the Excel

    def evaluate(self, ctx: ImageContext, pores: LabeledPores) -> EvaluationResult:
        raise NotImplementedError

    # optional:
    # def finalize(self, per_image: pd.DataFrame, per_pore: Iterator[pd.DataFrame]) -> dict[str, float | int | str]


# --- plots: pictures of the finished values of one image (PNG in the results, shown by the viewer) ---

class Plot(Step):
    columns: list[str] = []         # per-image values the plot needs (missing -> plot skipped, logged)
    pore_columns: list[str] = []    # per-pore values the plot needs (missing -> plot skipped, logged)
    variant: str = ""               # optional second view (e.g. "normalized"): drawn too, switch in the viewer
    needs_picture: bool = False     # True: plot() also gets picture= (core.plotting.SamplePicture or None)
    size_in: tuple[float, float] | None = None  # figure size in inch, None = core.plotting.SIZE_IN

    def plot(self, ax, image: dict, pores: pd.DataFrame) -> None:
        """Draw into the matplotlib Axes `ax`. image: the row of images.json, pores: pores.json.
        A plot with a `variant` takes a keyword argument variant ("" = normal view, else the variant).
        Raise core.plotting.MissingData if something else needed is missing. Size, look and
        saving are done by core (core/plotting.py); import matplotlib only inside this method."""
        raise NotImplementedError

    # optional: the same plot over the whole ID (all images), saved in <ID>_results/plots/
    # def plot_id(self, ax, images: pd.DataFrame, pores: pd.DataFrame, variant: str = "") -> None
    #     images: images.json (one row per image), pores: pores.json of all images + column "image"

    # optional: further copies of this plot that are only saved, not shown in the viewer
    # def extra_plots(self, pores: pd.DataFrame) -> dict[str, "Plot"]
    #     key = file stem below plots/ with a subfolder (e.g. "all_q/critical_pores_q5"), value = plot
    #     instance with other parameters; drawn for the image and (with plot_id) over the ID
