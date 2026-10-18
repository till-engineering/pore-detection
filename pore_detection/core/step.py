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

from .types import EvaluationResult, ImageContext, ImageInfo, LabeledPores, SampleFrame, ScaleResult


class Step:
    name: str = ""                  # registered name, set by @register
    params: dict[str, str] = {}     # parameter name -> description (for list_methods)
    tileable: bool = False          # local operation, may run on tiles
    tile_overlap_px: int = 0        # required border around each tile

    def __init__(self, **params):
        """Parameters exactly as in config.yaml; each one becomes an attribute (self.sigma_px)."""
        for key, value in params.items():
            setattr(self, key, value)

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
    def evaluate(self, ctx: ImageContext, pores: LabeledPores) -> EvaluationResult:
        raise NotImplementedError

    # optional:
    # def finalize(self, per_image: pd.DataFrame, per_pore: Iterator[pd.DataFrame]) -> dict[str, float | int | str]
