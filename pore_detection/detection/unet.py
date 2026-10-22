"""U-Net: a trained neural network decides per pixel "pore or not" (dataset_eval/train_unet.py trains it).

Input is the colour original (ctx.original), unchanged: the gray working image and with it every
preprocessing filter of the config are NOT used – the network gets exactly what it got in training
(see core/unet_model.py: RGB -> 0..1 -> mean / std of the encoder, original resolution, no scaling).
Use it with preprocessing: [] (configs/config_unet.yaml), the filters would only cost time.

model_dir: folder of a trained model (weights.pt + model.yaml with encoder, normalization, patch size,
overlap and the threshold found on the validation images); relative paths start at the project folder.
threshold: auto = the one of model.yaml, or a number 0..1 (pore probability at or above it = pore).

Tileable: core cuts the image into tiles with the model's overlap_px; inside a tile the network runs
on windows of the training patch size (patch_px) and keeps only their centres. GPU (detect_gpu) in
float16, the window batch is halved when the GPU memory runs out. Pores only inside the sample.
The model is valid for the magnification of its training images (no scaling).
"""

import threading
from pathlib import Path

from ..core import unet_model
from ..core.config import CONFIG_PATH
from ..core.registry import register
from ..core.step import Detector

CPU_BATCH = 4
GPU_BATCH = 16      # starting value, halved on out-of-memory

_cpu_lock = threading.Lock()    # torch uses all cores per call; parallel tiles would only compete


@register("unet")
class Unet(Detector):
    """Trained U-Net on the colour original: pore probability per pixel above the threshold = pore."""

    params = {
        "model_dir": "folder of the trained model (weights.pt + model.yaml); relative to the project folder",
        "threshold": "auto (threshold of model.yaml, best on the validation images) or pore probability 0..1",
    }
    tileable = True

    def __init__(self, **params):
        super().__init__(**params)
        folder = Path(self.model_dir)
        self.folder = folder if folder.is_absolute() else CONFIG_PATH.parent / folder
        recipe = unet_model.read_recipe(self.folder)
        unet_model._libraries()                     # missing torch / smp: clear message at the start
        self.tile_overlap_px = recipe["overlap_px"]
        self.cut = recipe["threshold"] if self.threshold == "auto" else float(self.threshold)

    def detect(self, ctx, image):
        recipe, model = unet_model.load(self.folder, "cpu")
        with _cpu_lock:
            prob = unet_model.predict(model, recipe, ctx.original, "cpu", CPU_BATCH)
        return (prob >= self.cut) & ctx.roi_mask

    def detect_gpu(self, ctxs, items):
        recipe, model = unet_model.load(self.folder, "cuda")
        return [(unet_model.predict(model, recipe, ctx.original, "cuda", GPU_BATCH) >= self.cut) & ctx.roi_mask
                for ctx in ctxs]
