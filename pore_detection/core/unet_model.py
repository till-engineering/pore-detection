"""U-Net model: the ONE place that turns an image into the network input and a prediction into pores.

Used by the detector detection/unet.py (pipeline run) AND by dataset_eval/train_unet.py (training,
validation), so both always see exactly the same input:

    original (H, W, 3) RGB, uint8 / uint16, unchanged (no filter, no scaling)
      -> to_unit():   0..1 (8 bit / 255, 16 bit / 65535)
      -> normalize(): (x - mean) / std per channel (statistics of the encoder pre-training)
      -> windows of patch_px x patch_px (the training patch size), overlapping by overlap_px;
         only the centre of each window is kept, the image border is mirrored (BORDER_REFLECT_101)
      -> network -> sigmoid -> pore probability 0..1 per pixel

A model folder (models/<name>/) holds weights.pt (state dict) and model.yaml (the recipe: encoder,
mean, std, patch_px, overlap_px, threshold, ...). The recipe is read without torch.

torch and segmentation_models_pytorch are imported lazily (optional dependencies).
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import yaml

MODEL_FILE = "model.yaml"
WEIGHTS_FILE = "weights.pt"
MULTIPLE_PX = 32            # 5 halvings in the encoder: network input sizes are multiples of 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# MicroNet weights of NASA (Stuckner et al. 2022): ResNet50 encoder pre-trained on ~110 000 micrographs,
# input RGB 0..1 normalized with the ImageNet statistics
MICRONET_URLS = {
    "resnet50": "https://huggingface.co/jstuckner/microscopy-resnet50-micronet/resolve/main/"
                "resnet50_micronet_weights.pth",
}
INSTALL_HINT = (".venv\\Scripts\\python -m pip install segmentation-models-pytorch "
                "(PyTorch: see requirements.txt)")

_cache: dict = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------- input

def to_unit(original: np.ndarray) -> np.ndarray:
    """RGB uint8 / uint16 -> float32 0..1 (by the full range of the dtype, not by the image content)."""
    scale = 255.0 if original.dtype == np.uint8 else 65535.0
    return original.astype(np.float32) * np.float32(1.0 / scale)


def normalize(unit: np.ndarray, recipe: dict) -> np.ndarray:
    """(H, W, 3) float32 0..1 -> (3, H, W) float32, normalized with the recipe's mean / std."""
    mean = np.asarray(recipe["mean"], dtype=np.float32)
    std = np.asarray(recipe["std"], dtype=np.float32)
    return ((unit - mean) / std).transpose(2, 0, 1).copy()


def pad(array: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    """Mirror the border (BORDER_REFLECT_101, as core.tiling does), also beyond the image size."""
    while top or bottom or left or right:
        height, width = array.shape[:2]
        step = [min(v, n - 1) for v, n in ((top, height), (bottom, height), (left, width), (right, width))]
        if array.dtype == bool:
            array = cv2.copyMakeBorder(array.view(np.uint8), *step, cv2.BORDER_REFLECT_101).view(bool)
        else:
            array = cv2.copyMakeBorder(array, *step, cv2.BORDER_REFLECT_101)
        top, bottom, left, right = (v - s for v, s in zip((top, bottom, left, right), step))
    return array


# ---------------------------------------------------------------- model

def _libraries():
    try:
        import segmentation_models_pytorch as smp
        import torch
    except ImportError as error:
        raise ImportError(f"the U-Net needs PyTorch and segmentation_models_pytorch ({error}); "
                          f"install: {INSTALL_HINT}") from None
    return torch, smp


def build(recipe: dict, pretrained: str | None = None):
    """The network of a recipe. pretrained: None (weights follow from weights.pt), imagenet or micronet."""
    torch, smp = _libraries()
    if recipe["architecture"] != "unet":
        raise ValueError(f"unknown architecture {recipe['architecture']!r} (only unet)")
    encoder = recipe["encoder"]
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained == "imagenet" else None,
                     in_channels=3, classes=1, activation=None)
    if pretrained == "micronet":
        if encoder not in MICRONET_URLS:
            raise ValueError(f"MicroNet weights only for {', '.join(MICRONET_URLS)}, not {encoder}")
        state = torch.hub.load_state_dict_from_url(MICRONET_URLS[encoder], map_location="cpu", progress=True)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}  # classification head
        model.encoder.load_state_dict(state)
    elif pretrained not in (None, "imagenet"):
        raise ValueError(f"unknown pretrained weights {pretrained!r} (imagenet | micronet)")
    return model


def read_recipe(model_dir: Path) -> dict:
    path = Path(model_dir) / MODEL_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no U-Net model in {model_dir} ({MODEL_FILE} missing)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def save(model_dir: Path, model, recipe: dict) -> None:
    torch, _ = _libraries()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / WEIGHTS_FILE)
    write_recipe(model_dir, recipe)


def write_recipe(model_dir: Path, recipe: dict) -> None:
    text = yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True)
    (Path(model_dir) / MODEL_FILE).write_text(text, encoding="utf-8")


def load(model_dir: Path, device: str):
    """(recipe, network in eval mode on `device`), loaded once per folder and device."""
    key = (str(Path(model_dir).resolve()), device)
    with _cache_lock:
        if key not in _cache:
            torch, _ = _libraries()
            recipe = read_recipe(model_dir)
            model = build(recipe)
            state = torch.load(Path(model_dir) / WEIGHTS_FILE, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            _cache[key] = (recipe, model.to(device).eval())
        return _cache[key]


# ---------------------------------------------------------------- prediction

def windows(height: int, width: int, recipe: dict) -> tuple[list[tuple[int, int]], int, int]:
    """Window starts in the padded image and the padding (bottom, right) beyond overlap_px.

    The image is padded by overlap_px on every side (+ bottom / right up to the window grid);
    window (y, x) covers padded[y:y + patch, x:x + patch] and gives the result for
    image[y:y + step, x:x + step] with step = patch - 2 x overlap.
    """
    patch, overlap = recipe["patch_px"], recipe["overlap_px"]
    step = patch - 2 * overlap
    rows, cols = -(-height // step), -(-width // step)
    starts = [(y * step, x * step) for y in range(rows) for x in range(cols)]
    return starts, rows * step - height, cols * step - width


def predict(model, recipe: dict, original: np.ndarray, device: str, batch_size: int = 8) -> np.ndarray:
    """Pore probability (H, W) float32 0..1 of an RGB original. On the GPU: float16 (autocast);
    out of GPU memory -> the batch is halved."""
    torch, _ = _libraries()
    height, width = original.shape[:2]
    patch, overlap = recipe["patch_px"], recipe["overlap_px"]
    step = patch - 2 * overlap
    starts, extra_y, extra_x = windows(height, width, recipe)
    data = normalize(to_unit(pad(original, overlap, overlap + extra_y, overlap, overlap + extra_x)), recipe)
    result = np.empty((height + extra_y, width + extra_x), dtype=np.float32)
    cuda = device == "cuda"
    done = 0
    with torch.inference_mode():
        while done < len(starts):
            batch = starts[done:done + batch_size]
            stack = np.stack([data[:, y:y + patch, x:x + patch] for y, x in batch])
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    logits = model(torch.from_numpy(stack).to(device))
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                batch_size //= 2
                continue
            prob = torch.sigmoid(logits.float())[:, 0, overlap:overlap + step, overlap:overlap + step]
            for (y, x), tile in zip(batch, prob.cpu().numpy()):
                result[y:y + step, x:x + step] = tile
            done += len(batch)
    return result[:height, :width]
