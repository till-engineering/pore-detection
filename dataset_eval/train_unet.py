"""Train the U-Net pore detector (detection method "unet") on micrographs + ground-truth masks.

    python dataset_eval/train_unet.py                     settings: dataset_eval/train_config.yaml
    python dataset_eval/train_unet.py --config other.yaml

Images and masks are paired by file name as in evaluate.py. The images are split into training and
validation by sample (split.group_pattern) or by image. Training sees random patches (patch_px) of
the unchanged colour originals, augmented (augment.py); validation runs the network on the whole
validation images with the SAME prediction code as the pipeline (core/unet_model.py).

The best epoch (validation Dice at probability 0.5) is saved at once, so Ctrl+C keeps it. At the
end the threshold with the best validation Dice is searched and written to model.yaml.

Written to <output.models>/<name>/:
    weights.pt, model.yaml      the model and its recipe (read by detection/unet.py)
    split.yaml                  which images trained / validated (leave them out of evaluate.py)
    history.xlsx, training_curves.png   loss and validation values per epoch
    train_config_used.yaml      copy of the settings
"""

import argparse
import math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pore_detection import load_config  # noqa: E402
from pore_detection.core import io, unet_model  # noqa: E402

import augment  # noqa: E402
import metrics  # noqa: E402
from evaluate import pair_files  # noqa: E402

DEFAULT_CONFIG = HERE / "train_config.yaml"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)     # searched for the final threshold


@dataclass
class Sample:
    name: str               # file stem
    group: str              # sample the image belongs to (split)
    original: np.ndarray    # (H, W, 3) RGB uint8 / uint16, as the pipeline loads it
    mask: np.ndarray        # (H, W) bool, True = pore
    pores: np.ndarray       # (n, 2) y, x of pore pixels (subsample) for patches on pores


# ---------------------------------------------------------------- data

def load_settings(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, key in (("data", "images"), ("data", "masks"), ("output", "models")):
        folder = Path(settings[section][key])
        settings[section][key] = (folder if folder.is_absolute() else path.parent / folder).resolve()
    return settings


def load_samples(settings: dict, rng: np.random.Generator) -> list[Sample]:
    data = settings["data"]
    pairs, skipped = pair_files(data["images"], data["masks"], load_config().input.extensions)
    for row in skipped.itertuples():
        print(f"  skipped {row.file}: {row.reason}")
    pattern = settings["split"]["group_pattern"]
    samples = []
    for pair in tqdm(pairs, desc="loading", unit="image"):
        _, original = io.load_image(pair.image, data["images"].name)
        mask = metrics.read_mask(pair.mask, data["mask_threshold"])
        if mask.shape != original.shape[:2]:
            print(f"  skipped {pair.image.name}: mask {mask.shape} != image {original.shape[:2]}")
            continue
        stem = pair.image.stem
        match = re.search(pattern, stem) if pattern else None
        if pattern and not match:
            print(f"  {stem}: group_pattern does not match, the image is its own sample")
        pores = np.argwhere(mask)
        if len(pores) > 100_000:
            pores = pores[rng.choice(len(pores), 100_000, replace=False)]
        samples.append(Sample(stem, match.group(0) if match else stem, original, mask, pores))
    return samples


def split(samples: list[Sample], val_share: float, rng: np.random.Generator) -> tuple[list[Sample], list[Sample]]:
    """Whole samples (groups) go to validation until it holds val_share of the images."""
    groups = sorted({s.group for s in samples})
    rng.shuffle(groups)
    target = max(1, round(val_share * len(samples)))
    val_groups, count = set(), 0
    for group in groups:
        if count >= target or len(val_groups) == len(groups) - 1:
            break
        val_groups.add(group)
        count += sum(s.group == group for s in samples)
    return [s for s in samples if s.group not in val_groups], [s for s in samples if s.group in val_groups]


def patch(sample: Sample, size: int, on_pore: bool, settings: dict, recipe: dict, seed: int):
    """One augmented, normalized training patch: (3, size, size) float32 and (size, size) float32."""
    rng = np.random.default_rng(seed)
    height, width = sample.mask.shape
    if on_pore and len(sample.pores):
        y, x = sample.pores[rng.integers(len(sample.pores))]
        y0 = int(np.clip(y - rng.integers(size), 0, max(height - size, 0)))
        x0 = int(np.clip(x - rng.integers(size), 0, max(width - size, 0)))
    else:
        y0, x0 = int(rng.integers(max(height - size, 0) + 1)), int(rng.integers(max(width - size, 0) + 1))
    image = sample.original[y0:y0 + size, x0:x0 + size]
    mask = sample.mask[y0:y0 + size, x0:x0 + size]
    if image.shape[:2] != (size, size):        # image smaller than a patch: mirrored as in the run
        extra_y, extra_x = size - image.shape[0], size - image.shape[1]
        image = unet_model.pad(image, 0, extra_y, 0, extra_x)
        mask = unet_model.pad(mask, 0, extra_y, 0, extra_x)
    image, mask = augment.apply(unet_model.to_unit(image), mask, settings, rng)
    return unet_model.normalize(image, recipe), mask.astype(np.float32)


# ---------------------------------------------------------------- scores

def counts(prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per threshold of THRESHOLDS: (tp, fp, fn) pixels."""
    bins = np.minimum((prob * 100).astype(np.int64), 99)
    pore = np.bincount(bins[mask], minlength=100)
    other = np.bincount(bins[~mask], minlength=100)
    above_pore = np.cumsum(pore[::-1])[::-1]      # pixels with probability >= bin
    above_other = np.cumsum(other[::-1])[::-1]
    index = np.round(THRESHOLDS * 100).astype(int)
    tp, fp = above_pore[index], above_other[index]
    return np.stack([tp, fp, mask.sum() - tp], axis=1)


def dice(c: np.ndarray) -> np.ndarray:
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def validate(model, recipe: dict, samples: list[Sample], device: str) -> np.ndarray:
    """(tp, fp, fn) per threshold, summed over the validation images (whole images, as in the run)."""
    model.eval()
    total = np.zeros((len(THRESHOLDS), 3), dtype=np.int64)
    for sample in samples:
        total += counts(unet_model.predict(model, recipe, sample.original, device), sample.mask)
    model.train()
    return total


# ---------------------------------------------------------------- output

def write_history(history: list[dict], folder: Path) -> None:
    table = pd.DataFrame(history)
    table.to_excel(folder / "history.xlsx", index=False, engine="xlsxwriter")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pore_detection.core.plotting import COLORS, DPI, STYLE

    with plt.rc_context(STYLE):
        fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(9, 3.6))
        ax_loss.plot(table["epoch"], table["train_loss"], color=COLORS[0], lw=2)
        ax_loss.set(xlabel="epoch", ylabel="training loss", title="Loss (0.7 BCE + 0.3 Dice)")
        ax_dice.plot(table["epoch"], table["val_dice"], color=COLORS[0], lw=2, label="Dice at 0.5")
        ax_dice.plot(table["epoch"], table["val_dice_best_threshold"], color=COLORS[1], lw=2,
                     label="Dice at best threshold")
        best = table.loc[table["val_dice"].idxmax()]
        ax_dice.scatter([best["epoch"]], [best["val_dice"]], s=40, color=COLORS[0], edgecolor="white", zorder=3)
        ax_dice.set(xlabel="epoch", ylabel="validation Dice", title=f"Validation (best epoch {int(best['epoch'])})")
        ax_dice.legend(loc="lower right")
        for ax in (ax_loss, ax_dice):
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(folder / "training_curves.png", dpi=DPI)
        plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="training settings")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = load_settings(config_path)
    for key in ("images", "masks"):
        if not settings["data"][key].is_dir():
            sys.exit(f"folder not found: {settings['data'][key]}")

    torch, _ = unet_model._libraries()
    seed = settings["split"]["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("no CUDA GPU found - training on the CPU is very slow")

    samples = load_samples(settings, rng)
    if len(samples) < 2:
        sys.exit("at least 2 image / mask pairs are needed")
    train, val = split(samples, settings["split"]["val_share"], rng)
    groups = len({s.group for s in samples})
    if not val:
        sys.exit(f"all images belong to one sample ({samples[0].group}) - check split.group_pattern")
    print(f"{len(samples)} images of {groups} samples -> {len(train)} training, {len(val)} validation images")

    name = settings["output"]["name"] or f"unet_{datetime.now():%Y-%m-%d_%H%M}"
    folder = settings["output"]["models"] / name
    if folder.exists():
        sys.exit(f"model folder exists already: {folder} (other output.name)")
    folder.mkdir(parents=True)
    shutil.copyfile(config_path, folder / "train_config_used.yaml")
    (folder / "split.yaml").write_text(yaml.safe_dump({
        "group_pattern": settings["split"]["group_pattern"],
        "train": sorted(s.name for s in train), "val": sorted(s.name for s in val),
        "train_samples": sorted({s.group for s in train}), "val_samples": sorted({s.group for s in val}),
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")

    m, t = settings["model"], settings["training"]
    if m["patch_px"] % unet_model.MULTIPLE_PX or not 0 <= 2 * m["overlap_px"] < m["patch_px"]:
        sys.exit("patch_px must be a multiple of 32 and larger than 2 x overlap_px")
    recipe = {
        "architecture": "unet",
        "encoder": m["encoder"],
        "pretrained": m["pretrained"],
        "input": "RGB original, unchanged (no filter, no scaling)",
        "value_range": "0..1 (8 bit / 255, 16 bit / 65535)",
        "mean": unet_model.IMAGENET_MEAN,       # micronet and imagenet weights both use these
        "std": unet_model.IMAGENET_STD,
        "patch_px": m["patch_px"],
        "overlap_px": m["overlap_px"],
        "threshold": 0.5,
        "trained": f"{datetime.now():%Y-%m-%d %H:%M}",
        "images": str(settings["data"]["images"]),
        "train_images": len(train),
        "val_images": len(val),
    }
    model = unet_model.build(recipe, m["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    bce = torch.nn.BCEWithLogitsLoss()

    def loss_of(logits, target):
        prob = torch.sigmoid(logits)
        overlap = (prob * target).sum()
        dice_loss = 1 - (2 * overlap + 1) / (prob.sum() + target.sum() + 1)
        return t["bce_weight"] * bce(logits, target) + t["dice_weight"] * dice_loss

    history, best_dice, best_epoch = [], -1.0, 0
    started = time.time()
    with ThreadPoolExecutor(t["workers"]) as pool:
        try:
            for epoch in range(1, t["epochs"] + 1):
                jobs = [(s, rng.random() < t["pore_share"], int(rng.integers(2 ** 63)))
                        for s in train for _ in range(t["patches_per_image"])]
                order = rng.permutation(len(jobs))
                steps = math.ceil(len(jobs) / t["batch_size"])
                losses = []
                bar = tqdm(range(steps), desc=f"epoch {epoch}", unit="batch", leave=False)
                for step in bar:
                    chosen = [jobs[i] for i in order[step * t["batch_size"]:(step + 1) * t["batch_size"]]]
                    made = list(pool.map(lambda j: patch(j[0], m["patch_px"], j[1], settings["augment"],
                                                         recipe, j[2]), chosen))
                    images = torch.from_numpy(np.stack([a for a, _ in made])).to(device)
                    masks = torch.from_numpy(np.stack([b for _, b in made]))[:, None].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                        logits = model(images)
                    loss = loss_of(logits.float(), masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(loss.item())
                    bar.set_postfix(loss=f"{np.mean(losses):.4f}")
                scheduler.step()

                c = validate(model, recipe, val, device)
                scores = dice(c)
                at_half = float(scores[np.searchsorted(THRESHOLDS, 0.5)])
                history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_dice": at_half,
                                "val_dice_best_threshold": float(scores.max()),
                                "best_threshold": float(THRESHOLDS[scores.argmax()]),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "minutes": (time.time() - started) / 60})
                improved = at_half > best_dice
                if improved:
                    best_dice, best_epoch = at_half, epoch
                    unet_model.save(folder, model, {**recipe, "best_epoch": epoch, "val_dice_at_0.5": round(at_half, 4)})
                tqdm.write(f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  val Dice {at_half:.4f}"
                           f"{'  * saved' if improved else ''}")
                write_history(history, folder)
                if epoch - best_epoch >= t["early_stopping"]:
                    print(f"no improvement for {t['early_stopping']} epochs - stopped")
                    break
        except KeyboardInterrupt:
            print("\nstopped by hand - the best epoch so far is kept")
    if not history:
        print("no epoch finished, no model written")
        return 1

    # threshold of the best epoch on the validation images
    state = torch.load(folder / unet_model.WEIGHTS_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state)
    c = validate(model, recipe, val, device)
    scores = dice(c)
    best = int(scores.argmax())
    tp, fp, fn = (int(v) for v in c[best])
    recipe.update({
        "threshold": float(THRESHOLDS[best]),
        "best_epoch": best_epoch,
        "val_dice_at_0.5": round(best_dice, 4),
        "val_dice": round(float(scores[best]), 4),
        "val_precision": round(tp / max(tp + fp, 1), 4),
        "val_recall": round(tp / max(tp + fn, 1), 4),
    })
    unet_model.write_recipe(folder, recipe)
    print(f"model: {folder}\n  best epoch {best_epoch}, threshold {recipe['threshold']}, "
          f"validation Dice {recipe['val_dice']} (precision {recipe['val_precision']}, "
          f"recall {recipe['val_recall']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
