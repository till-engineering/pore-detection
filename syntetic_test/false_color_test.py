"""Synthetic test of the false color height chain (configs/relief/config_false_color_height.yaml).

Builds height images as a Keyence VHX shows its depth composition, with known pores, runs the
pipeline on them with that config and compares the result with the ground truth.

Height per image, in units of the noise sigma (what the chain measures in):
    surface   tilt + slow waves of random strength
    noise     gaussian, slightly correlated (sigma 1)
    pores     ellipses, steep walls and a flat bottom, depth 4..40 sigma, a raised rim ("red rim"
              of the focus variation at steep edges) at some of them
    scratches long, shallow lines (1.5 sigma) - must not become pores
    gap       on some images a very deep line along one border (gap to the resin): squeezes the
              color scale, the matrix then gets only few colors. It is deep, but no pore: it lies
              outside the sample, which the ROI removes later - here it does not count (don't care)
False colors: height stretched from its own minimum to maximum (as the microscope does) onto
hue 240 (blue, deep) .. 0 (red, high), random brightness, saved as TIFF (lossless, as the VHX
exports it) or JPEG (--jpeg: its chroma compression removes fine color noise, unevenly along the
scale - the noise measure and with it all depths in sigma then go wrong).

Printed per image: pixel precision / recall / IoU, pores found per depth class, false objects
(both without the gap).

Usage (from the project folder):
    .venv\\Scripts\\python syntetic_test\\false_color_test.py [--count 8] [--seed 1] [--no-run]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from pore_detection import Config, run  # noqa: E402
from pore_detection.core import writer  # noqa: E402

CONFIG = PROJECT / "configs" / "relief" / "config_false_color_height.yaml"
OUT = PROJECT / "syntetic_test" / "output" / "false_color"
ID = "vhx_synth"
WIDTH, HEIGHT = 1200, 800
DEPTH_CLASSES = [(4, 6), (6, 10), (10, 20), (20, 41)]      # sigma


def make_image(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float]]]:
    """(RGB uint8 false color image, pore id image, gap mask, [(pore id, depth in sigma)])."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    tilt = rng.uniform(0, 60)
    angle = rng.uniform(0, 2 * np.pi)
    height = tilt * (np.cos(angle) * x / WIDTH + np.sin(angle) * y / HEIGHT)
    for _ in range(3):                                  # 1..4 waves across the image width
        k = rng.uniform(1, 4) * 2 * np.pi / WIDTH
        direction = rng.uniform(0, 2 * np.pi)
        height += rng.uniform(0, 8) * np.sin(k * (x * np.cos(direction) + y * np.sin(direction))
                                             + rng.uniform(0, 6))
    noise = cv2.GaussianBlur(rng.normal(size=(HEIGHT, WIDTH)).astype(np.float32), (0, 0), 0.7)
    height += noise / noise.std()

    for _ in range(rng.integers(3, 8)):                 # shallow scratches
        p0 = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
        a = rng.uniform(0, np.pi)
        length = rng.uniform(150, 600)
        line = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.line(line, (int(p0[0]), int(p0[1])),
                 (int(p0[0] + length * np.cos(a)), int(p0[1] + length * np.sin(a))), 1, 2)
        height -= 1.5 * cv2.GaussianBlur(line.astype(np.float32), (0, 0), 1.0)

    ids = np.zeros((HEIGHT, WIDTH), np.int32)
    pores = []
    for pore_id in range(1, rng.integers(25, 45)):
        cx, cy = rng.uniform(40, WIDTH - 40), rng.uniform(40, HEIGHT - 40)
        ra = rng.uniform(2.5, 22)
        rb = ra * rng.uniform(0.5, 1)
        a = rng.uniform(0, np.pi)
        u = ((x - cx) * np.cos(a) + (y - cy) * np.sin(a)) / ra
        v = (-(x - cx) * np.sin(a) + (y - cy) * np.cos(a)) / rb
        r = np.sqrt(u * u + v * v)
        inside = (r < 1) & (ids == 0)
        if inside.sum() < 12:
            continue
        depth = float(np.exp(rng.uniform(np.log(4), np.log(40))))
        height -= depth * np.clip(3 * (1 - r), 0, 1)                   # steep walls, flat bottom
        if rng.random() < 0.5:                                         # raised rim
            height += 0.3 * depth * np.exp(-((r - 1.15) / 0.12) ** 2)
        ids[inside] = pore_id
        pores.append((pore_id, depth))

    gap = np.zeros((HEIGHT, WIDTH), bool)
    if rng.random() < 0.5:                              # deep gap along one border
        row = int(rng.uniform(5, 30))
        height[row:row + 4] -= rng.uniform(80, 200)
        gap[row:row + 4] = True

    position = (height.max() - height) / (height.max() - height.min())   # 0 = highest .. 1 = deepest
    hsv = np.stack([240 * position, np.ones_like(position),
                    rng.uniform(0.8, 1.0) * np.ones_like(position)], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    return rgb, ids, gap, pores


def generate(count: int, seed: int, jpeg: bool) -> None:
    rng = np.random.default_rng(seed)
    images, truth = OUT / ID, OUT / f"{ID}_gt"
    for old in images.glob("*") if images.exists() else []:
        old.unlink()
    images.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    meta = {}
    for index in range(count):
        rgb, ids, gap, pores = make_image(rng)
        name = f"img_{index:02d}"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if jpeg:
            cv2.imwrite(str(images / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            cv2.imwrite(str(images / f"{name}.tif"), bgr)
        cv2.imwrite(str(truth / f"{name}_ids.png"), ids.astype(np.uint16))
        cv2.imwrite(str(truth / f"{name}_gap.png"), gap.astype(np.uint8) * 255)
        meta[name] = {str(pore_id): depth for pore_id, depth in pores}
    (truth / "depths.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")


def run_pipeline() -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["logging"]["level"] = "WARNING"
    result = run(Config(data), input_path=OUT / ID, output_dir=OUT / "results")
    for error in result.errors:
        print(f"  FAILED {error.image}: {error.stage} / {error.method}: {error.message}")
    return result.results_dir


def evaluate(results_dir: Path) -> None:
    truth = OUT / f"{ID}_gt"
    depths = yaml.safe_load((truth / "depths.yaml").read_text(encoding="utf-8"))
    found_per_class = np.zeros(len(DEPTH_CLASSES), int)
    total_per_class = np.zeros(len(DEPTH_CLASSES), int)
    print(f"{'image':8} {'precision':>9} {'recall':>7} {'IoU':>6} {'pores':>9} {'false obj':>9}")
    for name, pore_depths in sorted(depths.items()):
        ids = cv2.imread(str(truth / f"{name}_ids.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        labels = cv2.imread(str(results_dir / f"{ID}_{name}" / writer.LABELS_TIF), cv2.IMREAD_UNCHANGED)
        # don't care: the gap and a margin of 15 px (smoothing, rim of the detection)
        gap = cv2.dilate(cv2.imread(str(truth / f"{name}_gap.png"), cv2.IMREAD_GRAYSCALE), np.ones((31, 31), np.uint8)) > 0
        labels = np.where(gap, 0, labels)
        detected = labels > 0
        gt = ids > 0
        tp = (detected & gt).sum()
        precision = tp / max(detected.sum(), 1)
        recall = tp / max(gt.sum(), 1)
        iou = tp / max((detected | gt).sum(), 1)
        hit = set(np.unique(ids[detected])) - {0}
        for pore_id, depth in pore_depths.items():
            c = next(i for i, (low, high) in enumerate(DEPTH_CLASSES) if low <= depth < high)
            total_per_class[c] += 1
            found_per_class[c] += int(pore_id) in hit
        false_objects = len(set(np.unique(labels[detected & ~gt])) - set(np.unique(labels[detected & gt])))
        print(f"{name:8} {precision:9.3f} {recall:7.3f} {iou:6.3f} {len(hit):4d}/{len(pore_depths):<4d} "
              f"{false_objects:9d}")
    print("\npores found per depth class (sigma):")
    for (low, high), found, total in zip(DEPTH_CLASSES, found_per_class, total_per_class):
        print(f"  {low:>2}-{high:<2}: {found:3d} / {total:3d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jpeg", action="store_true", help="save the images as JPEG instead of TIFF")
    parser.add_argument("--no-run", action="store_true", help="evaluate the last results again")
    args = parser.parse_args()
    if not args.no_run:
        generate(args.count, args.seed, args.jpeg)
        results_dir = run_pipeline()
    else:
        results_dir = OUT / "results" / f"{ID}_results"
    evaluate(results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
