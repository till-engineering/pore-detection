"""Synthetic test of the false color height chain (configs/relief/config_false_color_height.yaml).

Builds height images as a Keyence VHX shows its depth composition, with known pores and a known
sample outline, runs the pipeline on them with that config and compares the result with the ground
truth.

Height per image, in units of the noise sigma (what the chain measures in):
    surface   tilt + slow waves of random strength
    noise     gaussian, slightly correlated (sigma 1)
    pores     ellipses, steep walls and a flat bottom, depth 4..40 sigma, a raised rim ("red rim"
              of the focus variation at steep edges) at some of them
    scratches long, shallow lines (1.5 sigma) - must not become pores
    sample    a slightly rotated rectangle with a rounded edge (drops a few sigma over ~8 px); on
              some images it fills the whole image (cut-out from inside: no resin)
    resin     around the sample: lower, 2-4x the noise and blobs (rough)
    gap       on some images a very deep line along parts of the sample edge (gap to the resin):
              squeezes the color scale, the matrix then gets only few colors. It is deep, but no
              pore: the ROI has to keep it out - pores near it do not count (don't care)
False colors: height stretched from its own minimum to maximum (as the microscope does) onto
hue 240 (blue, deep) .. 0 (red, high), random brightness, saved as TIFF (lossless, as the VHX
exports it) or JPEG (--jpeg: its chroma compression removes fine color noise, unevenly along the
scale - the noise measure and with it all depths in sigma then go wrong).

Printed per image: IoU of the detected sample (ROI) with the true one, pixel precision / recall /
IoU of the pores, pores found per depth class, false objects (pore values without the gap).

Usage (from the project folder):
    .venv\\Scripts\\python syntetic_test\\false_color_test.py [--count 8] [--seed 1] [--jpeg] [--no-run]
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


def make_image(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                  list[tuple[int, float]]]:
    """(RGB uint8 false color image, pore id image, sample mask, gap mask, [(pore id, depth in sigma)])."""
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
    noise /= noise.std()

    # sample: rotated rectangle, or the whole image (cut-out)
    if rng.random() < 1 / 6:
        sample = np.ones((HEIGHT, WIDTH), bool)
    else:
        top, bottom = rng.uniform(40, 140), rng.uniform(40, 140)
        left, right = rng.uniform(30, 120), rng.uniform(30, 120)
        box = (((left + WIDTH - right) / 2, (top + HEIGHT - bottom) / 2),
               (WIDTH - left - right, HEIGHT - top - bottom), rng.uniform(-3, 3))
        outline = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(outline, [cv2.boxPoints(box).astype(np.int32)], 1)
        sample = outline > 0
    inside = cv2.distanceTransform(sample.astype(np.uint8), cv2.DIST_L2, 5)
    if not sample.all():
        height -= rng.uniform(2, 6) * np.exp(-inside / 8) * sample            # rounded edge
        blobs = cv2.GaussianBlur(rng.normal(size=(HEIGHT, WIDTH)).astype(np.float32), (0, 0), 3)
        blobs /= blobs.std()
        resin_height = -rng.uniform(2, 10) + rng.uniform(2, 4) * noise + rng.uniform(2, 5) * blobs
        height += np.where(sample, noise, resin_height)
    else:
        height += noise

    for _ in range(rng.integers(3, 8)):                 # shallow scratches (sample and resin)
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
        if inside[int(cy), int(cx)] < 30:               # pores only well inside the sample
            continue
        ra = rng.uniform(2.5, 22)
        rb = ra * rng.uniform(0.5, 1)
        a = rng.uniform(0, np.pi)
        u = ((x - cx) * np.cos(a) + (y - cy) * np.sin(a)) / ra
        v = (-(x - cx) * np.sin(a) + (y - cy) * np.cos(a)) / rb
        r = np.sqrt(u * u + v * v)
        pore = (r < 1) & (ids == 0)
        if pore.sum() < 12:
            continue
        depth = float(np.exp(rng.uniform(np.log(4), np.log(40))))
        height -= depth * np.clip(3 * (1 - r), 0, 1)                   # steep walls, flat bottom
        if rng.random() < 0.5:                                         # raised rim
            height += 0.3 * depth * np.exp(-((r - 1.15) / 0.12) ** 2)
        ids[pore] = pore_id
        pores.append((pore_id, depth))

    gap = np.zeros((HEIGHT, WIDTH), bool)
    if not sample.all() and rng.random() < 0.5:         # deep gap along parts of the sample edge
        edge = (inside > 0) & (inside <= 3)
        segments = cv2.resize((rng.random((4, 6)) < 0.7).astype(np.uint8), (WIDTH, HEIGHT),
                              interpolation=cv2.INTER_NEAREST) > 0
        gap = edge & segments
        height[gap] -= rng.uniform(80, 200)

    position = (height.max() - height) / (height.max() - height.min())   # 0 = highest .. 1 = deepest
    hsv = np.stack([240 * position, np.ones_like(position),
                    rng.uniform(0.8, 1.0) * np.ones_like(position)], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    return rgb, ids, sample, gap, pores


def generate(count: int, seed: int, jpeg: bool) -> None:
    rng = np.random.default_rng(seed)
    images, truth = OUT / ID, OUT / f"{ID}_gt"
    for old in images.glob("*") if images.exists() else []:
        old.unlink()
    images.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    meta = {}
    for index in range(count):
        rgb, ids, sample, gap, pores = make_image(rng)
        name = f"img_{index:02d}"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if jpeg:
            cv2.imwrite(str(images / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            cv2.imwrite(str(images / f"{name}.tif"), bgr)
        cv2.imwrite(str(truth / f"{name}_ids.png"), ids.astype(np.uint16))
        cv2.imwrite(str(truth / f"{name}_gap.png"), gap.astype(np.uint8) * 255)
        cv2.imwrite(str(truth / f"{name}_sample.png"), sample.astype(np.uint8) * 255)
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
    print(f"{'image':8} {'ROI IoU':>7} {'precision':>9} {'recall':>7} {'IoU':>6} {'pores':>9} {'false obj':>9}")
    for name, pore_depths in sorted(depths.items()):
        ids = cv2.imread(str(truth / f"{name}_ids.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        folder = results_dir / f"{ID}_{name}"
        if not (folder / writer.LABELS_TIF).exists():
            print(f"{name:8} failed")
            continue
        labels = cv2.imread(str(folder / writer.LABELS_TIF), cv2.IMREAD_UNCHANGED)
        roi = cv2.imread(str(folder / writer.ROI_MASK_TIF), cv2.IMREAD_GRAYSCALE) > 0
        sample = cv2.imread(str(truth / f"{name}_sample.png"), cv2.IMREAD_GRAYSCALE) > 0
        roi_iou = (roi & sample).sum() / max((roi | sample).sum(), 1)
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
        print(f"{name:8} {roi_iou:7.3f} {precision:9.3f} {recall:7.3f} {iou:6.3f} {len(hit):4d}/{len(pore_depths):<4d} "
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
