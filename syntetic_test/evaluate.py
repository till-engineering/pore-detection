"""Run the pore detection pipeline on the synthetic images and compare it with the ground truth.

Uses the project config.yaml unchanged except for:
    input / output paths  -> synthetic image folder / output/pipeline
    sample.scale          -> fixed_scale with um_per_px of the synthetic images (no scale bar)
    logging.level         -> WARNING (keeps the console readable)
The pipeline is only called through its public API; nothing in pore_detection is changed.

Left out of the evaluation:
    scale bar     the box (+ margin) of the burnt-in scale bar, for GT and detection, as
                  the scale_bar method would mask it out (its values are random, so the
                  run uses fixed_scale and gets no overlay mask)
    invisible     GT pores below --min-visible-px (+ 1 px) are a don't-care zone for the
                  pixel and object scores; the porosity still counts them

Printed per image and over all images:
    ROI          detected sample vs. ground-truth sample
    porosity     ground truth vs. detected
    pixels       precision / recall / F1 / IoU of the pore mask
    errors       where false-positive pixels lie, why pore pixels are missed
    objects      matching of GT pores and detected pores (IoU), splits, merges,
                 recall per pore type (needs meta.json) and size class
    filters      per filter: real pores rejected (wrong) vs. false objects rejected (right)
    remaining    false objects that are still counted (what a new filter could remove)
    timing       per pipeline stage

Usage:
    python evaluate.py                  # run pipeline + evaluate
    python evaluate.py --no-run         # evaluate the last pipeline results again
    python evaluate.py --iou 0.3        # other IoU threshold for object matching
    python evaluate.py --min-visible-px 0   # score all GT pores, also invisible ones
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pore_detection import Config, RunAborted, load_config, run  # noqa: E402

RESIN, MATRIX, PORE = 0, 1, 2                       # classes of <name>_labels.png
SIZE_BINS_UM = [0, 5, 10, 20, 50, math.inf]         # equivalent diameter classes of GT pores
SEAM_HALF_WIDTH_PX = 2                              # stitching seam band: +- this many px
SCALE_BAR_PAD_PX = 4                                # margin around the scale bar box (as exclusion_pad_px)
MIN_VISIBLE_PX = 3                                  # default: smaller GT pores are invisible (don't care)
FP_CATEGORIES = ["resin", "pore_rim", "seam", "sample_edge", "dark_phase", "bright_matrix"]
FN_CATEGORIES = ["outside_roi", "pore_rim", "interior"]


# ---------------------------------------------------------------- reading

def read_image(path: Path) -> np.ndarray:
    """cv2.imread for non-ASCII paths, unchanged dtype."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def read_table(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def load_synth_settings(gt_dir: Path) -> tuple[dict, dict | None]:
    """Generator config (from meta.json if present, else synth_config.yaml) and meta.json."""
    meta_path = gt_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta["config"], meta
    print(f"note: {meta_path} missing (generator run not finished?) – using synth_config.yaml, "
          "no pore types")
    return yaml.safe_load((HERE / "synth_config.yaml").read_text(encoding="utf-8")), None


# ---------------------------------------------------------------- pipeline

def pipeline_config(um_per_px: float) -> Config:
    data = load_config().to_dict()
    data["sample"]["scale"] = {"name": "fixed_scale", "um_per_px": um_per_px}
    data["logging"]["level"] = "WARNING"
    return Config(data)


class StageTimer:
    """Progress callback of run(): time per stage and per image (main thread view)."""

    def __init__(self):
        self.stage_s = defaultdict(float)
        self.image_s = {}
        self._last = None           # (time, stage, image name)
        self._image_start = {}

    def __call__(self, index: int, total: int, stage: str, name: str):
        now = time.perf_counter()
        if self._last is not None:
            t, last_stage, last_name = self._last
            self.stage_s[last_stage] += now - t
            if last_name and name != last_name:
                self.image_s[last_name] = now - self._image_start[last_name]
        if name and name not in self._image_start:
            self._image_start[name] = now
        self._last = (now, stage, name)


def run_pipeline(images_dir: Path, out_dir: Path, um_per_px: float):
    config = pipeline_config(um_per_px)
    timer = StageTimer()
    print(f"running pipeline on {images_dir} ...")
    try:
        result = run(config, input_path=images_dir, output_dir=out_dir, progress=timer)
    except RunAborted as error:
        sys.exit(f"run aborted: {error}")
    print(f"{result.succeeded} images succeeded, {result.failed} failed, {result.duration_s:.1f} s, "
          f"device {result.device}")
    for error in result.errors:
        print(f"  FAILED {error.image}: {error.stage} / {error.method}: {error.message}")
    return result.results_dir, timer, config


# ---------------------------------------------------------------- ground truth

class GroundTruth:
    def __init__(self, gt_dir: Path, name: str, synth_cfg: dict, meta_image: dict | None, min_visible_px: int):
        um = synth_cfg["image"]["um_per_px"]
        labels = read_image(gt_dir / f"{name}_labels.png")
        # Scale bar box (+ the pipeline's exclusion margin): not part of the image for
        # GT and detection alike, as the scale_bar method would mask it out
        bar_path = gt_dir / f"{name}_scalebar.png"
        self.overlay = np.zeros(labels.shape, bool)
        if bar_path.is_file():
            size = 2 * SCALE_BAR_PAD_PX + 1
            self.overlay = cv2.dilate(read_image(bar_path), np.ones((size, size), np.uint8)) > 0
        self.sample = (labels != RESIN) & ~self.overlay
        self.pore = (labels == PORE) & ~self.overlay
        self.phase = read_image(gt_dir / f"{name}_phase.png") > 0
        self.inst = read_image(gt_dir / f"{name}_pores.png").astype(np.int32)
        self.inst[self.overlay] = 0
        self.seam = self._seams(gt_dir / f"{name}_frames.png")
        edge_um = synth_cfg.get("resin", {}).get("edge_zone_um", 0) if synth_cfg.get("resin", {}).get("enabled") else 0
        self.edge = self._edge_zone(edge_um / um)
        n = int(self.inst.max())
        self.area = np.bincount(self.inst.ravel(), minlength=n + 1)
        self.eq_diameter_um = 2 * np.sqrt(self.area / math.pi) * um
        self.type = np.array([None] * (n + 1), dtype=object)
        if meta_image is not None:
            for pore in meta_image["pores"]:
                self.type[pore["id"]] = pore["type"]
        # Invisible pores: after blur and noise a pore of 1-2 px is brighter than the
        # dark phase and cannot be found by any method. They (+ 1 px) are a don't-care
        # zone for pixel and object scores; the porosity still counts them.
        self.invisible = (self.area < min_visible_px) & (self.area > 0)
        self.invisible[0] = False
        self.dontcare = cv2.dilate(self.invisible[self.inst].view(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def _seams(self, path: Path) -> np.ndarray:
        if not path.is_file():
            return np.zeros_like(self.sample)
        frames = read_image(path)
        seam = np.zeros(frames.shape, bool)
        seam[:, 1:] |= frames[:, 1:] != frames[:, :-1]
        seam[1:, :] |= frames[1:, :] != frames[:-1, :]
        size = 2 * SEAM_HALF_WIDTH_PX + 1
        return cv2.dilate(seam.view(np.uint8), np.ones((size, size), np.uint8)) > 0

    def _edge_zone(self, width_px: float) -> np.ndarray:
        if width_px <= 0 or self.sample.all():
            return np.zeros_like(self.sample)
        distance = cv2.distanceTransform(self.sample.view(np.uint8), cv2.DIST_L2, 5)
        return self.sample & (distance <= width_px)


# ---------------------------------------------------------------- comparison helpers

def overlap(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (id_a, id_b) with their number of shared pixels, both ids > 0."""
    both = (a > 0) & (b > 0)
    av, bv = a[both].astype(np.int64), b[both].astype(np.int64)
    code = av * (int(b.max()) + 1) + bv
    pairs, counts = np.unique(code, return_counts=True)
    return pairs // (int(b.max()) + 1), pairs % (int(b.max()) + 1), counts


def compact_ids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels renumbered 1..n (pore ids may have gaps), plus the original id per new id."""
    ids, inverse = np.unique(labels, return_inverse=True)
    if ids[0] != 0:
        ids, inverse = np.concatenate([[0], ids]), inverse + 1
    return inverse.reshape(labels.shape).astype(np.int32), ids


def categorize(mask: np.ndarray, layers: list[tuple[str, np.ndarray]], rest: str) -> tuple[dict, np.ndarray]:
    """Assign each pixel of mask to the first layer containing it. Returns counts and category image."""
    counts, cat = {}, np.full(mask.shape, len(layers), np.uint8)
    remaining = mask.copy()
    for i, (name, layer) in enumerate(layers):
        hit = remaining & layer
        counts[name] = int(np.count_nonzero(hit))
        cat[hit] = i
        remaining &= ~layer
    counts[rest] = int(np.count_nonzero(remaining))
    return counts, cat


def size_class(d_um: float) -> str:
    for lo, hi in zip(SIZE_BINS_UM, SIZE_BINS_UM[1:]):
        if d_um < hi:
            return f"{lo:g}-{hi:g} um" if hi != math.inf else f">{lo:g} um"
    return "?"


def ratio(a, b):
    return a / b if b else float("nan")


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else float("nan")


# ---------------------------------------------------------------- per image

def evaluate_image(gt: GroundTruth, folder: Path, iou_threshold: float) -> dict:
    labels = read_image(folder / "labels.tif").astype(np.int32)
    rejected_labels = read_image(folder / "rejected_labels.tif").astype(np.int32)
    roi = read_image(folder / "roi_mask.tif") > 0
    rejected = read_table(folder / "rejected.json") if (folder / "rejected.json").is_file() else pd.DataFrame()
    labels[gt.overlay] = 0
    rejected_labels[gt.overlay] = 0
    roi &= ~gt.overlay
    pred = labels > 0
    res = {}

    # --- ROI
    inter, union = np.count_nonzero(roi & gt.sample), np.count_nonzero(roi | gt.sample)
    res["roi_iou"] = ratio(inter, union)
    res["resin_as_sample_px"] = int(np.count_nonzero(roi & ~gt.sample))
    res["sample_missed_px"] = int(np.count_nonzero(gt.sample & ~roi))
    res["gt_sample_px"] = int(np.count_nonzero(gt.sample))
    res["gt_resin_px"] = int(gt.sample.size - res["gt_sample_px"])

    # --- porosity (each relative to its own sample area)
    res["gt_pore_px"] = int(np.count_nonzero(gt.pore))
    res["det_pore_px"] = int(np.count_nonzero(pred))
    res["det_roi_px"] = int(np.count_nonzero(roi))
    res["gt_porosity"] = 100 * ratio(res["gt_pore_px"], res["gt_sample_px"])
    res["det_porosity"] = 100 * ratio(res["det_pore_px"], res["det_roi_px"])

    # --- pixels (outside the don't-care zone of invisible pores)
    pred_px, pore_px = pred & ~gt.dontcare, gt.pore & ~gt.dontcare
    tp = int(np.count_nonzero(pred_px & pore_px))
    res.update(tp_px=tp, fp_px=int(np.count_nonzero(pred_px)) - tp, fn_px=int(np.count_nonzero(pore_px)) - tp)

    rim_out = cv2.dilate(gt.pore.view(np.uint8), np.ones((3, 3), np.uint8)) > 0
    fp_counts, fp_cat = categorize(pred_px & ~gt.pore, [("resin", ~gt.sample), ("pore_rim", rim_out),
                                                        ("seam", gt.seam), ("sample_edge", gt.edge),
                                                        ("dark_phase", gt.phase)], "bright_matrix")
    res["fp_cat"] = fp_counts
    rim_in = gt.pore & ~(cv2.erode(gt.pore.view(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    res["fn_cat"], _ = categorize(pore_px & ~pred, [("outside_roi", ~roi), ("pore_rim", rim_in)], "interior")

    # --- invisible GT pores: only reported, how many were found anyway
    hit = np.bincount(gt.inst[pred], minlength=gt.area.size) > 0
    res["invisible_gt"] = int(np.count_nonzero(gt.invisible))
    res["invisible_found"] = int(np.count_nonzero(gt.invisible & hit))

    # --- objects: visible GT pores vs. counted pores (objects mostly in the don't-care zone left out)
    def drop_dontcare(obj_labels):
        obj_c, _ = compact_ids(obj_labels)
        n = int(obj_c.max())
        in_dc = np.bincount(obj_c[gt.dontcare], minlength=n + 1)
        drop = in_dc / np.maximum(np.bincount(obj_c.ravel(), minlength=n + 1), 1) >= 0.5
        drop[0] = False
        return np.where(drop[obj_c], 0, obj_labels)

    labels = drop_dontcare(labels)
    inst = np.where(gt.invisible[gt.inst], 0, gt.inst)
    visible = (gt.area > 0) & ~gt.invisible
    visible[0] = False
    pred_c, _ = compact_ids(labels)
    n_gt, n_pred = int(gt.inst.max()), int(pred_c.max())
    n_visible = int(np.count_nonzero(visible))
    pred_area = np.bincount(pred_c.ravel(), minlength=n_pred + 1)
    g, p, shared = overlap(inst, pred_c)
    iou = shared / (gt.area[g] + pred_area[p] - shared)
    good = iou >= iou_threshold
    matched_gt = np.zeros(n_gt + 1, bool)
    matched_gt[g[good]] = True
    matched_pred = np.zeros(n_pred + 1, bool)
    matched_pred[p[good]] = True
    area_err = pred_area[p[good]] / gt.area[g[good]] - 1

    # major partner: for each pred object the GT pore it overlaps most, and vice versa
    order = np.argsort(shared)
    major_gt_of_pred = np.zeros(n_pred + 1, np.int64)
    major_gt_of_pred[p[order]] = g[order]
    major_pred_of_gt = np.zeros(n_gt + 1, np.int64)
    major_pred_of_gt[g[order]] = p[order]
    cover_of_gt = np.zeros(n_gt + 1)
    np.maximum.at(cover_of_gt, g, shared / gt.area[g])
    pieces = np.bincount(major_gt_of_pred[1:], minlength=n_gt + 1)
    pieces[0] = 0
    merged_into = np.bincount(major_pred_of_gt[1:][cover_of_gt[1:] >= 0.5], minlength=n_pred + 1)
    merged_into[0] = 0

    res.update(gt_objects=n_visible, det_objects=n_pred, matched=int(matched_gt[1:].sum()),
               split_gt=int(np.count_nonzero(pieces >= 2)), merged_det=int(np.count_nonzero(merged_into >= 2)),
               missed_gt=int(n_visible - np.count_nonzero(np.bincount(g, minlength=n_gt + 1)[1:])),
               phantom_det=int(n_pred - np.count_nonzero(np.bincount(p, minlength=n_pred + 1)[1:])),
               area_err=area_err)

    groups = defaultdict(lambda: [0, 0, 0, 0])      # gt count, matched, gt px, detected px of those
    det_px_of_gt = np.bincount(gt.inst[pred], minlength=n_gt + 1)
    for i in np.nonzero(visible)[0]:
        for key in (("type", gt.type[i]), ("size", size_class(gt.eq_diameter_um[i]))):
            if key[1] is None:
                continue
            group = groups[key]
            group[0] += 1
            group[1] += int(matched_gt[i])
            group[2] += int(gt.area[i])
            group[3] += int(det_px_of_gt[i])
    res["groups"] = dict(groups)

    # --- objects judged by their pixels: real pore (>= 50 % GT pore) or false object
    def judge(obj_labels):
        obj_c, ids = compact_ids(obj_labels)
        n = int(obj_c.max())
        area = np.bincount(obj_c.ravel(), minlength=n + 1)
        on_pore = np.bincount(obj_c[gt.pore], minlength=n + 1)
        cat = np.bincount(obj_c.ravel() * (len(FP_CATEGORIES)) + np.minimum(fp_cat.ravel(), len(FP_CATEGORIES) - 1),
                          weights=(~gt.pore).ravel(), minlength=(n + 1) * len(FP_CATEGORIES))
        cat = cat.reshape(n + 1, len(FP_CATEGORIES))
        return ids, area, on_pore / np.maximum(area, 1) >= 0.5, cat.argmax(axis=1), on_pore

    # categories of the non-pore pixels of rejected objects (outside the counted mask: own image)
    rej_mask = rejected_labels > 0
    _, fp_cat_rej = categorize(rej_mask & ~gt.pore, [("resin", ~gt.sample), ("pore_rim", rim_out),
                                                    ("seam", gt.seam), ("sample_edge", gt.edge),
                                                    ("dark_phase", gt.phase)], "bright_matrix")
    fp_cat = np.where(pred, fp_cat, fp_cat_rej)

    ids, area, real, origin, _ = judge(labels)
    false_counted = defaultdict(list)
    for i in range(1, len(ids)):
        if not real[i]:
            false_counted[FP_CATEGORIES[origin[i]]].append(int(area[i]))
    res["false_counted"] = dict(false_counted)

    res["filters"] = defaultdict(lambda: [0, 0, 0, 0])   # rejected, real pores, false objects, pore px lost
    if len(rejected) and rej_mask.any():
        ids, area, real, _, on_pore = judge(drop_dontcare(rejected_labels))
        filter_of = dict(zip(rejected["pore_id"], rejected["filter"]))
        for i in range(1, len(ids)):
            stats = res["filters"][filter_of.get(int(ids[i]), "?")]
            stats[0] += 1
            stats[1 if real[i] else 2] += 1
            stats[3] += int(on_pore[i])
    res["filters"] = dict(res["filters"])
    return res


# ---------------------------------------------------------------- printing

def pct(v):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f}"


def print_section(title: str):
    print(f"\n=== {title} " + "=" * max(0, 78 - len(title)))


def print_table(rows: list[dict]):
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def print_report(results: dict[str, dict], timer: StageTimer | None, iou_threshold: float, min_visible_px: int):
    names = list(results)
    r_all = list(results.values())

    print_section("per image")
    rows = []
    for name, r in results.items():
        p, rc = ratio(r["tp_px"], r["tp_px"] + r["fp_px"]), ratio(r["tp_px"], r["tp_px"] + r["fn_px"])
        op, orc = ratio(r["matched"], r["det_objects"]), ratio(r["matched"], r["gt_objects"])
        rows.append({"image": name, "roi_iou%": pct(r["roi_iou"]),
                     "por_gt%": f"{r['gt_porosity']:.3f}", "por_det%": f"{r['det_porosity']:.3f}",
                     "pix_P%": pct(p), "pix_R%": pct(rc), "pix_F1%": pct(f1(p, rc)),
                     "obj_gt": r["gt_objects"], "obj_det": r["det_objects"], "match": r["matched"],
                     "obj_P%": pct(op), "obj_R%": pct(orc), "split": r["split_gt"], "merge": r["merged_det"],
                     "time_s": f"{timer.image_s.get(name, float('nan')):.2f}" if timer else "-"})
    print_table(rows)

    def total(key):
        return sum(r[key] for r in r_all)

    print_section("ROI (sample vs. embedding resin)")
    iou = np.array([r["roi_iou"] for r in r_all])
    print(f"IoU mean {pct(iou.mean())} %, min {pct(iou.min())} % ({names[int(iou.argmin())]})")
    print(f"resin detected as sample : {pct(ratio(total('resin_as_sample_px'), total('gt_resin_px')))} % of GT resin")
    print(f"sample detected as resin : {pct(ratio(total('sample_missed_px'), total('gt_sample_px')))} % of GT sample")

    print_section("porosity")
    gt_p = np.array([r["gt_porosity"] for r in r_all])
    det_p = np.array([r["det_porosity"] for r in r_all])
    err = det_p - gt_p
    rel = err / np.where(gt_p > 0, gt_p, np.nan)
    print(f"all images together : GT {100 * ratio(total('gt_pore_px'), total('gt_sample_px')):.3f} %, "
          f"detected {100 * ratio(total('det_pore_px'), total('det_roi_px')):.3f} %")
    print(f"error (det - GT)    : mean {err.mean():+.3f} pp, mean |err| {np.abs(err).mean():.3f} pp, "
          f"max |err| {np.abs(err).max():.3f} pp ({names[int(np.abs(err).argmax())]})")
    print(f"relative error      : mean {np.nanmean(rel) * 100:+.1f} %, mean |rel| {np.nanmean(np.abs(rel)) * 100:.1f} %")
    print(f"correlation GT/det  : {np.corrcoef(gt_p, det_p)[0, 1]:.3f}" if len(r_all) > 1 else "")

    print_section("pixels (all images summed)")
    tp, fp, fn = total("tp_px"), total("fp_px"), total("fn_px")
    p, rc = ratio(tp, tp + fp), ratio(tp, tp + fn)
    print(f"TP {tp}  FP {fp}  FN {fn}")
    print(f"precision {pct(p)} %  recall {pct(rc)} %  F1/Dice {pct(f1(p, rc))} %  IoU {pct(ratio(tp, tp + fp + fn))} %")

    print("\nfalse-positive pixels by origin (first matching layer counts):")
    fp_rows = [{"origin": c, "px": sum(r["fp_cat"][c] for r in r_all),
                "% of FP": pct(ratio(sum(r["fp_cat"][c] for r in r_all), fp)),
                "% of detected": pct(ratio(sum(r["fp_cat"][c] for r in r_all), tp + fp))} for c in FP_CATEGORIES]
    print_table(fp_rows)
    print("\nmissed pore pixels by reason:")
    print_table([{"reason": c, "px": sum(r["fn_cat"][c] for r in r_all),
                  "% of FN": pct(ratio(sum(r["fn_cat"][c] for r in r_all), fn)),
                  "% of GT pore": pct(ratio(sum(r["fn_cat"][c] for r in r_all), tp + fn))} for c in FN_CATEGORIES])

    print_section(f"objects (match = IoU >= {iou_threshold:g})")
    n_gt, n_det, m = total("gt_objects"), total("det_objects"), total("matched")
    op, orc = ratio(m, n_det), ratio(m, n_gt)
    n_inv = total("invisible_gt")
    print(f"invisible GT pores (< {min_visible_px} px, don't care): {n_inv}, "
          f"touched by a detection anyway: {total('invisible_found')} ({pct(ratio(total('invisible_found'), n_inv))} %)")
    print(f"visible GT pores {n_gt}, counted {n_det} ({n_det - n_gt:+d}), matched {m}")
    print(f"precision {pct(op)} %  recall {pct(orc)} %  F1 {pct(f1(op, orc))} %")
    print(f"GT pores missed completely : {total('missed_gt')}")
    print(f"counted without any GT pore: {total('phantom_det')}")
    print(f"GT pores split into >= 2   : {total('split_gt')}")
    print(f"counted pores holding >= 2 GT pores (merged): {total('merged_det')}")
    errs = np.concatenate([r["area_err"] for r in r_all]) if r_all else np.array([])
    if errs.size:
        print(f"area of matched pores (det / GT - 1): median {np.median(errs) * 100:+.1f} %, "
              f"mean {errs.mean() * 100:+.1f} %, mean |err| {np.abs(errs).mean() * 100:.1f} %")

    groups = defaultdict(lambda: [0, 0, 0, 0])
    for r in r_all:
        for key, v in r["groups"].items():
            groups[key] = [a + b for a, b in zip(groups[key], v)]
    for kind, title in (("type", "per pore type"), ("size", "per size class (GT equivalent diameter)")):
        keys = [k for k in groups if k[0] == kind]
        if kind == "size":
            keys.sort(key=lambda k: float(k[1].lstrip(">").split("-")[0].split()[0]))
        if not keys:
            continue
        print(f"\n{title}:")
        print_table([{kind: k[1], "gt": groups[k][0], "matched": groups[k][1],
                      "obj_recall%": pct(ratio(groups[k][1], groups[k][0])),
                      "pix_recall%": pct(ratio(groups[k][3], groups[k][2])),
                      "gt_px": groups[k][2]} for k in keys])

    print_section("filters (rejected objects; real pore = >= 50 % of its pixels are GT pore)")
    filters = defaultdict(lambda: [0, 0, 0, 0])
    for r in r_all:
        for name, v in r["filters"].items():
            filters[name] = [a + b for a, b in zip(filters[name], v)]
    if filters:
        print_table([{"filter": k, "rejected": v[0], "real pores (wrong)": v[1], "false obj (right)": v[2],
                      "correct%": pct(ratio(v[2], v[0])), "pore px lost": v[3],
                      "% of GT pore": pct(ratio(v[3], tp + fn))} for k, v in filters.items()])
    else:
        print("no filter rejected anything")

    print_section("false objects still counted (candidates for filters)")
    false_counted = defaultdict(list)
    for r in r_all:
        for c, areas in r["false_counted"].items():
            false_counted[c].extend(areas)
    n_false = sum(len(v) for v in false_counted.values())
    print(f"{n_false} of {n_det} counted objects are false ({pct(ratio(n_false, n_det))} %)")
    if false_counted:
        print_table([{"origin": c, "objects": len(a), "px": int(sum(a)), "median area px": int(np.median(a)),
                      "max area px": int(max(a))} for c, a in sorted(false_counted.items(), key=lambda x: -len(x[1]))])

    if timer and timer.stage_s:
        print_section("timing (main thread, summed over all images)")
        whole = sum(timer.stage_s.values())
        print_table([{"stage": s, "s": f"{t:.2f}", "%": pct(ratio(t, whole)),
                      "s/image": f"{t / max(len(r_all), 1):.3f}"}
                     for s, t in sorted(timer.stage_s.items(), key=lambda x: -x[1])])


# ---------------------------------------------------------------- main

def main() -> int:
    synth_default = yaml.safe_load((HERE / "synth_config.yaml").read_text(encoding="utf-8"))["run"]
    base = HERE / synth_default["out_dir"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=base / synth_default["id"], help="synthetic image folder (= ID)")
    parser.add_argument("--gt", type=Path, help="ground-truth folder (default: <images>_gt)")
    parser.add_argument("--out", type=Path, default=base / "pipeline", help="output folder of the pipeline run")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for object matching")
    parser.add_argument("--min-visible-px", type=int, default=MIN_VISIBLE_PX,
                        help="smaller GT pores are invisible: don't care for pixel and object scores (0 = off)")
    parser.add_argument("--no-run", action="store_true", help="only evaluate existing pipeline results")
    args = parser.parse_args()

    images_dir = args.images.resolve()
    gt_dir = (args.gt or images_dir.with_name(images_dir.name + "_gt")).resolve()
    synth_cfg, meta = load_synth_settings(gt_dir)
    um_per_px = synth_cfg["image"]["um_per_px"]
    meta_images = {m["name"]: m for m in meta["images"]} if meta else {}

    timer = None
    if args.no_run:
        results_dir = args.out.resolve() / f"{images_dir.name}_results"
    else:
        results_dir, timer, _ = run_pipeline(images_dir, args.out.resolve(), um_per_px)

    results = {}
    for path in sorted(images_dir.glob("*.png")):
        folder = results_dir / f"{images_dir.name}_{path.stem}"
        if not (folder / "labels.tif").is_file():
            print(f"  no results for {path.stem} – skipped")
            continue
        gt = GroundTruth(gt_dir, path.stem, synth_cfg, meta_images.get(path.stem), args.min_visible_px)
        name = f"{images_dir.name}_{path.stem}"
        results[name] = evaluate_image(gt, folder, args.iou)
    if not results:
        print("nothing to evaluate")
        return 1

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print_report(results, timer, args.iou, args.min_visible_px)
    return 0


if __name__ == "__main__":
    sys.exit(main())
