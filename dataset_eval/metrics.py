"""Comparison of one image: ground-truth mask vs. the pipeline results of that image.

Area of the comparison:
    scale bar     the scale overlay box of the pipeline (images.json, scale_box_px) is left out
                  for GT and detection alike – the masks have no pores there anyway
    porosity      GT and detection both relative to the ROI detected by the pipeline, so only the
                  pores are compared; GT pore area outside the ROI is reported as a check value
    pixels        whole image (outside the scale box): GT pores outside the ROI are missed pixels
                  (reason "outside_roi")

GT pores are the connected white areas of the mask (same connectivity as the pipeline's labeling).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FP_ORIGINS = ["pore_rim", "matrix"]                     # where false-positive pixels lie
FN_REASONS = ["outside_roi", "pore_rim", "interior"]    # why GT pore pixels are missed


@dataclass(frozen=True)
class Pair:
    name: str           # <ID>_<stem> = image folder of the pipeline results
    image: Path
    mask: Path


@dataclass
class ImageData:
    """Everything needed to compare one image, all (H, W) in full image size."""

    gt: np.ndarray              # bool, GT pore
    labels: np.ndarray          # int32, counted pores of the pipeline
    rejected_labels: np.ndarray  # int32, rejected pores of the pipeline
    rejected: pd.DataFrame      # pore_id, filter, reason
    roi: np.ndarray             # bool, detected sample (scale box removed)
    valid: np.ndarray           # bool, False = scale box (left out)


@dataclass
class ImageResult:
    values: dict                # one row of the per-image table
    gt_pores: pd.DataFrame      # diameter_px, matched
    det_pores: pd.DataFrame     # diameter_px, matched, real
    pairs: pd.DataFrame         # gt_diameter_px, area_error (det / GT - 1) of matched pores
    filters: pd.DataFrame       # filter, rejected, real, false, pore_px_lost


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


def read_mask(path: Path, threshold: int) -> np.ndarray:
    """White = pore; color masks: brightest channel, 16 bit scaled to 0-255."""
    mask = read_image(path)
    if mask.ndim == 3:
        mask = mask[..., :3].max(axis=2)
    if mask.dtype == np.uint16:
        mask = mask // 257
    return mask > threshold


def load(pair: Pair, folder: Path, row: dict, threshold: int) -> ImageData:
    """GT mask and pipeline results of one image; raises ValueError if the sizes differ."""
    gt = read_mask(pair.mask, threshold)
    labels = read_image(folder / "labels.tif").astype(np.int32)
    if gt.shape != labels.shape:
        raise ValueError(f"mask {gt.shape[1]} x {gt.shape[0]} px, image {labels.shape[1]} x {labels.shape[0]} px")
    rejected_labels = read_image(folder / "rejected_labels.tif").astype(np.int32)
    roi = read_image(folder / "roi_mask.tif") > 0
    rejected_path = folder / "rejected.json"
    rejected = read_table(rejected_path) if rejected_path.is_file() else pd.DataFrame(columns=["pore_id", "filter"])

    valid = np.ones(gt.shape, bool)
    box = row.get("scale_box_px")
    if isinstance(box, list):
        top, bottom, left, right = box
        valid[top:bottom, left:right] = False
    gt &= valid
    roi &= valid
    labels[~valid] = 0
    rejected_labels[~valid] = 0
    return ImageData(gt, labels, rejected_labels, rejected, roi, valid)


# ---------------------------------------------------------------- helpers

def overlap(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (id_a, id_b) with their number of shared pixels, both ids > 0."""
    both = (a > 0) & (b > 0)
    base = int(b.max()) + 1
    code = a[both].astype(np.int64) * base + b[both]
    pairs, counts = np.unique(code, return_counts=True)
    return pairs // base, pairs % base, counts


def compact_ids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels renumbered 1..n (pore ids may have gaps), plus the original id per new id."""
    ids, inverse = np.unique(labels, return_inverse=True)
    if ids[0] != 0:
        ids, inverse = np.concatenate([[0], ids]), inverse + 1
    return inverse.reshape(labels.shape).astype(np.int32), ids


def diameter(area_px: np.ndarray) -> np.ndarray:
    """Equivalent diameter (circle of the same area) in px."""
    return 2 * np.sqrt(area_px / math.pi)


def ratio(a, b) -> float:
    return a / b if b else float("nan")


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else float("nan")


def disk(radius_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


# ---------------------------------------------------------------- one image

def evaluate(data: ImageData, match_iou: float, rim_px: int, connectivity: int) -> ImageResult:
    gt, roi, det = data.gt, data.roi, data.labels > 0
    v: dict = {}

    # --- porosity, both relative to the detected ROI
    roi_px = int(np.count_nonzero(roi))
    gt_px = int(np.count_nonzero(gt))
    gt_in_roi = int(np.count_nonzero(gt & roi))
    det_px = int(np.count_nonzero(det))
    v["roi_px"] = roi_px
    v["roi_percent_of_image"] = 100 * ratio(roi_px, int(np.count_nonzero(data.valid)))
    v["gt_pore_px"] = gt_px
    v["gt_outside_roi_percent"] = 100 * ratio(gt_px - gt_in_roi, gt_px) if gt_px else 0.0
    v["gt_porosity_percent"] = 100 * ratio(gt_in_roi, roi_px)
    v["det_porosity_percent"] = 100 * ratio(det_px, roi_px)
    v["porosity_error_pp"] = v["det_porosity_percent"] - v["gt_porosity_percent"]
    v["porosity_error_rel_percent"] = 100 * ratio(v["porosity_error_pp"], v["gt_porosity_percent"])

    # --- pixels
    tp = int(np.count_nonzero(det & gt))
    fp, fn = det_px - tp, gt_px - tp
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    v.update(tp_px=tp, fp_px=fp, fn_px=fn, pixel_precision=precision, pixel_recall=recall,
             pixel_f1=f1(precision, recall), pixel_iou=ratio(tp, tp + fp + fn))

    near_gt = cv2.dilate(gt.view(np.uint8), disk(rim_px)) > 0
    fp_mask = det & ~gt
    v["fp_pore_rim_px"] = int(np.count_nonzero(fp_mask & near_gt))
    v["fp_matrix_px"] = fp - v["fp_pore_rim_px"]
    fn_mask = gt & ~det
    gt_core = cv2.erode(gt.view(np.uint8), disk(rim_px)) > 0
    v["fn_outside_roi_px"] = int(np.count_nonzero(fn_mask & ~roi))
    v["fn_pore_rim_px"] = int(np.count_nonzero(fn_mask & roi & ~gt_core))
    v["fn_interior_px"] = fn - v["fn_outside_roi_px"] - v["fn_pore_rim_px"]

    # --- objects
    n_gt, gt_inst = cv2.connectedComponents(gt.view(np.uint8), connectivity=connectivity, ltype=cv2.CV_32S)
    n_gt -= 1
    det_c, _ = compact_ids(data.labels)
    n_det = int(det_c.max())
    gt_area = np.bincount(gt_inst.ravel(), minlength=n_gt + 1)
    det_area = np.bincount(det_c.ravel(), minlength=n_det + 1)
    g, p, shared = overlap(gt_inst, det_c)
    iou = shared / (gt_area[g] + det_area[p] - shared)
    good = iou >= match_iou                 # with match_iou >= 0.5 at most one partner each
    matched_gt = np.zeros(n_gt + 1, bool)
    matched_gt[g[good]] = True
    matched_det = np.zeros(n_det + 1, bool)
    matched_det[p[good]] = True

    # major partner: for each detected pore the GT pore it overlaps most, and vice versa
    order = np.argsort(shared)
    major_gt_of_det = np.zeros(n_det + 1, np.int64)
    major_gt_of_det[p[order]] = g[order]
    major_det_of_gt = np.zeros(n_gt + 1, np.int64)
    major_det_of_gt[g[order]] = p[order]
    cover_of_gt = np.zeros(n_gt + 1)
    np.maximum.at(cover_of_gt, g, shared / gt_area[g])
    pieces = np.bincount(major_gt_of_det[1:], minlength=n_gt + 1)
    pieces[0] = 0
    merged_into = np.bincount(major_det_of_gt[1:][cover_of_gt[1:] >= 0.5], minlength=n_det + 1)
    merged_into[0] = 0
    on_gt = np.bincount(det_c[gt], minlength=n_det + 1)
    real = on_gt / np.maximum(det_area, 1) >= 0.5       # detected object is mostly GT pore

    matched = int(matched_gt[1:].sum())
    object_precision, object_recall = ratio(matched, n_det), ratio(matched, n_gt)
    v.update(gt_pores=n_gt, det_pores=n_det, matched_pores=matched,
             object_precision=object_precision, object_recall=object_recall,
             object_f1=f1(object_precision, object_recall),
             missed_pores=int(n_gt - np.count_nonzero(np.bincount(g, minlength=n_gt + 1)[1:])),
             phantom_pores=int(n_det - np.count_nonzero(np.bincount(p, minlength=n_det + 1)[1:])),
             split_pores=int(np.count_nonzero(pieces >= 2)),
             merged_pores=int(np.count_nonzero(merged_into >= 2)),
             false_objects=int(np.count_nonzero(~real[1:])))

    gt_pores = pd.DataFrame({"diameter_px": diameter(gt_area[1:]), "matched": matched_gt[1:]})
    det_pores = pd.DataFrame({"diameter_px": diameter(det_area[1:]), "area_px": det_area[1:],
                              "matched": matched_det[1:], "real": real[1:]})
    pairs = pd.DataFrame({"gt_diameter_px": diameter(gt_area[g[good]]),
                          "area_error": det_area[p[good]] / gt_area[g[good]] - 1})
    return ImageResult(v, gt_pores, det_pores, pairs, _filters(data))


def _filters(data: ImageData) -> pd.DataFrame:
    """Per filter: rejected objects, of them real pores (>= 50 % GT pore) and false objects, GT pore px lost."""
    columns = ["filter", "rejected", "real", "false", "pore_px_lost"]
    if not len(data.rejected) or not data.rejected_labels.any():
        return pd.DataFrame(columns=columns)
    rej_c, ids = compact_ids(data.rejected_labels)
    area = np.bincount(rej_c.ravel())
    on_gt = np.bincount(rej_c[data.gt], minlength=area.size)
    filter_of = dict(zip(data.rejected["pore_id"].astype(int), data.rejected["filter"]))
    table = pd.DataFrame({"filter": [filter_of.get(int(i), "?") for i in ids[1:]],
                          "real": on_gt[1:] / np.maximum(area[1:], 1) >= 0.5, "pore_px_lost": on_gt[1:]})
    table["false"] = ~table["real"]
    table["rejected"] = 1
    return table.groupby("filter", as_index=False)[["rejected", "real", "false", "pore_px_lost"]].sum()[columns]


def classes(data: ImageData) -> np.ndarray:
    """Class per pixel for the overlay: 0 = left out (outside ROI / scale box, no pore), 1 = matrix,
    2 = correct pore (TP), 3 = falsely detected (FP), 4 = missed (FN)."""
    det = data.labels > 0
    result = np.where(data.roi, 1, 0).astype(np.uint8)
    result[det & data.gt] = 2
    result[det & ~data.gt] = 3
    result[data.gt & ~det] = 4
    return result
