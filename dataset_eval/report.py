"""Results of the evaluation: summary over all images, Excel, plots, overlays of the worst images.

The look of the plots is the one of the program (pore_detection.core.plotting: STYLE, COLORS, size),
only read from there.
"""

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pore_detection.core.plotting import COLORS, DPI, SIZE_IN, STYLE

from metrics import FN_REASONS, FP_ORIGINS, ImageResult, ratio, f1

# overlay: white background, errors as the only colors (orange / blue: safe with red-green weakness)
CLASS_COLORS = {
    0: ("#e9ecef", "left out (outside ROI / scale)"),
    1: ("#ffffff", "matrix"),
    2: ("#495057", "pore, correct"),
    3: ("#E69F00", "falsely detected"),
    4: ("#0072B2", "missed"),
}
OVERLAY_MIN_PANEL_PX = 600      # small patches are enlarged (whole factors, pixels stay sharp)
OVERLAY_HEADER_PX = 64
OVERLAY_GAP_PX = 12


# ---------------------------------------------------------------- tables

def size_unit(images: pd.DataFrame) -> tuple[str, pd.Series]:
    """um if every image has a scale, else px; returns the unit and the factor px -> unit per image."""
    if images["pixel_size_um"].notna().all():
        return "um", images["pixel_size_um"]
    return "px", pd.Series(1.0, index=images.index)


def pore_tables(results: dict[str, ImageResult], factor: dict[str, float]):
    """GT pores, detected pores and matched pairs of all images, diameters in the size unit."""
    def collect(attr: str, column: str) -> pd.DataFrame:
        parts = []
        for name, result in results.items():
            table = getattr(result, attr).copy()
            table.insert(0, "image", name)
            table["diameter"] = table.pop(column) * factor[name]
            parts.append(table)
        return pd.concat(parts, ignore_index=True)

    return collect("gt_pores", "diameter_px"), collect("det_pores", "diameter_px"), collect("pairs", "gt_diameter_px")


def class_labels(edges: list[float], unit: str) -> list[str]:
    return [f"{lo:g}-{hi:g} {unit}" for lo, hi in zip(edges, edges[1:])] + [f">{edges[-1]:g} {unit}"]


def class_of(values: pd.Series, edges: list[float], unit: str) -> pd.Series:
    labels = class_labels(edges, unit)
    return pd.cut(values, list(edges) + [math.inf], right=False, labels=labels)


def size_class_table(gt: pd.DataFrame, det: pd.DataFrame, pairs: pd.DataFrame, edges: list[float],
                     unit: str) -> pd.DataFrame:
    gt_class, det_class = class_of(gt["diameter"], edges, unit), class_of(det["diameter"], edges, unit)
    pair_class = class_of(pairs["diameter"], edges, unit)
    rows = []
    for label in class_labels(edges, unit):
        g, d = gt[gt_class == label], det[det_class == label]
        errors = pairs.loc[pair_class == label, "area_error"]
        rows.append({"size_class": label, "gt_pores": len(g), "gt_found": int(g["matched"].sum()),
                     "recall": ratio(int(g["matched"].sum()), len(g)),
                     "det_pores": len(d), "det_correct": int(d["matched"].sum()),
                     "precision": ratio(int(d["matched"].sum()), len(d)),
                     "area_error_median_percent": 100 * errors.median() if len(errors) else float("nan")})
    return pd.DataFrame(rows)


def filter_table(results: dict[str, ImageResult], total_gt_px: int) -> pd.DataFrame:
    parts = [r.filters for r in results.values() if len(r.filters)]
    if not parts:
        return pd.DataFrame(columns=["filter", "rejected", "real", "false", "correct_percent",
                                     "pore_px_lost", "percent_of_gt_pore"])
    table = pd.concat(parts).groupby("filter", as_index=False).sum()
    table["correct_percent"] = 100 * table["false"] / table["rejected"]
    table["percent_of_gt_pore"] = 100 * table["pore_px_lost"] / total_gt_px if total_gt_px else float("nan")
    return table.rename(columns={"real": "real_pores_wrong", "false": "false_objects_right"})


def error_table(images: pd.DataFrame) -> pd.DataFrame:
    fp, fn = images["fp_px"].sum(), images["fn_px"].sum()
    rows = [{"kind": "falsely detected", "where": o, "px": int(images[f"fp_{o}_px"].sum()),
             "percent_of_kind": 100 * ratio(images[f"fp_{o}_px"].sum(), fp)} for o in FP_ORIGINS]
    rows += [{"kind": "missed", "where": r, "px": int(images[f"fn_{r}_px"].sum()),
              "percent_of_kind": 100 * ratio(images[f"fn_{r}_px"].sum(), fn)} for r in FN_REASONS]
    return pd.DataFrame(rows)


def summary(images: pd.DataFrame, det: pd.DataFrame, pairs: pd.DataFrame, settings: dict,
            unit: str, counts: dict) -> pd.DataFrame:
    """Key figures over all images: (value, meaning) rows."""
    tp, fp, fn = images["tp_px"].sum(), images["fp_px"].sum(), images["fn_px"].sum()
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    roi = images["roi_px"].sum()
    gt_in_roi = (images["gt_porosity_percent"] * images["roi_px"] / 100).sum()
    det_px = (images["det_porosity_percent"] * images["roi_px"] / 100).sum()
    err = images["porosity_error_pp"]
    n_gt, n_det, m = images["gt_pores"].sum(), images["det_pores"].sum(), images["matched_pores"].sum()
    op, orc = ratio(m, n_det), ratio(m, n_gt)
    corr = images[["gt_porosity_percent", "det_porosity_percent"]].corr().iloc[0, 1] if len(images) > 2 else None
    rows = [
        ("images evaluated", len(images), "image + mask + pipeline result"),
        ("images skipped", counts["skipped"], "see sheet Skipped"),
        ("images flagged (ROI)", int(images["roi_warning"].sum()),
         f"more than {settings['roi_warn_percent']:g} % of the GT pore area outside the detected ROI"),
        ("size unit", unit, "um if every image has a readable scale, else px"),
        ("porosity GT, all images [%]", 100 * ratio(gt_in_roi, roi), "GT pore area / ROI area, all images summed"),
        ("porosity detected, all images [%]", 100 * ratio(det_px, roi), "counted pore area / ROI area"),
        ("porosity error mean [pp]", err.mean(), "detected - GT per image, mean (bias)"),
        ("porosity error mean |abs| [pp]", err.abs().mean(), "mean absolute error per image"),
        ("porosity error SD [pp]", err.std(), "spread of the error"),
        ("porosity error max |abs| [pp]", err.abs().max(), images.loc[err.abs().idxmax(), "image"] if len(err) else ""),
        ("porosity correlation GT / detected", corr, "Pearson r over the images"),
        ("pixel precision", p, "detected pore pixels that are GT pore"),
        ("pixel recall", r, "GT pore pixels that are detected"),
        ("pixel F1 / Dice", f1(p, r), "all images summed"),
        ("pixel IoU", ratio(tp, tp + fp + fn), "all images summed"),
        ("pixel F1 per image, median", images["pixel_f1"].median(), ""),
        ("pixel F1 per image, minimum", images["pixel_f1"].min(),
         images.loc[images["pixel_f1"].idxmin(), "image"] if images["pixel_f1"].notna().any() else ""),
        ("GT pores", int(n_gt), "connected white areas of the masks"),
        ("detected pores", int(n_det), "counted by the pipeline"),
        ("matched pores", int(m), f"same pore: IoU >= {settings['match_iou']:g}"),
        ("object precision", op, "detected pores that match a GT pore"),
        ("object recall", orc, "GT pores that are found"),
        ("object F1", f1(op, orc), ""),
        ("GT pores missed completely", int(images["missed_pores"].sum()), "no detected pixel on them"),
        ("detected without any GT pore", int(images["phantom_pores"].sum()), "no GT pixel under them"),
        ("GT pores split into >= 2", int(images["split_pores"].sum()), ""),
        ("detected pores holding >= 2 GT pores", int(images["merged_pores"].sum()), "merged"),
        ("false objects still counted", int(images["false_objects"].sum()), "less than 50 % of their pixels GT pore"),
        ("false objects, median area [px]", det.loc[~det["real"], "area_px"].median() if (~det["real"]).any() else None,
         "candidates for a filter"),
        ("area error matched pores, median [%]", 100 * pairs["area_error"].median() if len(pairs) else None,
         "detected / GT area - 1: + = outlined too large"),
        ("area error matched pores, mean |abs| [%]", 100 * pairs["area_error"].abs().mean() if len(pairs) else None, ""),
    ]
    return pd.DataFrame(rows, columns=["key figure", "value", "meaning"])


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as excel:
        for name, table in sheets.items():
            table.to_excel(excel, sheet_name=name, index=False)
            sheet = excel.sheets[name]
            for i, column in enumerate(table.columns):
                width = max([len(str(column))] + [len(str(v)) for v in table[column].head(200)])
                sheet.set_column(i, i, min(max(10, width + 2), 60))
            sheet.freeze_panes(1, 0)


# ---------------------------------------------------------------- plots

def _figure(plt, axes: int = 1):
    fig, ax = plt.subplots(1, axes, figsize=(SIZE_IN[0] * (1 if axes == 1 else 1.6), SIZE_IN[1]), dpi=DPI)
    return fig, ax


def _save(fig, path: Path, fmt: str = "png", **options) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, **(options or {"bbox_inches": "tight", "pad_inches": 0.04}))
    path.write_bytes(buffer.getvalue())     # also for non-ASCII paths


def write_plots(folder: Path, images: pd.DataFrame, gt: pd.DataFrame, det: pd.DataFrame, pairs: pd.DataFrame,
                            sizes: pd.DataFrame, filters: pd.DataFrame, errors: pd.DataFrame, unit: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    plots = [("1_porosity_gt_vs_detected", _porosity_scatter, (images,)),
             ("2_porosity_error", _porosity_error, (images,)),
             ("3_f1_per_image", _f1_histogram, (images,)),
             ("4_recall_precision_over_size", _recall_precision, (sizes,)),
             ("5_size_distribution", _size_distribution, (gt, det, unit)),
             ("6_area_error_over_size", _area_error, (pairs, sizes, unit)),
             ("7_filter_balance", _filter_balance, (filters,)),
             ("8_error_origin", _error_origin, (errors,))]
    with plt.rc_context(STYLE):
        for name, draw, args in plots:
            fig = draw(plt, *args)
            _save(fig, folder / f"{name}.png")
            plt.close(fig)


def _no_data(ax, text: str = "no data"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#495057")
    ax.set_xticks([])
    ax.set_yticks([])


def _porosity_scatter(plt, images):
    fig, ax = _figure(plt)
    top = max(images["gt_porosity_percent"].max(), images["det_porosity_percent"].max(), 1e-3) * 1.05
    ax.plot([0, top], [0, top], color="#868e96", lw=1, ls="--", label="1:1")
    flagged = images["roi_warning"]
    ax.scatter(images.loc[~flagged, "gt_porosity_percent"], images.loc[~flagged, "det_porosity_percent"],
               s=18, color=COLORS[0], edgecolor="white", lw=0.5, label="image", zorder=3)
    if flagged.any():
        ax.scatter(images.loc[flagged, "gt_porosity_percent"], images.loc[flagged, "det_porosity_percent"],
                   s=18, color=COLORS[1], edgecolor="white", lw=0.5, marker="s", label="flagged (ROI)", zorder=3)
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="porosity GT [%]", ylabel="porosity detected [%]",
           title="Porosity per image")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")
    return fig


def _porosity_error(plt, images):
    fig, ax = _figure(plt)
    err = images["porosity_error_pp"]
    ax.scatter(images["gt_porosity_percent"], err, s=18, color=COLORS[0], edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    mean, sd = err.mean(), err.std()
    ax.axhline(mean, color=COLORS[1], lw=1.2, label=f"mean {mean:+.3f} pp")
    if len(err) > 2 and sd == sd:
        for limit in (mean - 1.96 * sd, mean + 1.96 * sd):
            ax.axhline(limit, color=COLORS[1], lw=1, ls="--")
        ax.plot([], [], color=COLORS[1], lw=1, ls="--", label=f"±1.96 SD ({1.96 * sd:.3f} pp)")
    ax.set(xlabel="porosity GT [%]", ylabel="detected - GT [pp]", title="Porosity error (Bland-Altman)")
    ax.legend(loc="best")
    return fig


def _f1_histogram(plt, images):
    fig, ax = _figure(plt)
    values = images["pixel_f1"].dropna()
    if not len(values):
        _no_data(ax)
        return fig
    ax.hist(values, bins=np.linspace(0, 1, 21), color=COLORS[0], edgecolor="white", lw=1)
    ax.axvline(values.median(), color=COLORS[1], lw=1.2, label=f"median {values.median():.3f}")
    ax.set(xlim=(0, 1), xlabel="pixel F1 / Dice", ylabel="images", title="Pixel F1 per image")
    ax.legend(loc="upper left")
    return fig


def _recall_precision(plt, sizes):
    fig, ax = _figure(plt)
    x = np.arange(len(sizes))
    ax.plot(x, sizes["recall"], marker="o", ms=5, lw=2, color=COLORS[0], label="recall (GT pores found)")
    ax.plot(x, sizes["precision"], marker="s", ms=5, lw=2, color=COLORS[1], label="precision (detected correct)")
    ax.set_xticks(x, [f"{c}\nn={g}/{d}" for c, g, d in zip(sizes["size_class"], sizes["gt_pores"], sizes["det_pores"])])
    ax.minorticks_off()
    ax.set(ylim=(0, 1.05), xlabel="equivalent diameter", ylabel="share",
           title="Recall and precision over pore size (n = GT / detected)")
    ax.legend(loc="lower right")
    return fig


def _size_distribution(plt, gt, det, unit):
    fig, ax = _figure(plt)
    values = pd.concat([gt["diameter"], det["diameter"]])
    values = values[values > 0]
    if not len(values):
        _no_data(ax)
        return fig
    bins = np.geomspace(values.min(), values.max() * 1.0001, 31) if values.max() > values.min() else 10
    ax.hist(gt["diameter"], bins=bins, histtype="step", lw=2, color=COLORS[0], label=f"GT ({len(gt)})")
    ax.hist(det["diameter"], bins=bins, histtype="step", lw=2, color=COLORS[1], label=f"detected ({len(det)})")
    ax.set_xscale("log")
    ax.set(xlabel=f"equivalent diameter [{unit}]", ylabel="pores", title="Pore size distribution")
    ax.legend(loc="upper right")
    return fig


def _area_error(plt, pairs, sizes, unit):
    fig, ax = _figure(plt)
    if not len(pairs):
        _no_data(ax, "no matched pores")
        return fig
    ax.scatter(pairs["diameter"], 100 * pairs["area_error"], s=6, color=COLORS[0], alpha=0.35, lw=0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set(xlabel=f"equivalent diameter GT [{unit}]", ylabel="area detected / GT - 1 [%]",
           title="Area error of matched pores")
    low, high = np.percentile(100 * pairs["area_error"], [1, 99])
    ax.set_ylim(min(low, -10) * 1.1, max(high, 10) * 1.1)
    groups = pd.qcut(np.log(pairs["diameter"]), min(10, len(pairs)), duplicates="drop")
    median = pairs.groupby(groups, observed=True).agg(d=("diameter", "median"), e=("area_error", "median"))
    ax.plot(median["d"], 100 * median["e"], color=COLORS[1], lw=2, marker="o", ms=4, label="median (10 groups)")
    ax.legend(loc="upper right")
    return fig


def _filter_balance(plt, filters):
    fig, ax = _figure(plt)
    if not len(filters):
        _no_data(ax, "no filter rejected anything")
        return fig
    y = np.arange(len(filters))
    h = 0.38
    ax.barh(y - h / 2, filters["false_objects_right"], h, color=COLORS[0], label="false objects (right)")
    ax.barh(y + h / 2, filters["real_pores_wrong"], h, color=COLORS[1], label="real pores (wrong)")
    ax.set_yticks(y, filters["filter"])
    ax.invert_yaxis()
    ax.set(xlabel="rejected objects", title="Filter balance")
    ax.legend(loc="lower right")
    return fig


def _error_origin(plt, errors):
    fig, axes = _figure(plt, 2)
    for ax, kind, color, title in ((axes[0], "falsely detected", CLASS_COLORS[3][0], "Falsely detected pixels"),
                                   (axes[1], "missed", CLASS_COLORS[4][0], "Missed pixels")):
        part = errors[errors["kind"] == kind]
        y = np.arange(len(part))
        ax.barh(y, part["percent_of_kind"].fillna(0), color=color)
        for i, (pct, px) in enumerate(zip(part["percent_of_kind"].fillna(0), part["px"])):
            inside = pct > 75                   # long bar: label inside its end, else right of it
            ax.annotate(f"{px:,} px", (pct, i), xytext=(-4 if inside else 4, 0), textcoords="offset points",
                        va="center", ha="right" if inside else "left", fontsize=8,
                        color="white" if inside else "black")
        ax.set_yticks(y, part["where"].str.replace("_", " "))
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), xlabel="share [%]", title=title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- overlays

def worst_images(images: pd.DataFrame, n: int) -> list[tuple[str, str]]:
    """(image, reason) of the n lowest F1 and the n largest |porosity error|, each image once."""
    by_f1 = images.dropna(subset=["pixel_f1"]).nsmallest(n, "pixel_f1")["image"]
    by_porosity = images.assign(e=images["porosity_error_pp"].abs()).nlargest(n, "e")["image"]
    chosen: dict[str, str] = {}
    for name in by_f1:
        chosen[name] = "F1"
    for name in by_porosity:
        chosen[name] = "F1 + porosity" if name in chosen else "porosity"
    return list(chosen.items())


def write_overlay(path: Path, original: np.ndarray, classes: np.ndarray, title: str, max_px: int) -> None:
    """Left the original, right the classes on white, same area; JPG."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if original.dtype == np.uint16:
        original = (original // 257).astype(np.uint8)
    h, w = classes.shape
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        original = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
        classes = cv2.resize(classes, size, interpolation=cv2.INTER_NEAREST)
    elif w < OVERLAY_MIN_PANEL_PX:
        up = math.ceil(OVERLAY_MIN_PANEL_PX / w)
        original = np.repeat(np.repeat(original, up, 0), up, 1)
        classes = np.repeat(np.repeat(classes, up, 0), up, 1)
    h, w = classes.shape
    palette = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c, _ in CLASS_COLORS.values()], np.uint8)
    picture = palette[classes]

    dpi = 100
    width, height = 2 * w + OVERLAY_GAP_PX, h + OVERLAY_HEADER_PX
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    for i, image in enumerate((original, picture)):
        ax = fig.add_axes((i * (w + OVERLAY_GAP_PX) / width, 0, w / width, h / height))
        ax.imshow(image, interpolation="nearest")
        ax.set_axis_off()
    fig.text(4 / width, 1 - 6 / height, title, ha="left", va="top", fontsize=10, family="monospace")
    handles = [Patch(facecolor=c, edgecolor="#868e96", lw=0.5, label=label) for c, label in CLASS_COLORS.values()]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(1 - 4 / width, 1 - 2 / height))
    _save(fig, path, "jpg", dpi=dpi, pil_kwargs={"quality": 90})
    plt.close(fig)
