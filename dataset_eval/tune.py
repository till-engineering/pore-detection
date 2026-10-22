"""Parameter search: ONE setting of the pipeline for all images, scored against the GT masks.

    python dataset_eval/tune.py                       settings: tune: in dataset_eval/eval_config.yaml
    python dataset_eval/tune.py --config other.yaml

Searches the values given under tune.space (paths into the pipeline config, e.g.
detection.seed_threshold or postprocessing.label_steps.0.min_contrast). If the full grid has at
most tune.trials combinations, all are tried, else tune.trials random ones.

Images are split into a search part and a test part (tune.test_share): the best setting is chosen
on the search part only and then scored on the test part – if it is much worse there, the setting
only fits the searched images.

Also reported: "best per image" – for every image the best of all tried settings. Its gap to the
one global setting shows how much a single setting costs: small gap = one setting is enough for
this method; large gap = the images need different settings, the method does not adapt enough.

Speed: everything up to the end of preprocessing is computed once per image and reused, as long
as only detection / postprocessing parameters are searched (keys under sample / preprocessing /
input recompute it per trial). Steps run on the whole image (no tiles) – same result.
Like tools/find_settings.py, this uses core internals; nothing in pore_detection is changed.

Written to <eval.output>/tune/: trials.xlsx (every trial, scores on both parts, per-image table),
best_pipeline.yaml (paste under pipeline: of eval_config.yaml or into config.yaml),
f1_global_vs_per_image.png.
"""

import argparse
import copy
import itertools
import random
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pore_detection import Config  # noqa: E402
from pore_detection.core import imaging, io, registry  # noqa: E402
from pore_detection.core.pipeline import _bounding_box  # noqa: E402
from pore_detection.core.types import ImageContext  # noqa: E402

import metrics  # noqa: E402
import report  # noqa: E402
from evaluate import DEFAULT_CONFIG, load_settings, pair_files  # noqa: E402

PREPARE_KEYS = ("input", "sample", "preprocessing")     # changing these invalidates the cached gray image
OBJECTIVES = {
    "f1_median": "median of the pixel F1 per image (maximised)",
    "f1_p10": "10 % quantile of the pixel F1 per image – the bad images count (maximised)",
    "f1_total": "pixel F1 of all images summed (maximised)",
    "porosity_mae": "mean |porosity error| per image in pp (minimised)",
}


# ---------------------------------------------------------------- config paths

def get_path(data, path: str):
    for key in path.split("."):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def set_path(data, path: str, value) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        data = data[int(key)] if isinstance(data, list) else data[key]
    last = keys[-1]
    if isinstance(data, list):
        data[int(last)] = value
    else:
        data[last] = value


def trials_of(space: dict, count: int, rng: random.Random) -> list[dict]:
    keys = list(space)
    sizes = [len(space[k]) for k in keys]
    total = int(np.prod(sizes)) if keys else 1
    if total <= count:
        combos = list(itertools.product(*(space[k] for k in keys)))
    else:
        indices = rng.sample(range(total), count)
        combos = [tuple(space[k][i // int(np.prod(sizes[j + 1:])) % sizes[j]] for j, k in enumerate(keys))
                  for i in indices]
    return [dict(zip(keys, combo)) for combo in combos]


# ---------------------------------------------------------------- pipeline (as core, without tiles and writing)

class Prepared:
    """One image up to the end of preprocessing, cropped to the sample, with its GT."""

    def __init__(self, pair: metrics.Pair, config: Config, threshold: int):
        info, original = io.load_image(pair.image, pair.image.parent.name)
        gt = metrics.read_mask(pair.mask, threshold)
        if gt.shape != original.shape[:2]:
            raise ValueError("mask and image differ in size")
        roi = registry.create("sample.roi", config.sample.roi).detect_roi(info, original)
        scale = registry.create("sample.scale", config.sample.scale).detect_scale(info, original)
        valid = np.ones(gt.shape, bool) if scale.scale_mask is None else ~scale.scale_mask
        roi &= valid
        gt &= valid
        top, bottom, left, right = _bounding_box(roi)
        self.ctx = ImageContext(replace(info, offset_px=(top, left)), original[top:bottom, left:right],
                                roi[top:bottom, left:right], scale.pixel_size_um, None)
        gray = imaging.to_gray(self.ctx.original, config.input.gray_weights)
        for entry in config.preprocessing:
            gray = registry.create("preprocessing", entry).process(self.ctx, gray)
        self.gray = gray
        self.gt = gt[top:bottom, left:right]
        self.gt_outside = int(np.count_nonzero(gt)) - int(np.count_nonzero(self.gt))   # cut away by the crop


def score_image(prepared: Prepared, config: Config) -> dict:
    """Detection + postprocessing with config on one prepared image; pixel scores and porosity."""
    ctx = prepared.ctx
    mask = registry.create("detection", config.detection).detect(ctx, prepared.gray)
    for entry in config.postprocessing.mask_steps:
        mask = registry.create("postprocessing.mask_steps", entry).process(ctx, mask)
    mask = mask & ctx.roi_mask
    pores = registry.create("postprocessing.labeling", config.postprocessing.labeling).label(ctx, mask)
    for entry in config.postprocessing.label_steps:
        pores = registry.create("postprocessing.label_steps", entry).process(ctx, pores)
    det, gt, roi = pores.labels > 0, prepared.gt, ctx.roi_mask
    tp = int(np.count_nonzero(det & gt))
    fp = int(np.count_nonzero(det)) - tp
    fn = int(np.count_nonzero(gt)) - tp + prepared.gt_outside
    roi_px = int(np.count_nonzero(roi))
    porosity_error = 100 * (np.count_nonzero(det) - np.count_nonzero(gt & roi)) / roi_px if roi_px else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
            "porosity_error_pp": porosity_error}


def objective(table: pd.DataFrame, name: str) -> float:
    """Score of one trial on a set of images; larger = better."""
    if name == "f1_median":
        return float(table["f1"].median())
    if name == "f1_p10":
        return float(table["f1"].quantile(0.1))
    if name == "f1_total":
        tp, fp, fn = table["tp"].sum(), table["fp"].sum(), table["fn"].sum()
        return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0
    return -float(table["porosity_error_pp"].abs().mean())


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="evaluation settings with tune:")
    args = parser.parse_args()

    config_path = args.config.resolve()
    settings, base_config = load_settings(config_path)
    tune = yaml.safe_load(config_path.read_text(encoding="utf-8"))["tune"]
    if tune["objective"] not in OBJECTIVES:
        sys.exit(f"tune.objective must be one of {', '.join(OBJECTIVES)}")
    rng = random.Random(tune["seed"])

    pairs, _ = pair_files(settings["images"], settings["masks"], base_config.input.extensions)
    rng.shuffle(pairs)
    if tune["images"]:
        pairs = pairs[:tune["images"]]
    n_test = round(len(pairs) * tune["test_share"])
    test_names = {p.name for p in pairs[:n_test]}
    space = {key: list(values) for key, values in tune["space"].items()}
    base = base_config.to_dict()
    for key in space:
        get_path(base, key)                         # unknown path -> error now, not after the first trial
    trials = trials_of(space, tune["trials"], rng)
    print(f"{len(pairs)} images ({len(pairs) - n_test} search, {n_test} test), {len(trials)} trials, "
          f"objective {tune['objective']}: {OBJECTIVES[tune['objective']]}")

    # trials grouped by their preparation part: the prepared images are reused within a group
    def prepare_key(trial: dict) -> tuple:
        return tuple(sorted((k, repr(v)) for k, v in trial.items() if k.split(".")[0] in PREPARE_KEYS))

    rows, per_image = [], []
    groups: dict[tuple, list[int]] = {}
    for i, trial in enumerate(trials):
        groups.setdefault(prepare_key(trial), []).append(i)
    bar = tqdm(total=len(trials) * len(pairs), unit="image", desc="tune")
    for indices in groups.values():
        data = copy.deepcopy(base)
        for key, value in trials[indices[0]].items():
            set_path(data, key, value)
        prepared = {}
        for pair in pairs:
            try:
                prepared[pair.name] = Prepared(pair, Config(data), settings["mask_threshold"])
            except Exception as error:      # noqa: BLE001 – one bad image must not stop the search
                tqdm.write(f"  {pair.image.name}: skipped ({type(error).__name__}: {error})")
        for i in indices:
            data = copy.deepcopy(base)
            for key, value in trials[i].items():
                set_path(data, key, value)
            config = Config(data)
            scores = []
            for name, image in prepared.items():
                try:
                    scores.append({"trial": i, "image": name, "part": "test" if name in test_names else "search",
                                   **score_image(image, config)})
                except Exception as error:  # noqa: BLE001
                    tqdm.write(f"  trial {i}, {name}: failed ({type(error).__name__}: {error})")
                bar.update()
            table = pd.DataFrame(scores)
            per_image += scores
            row = {"trial": i, **trials[i]}
            for part in ("search", "test"):
                sub = table[table["part"] == part] if len(table) else table
                row[f"{part}_score"] = objective(sub, tune["objective"]) if len(sub) else np.nan
                row[f"{part}_f1_median"] = sub["f1"].median() if len(sub) else np.nan
                row[f"{part}_porosity_mae_pp"] = sub["porosity_error_pp"].abs().mean() if len(sub) else np.nan
            rows.append(row)
    bar.close()

    trials_table = pd.DataFrame(rows).sort_values("search_score", ascending=False)
    images = pd.DataFrame(per_image)
    best = trials_table.iloc[0]
    best_trial = int(best["trial"])
    global_f1 = images[images["trial"] == best_trial].set_index("image")["f1"]
    oracle = images.loc[images.groupby("image")["f1"].idxmax()].set_index("image")
    compare = pd.DataFrame({"part": oracle["part"], "f1_global_best": global_f1, "f1_best_per_image": oracle["f1"],
                            "best_trial_per_image": oracle["trial"]})
    compare["gap"] = compare["f1_best_per_image"] - compare["f1_global_best"]

    out = settings["output"] / "tune"
    out.mkdir(parents=True, exist_ok=True)
    report.write_excel(out / "trials.xlsx", {"Trials": trials_table, "Global vs per image": compare.reset_index(),
                                             "Per image": images})
    data = copy.deepcopy(base)
    for key, value in trials[best_trial].items():       # the values as given (the table makes 80 -> 80.0)
        set_path(data, key, value)
    snippet = {key.split(".")[0]: data[key.split(".")[0]] for key in space}
    (out / "best_pipeline.yaml").write_text(
        f"# best of {len(trials)} trials, objective {tune['objective']}: search {best['search_score']:.4f}, "
        f"test {best['test_score']:.4f}\n" + yaml.safe_dump(snippet, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    _plot_gap(out / "f1_global_vs_per_image.png", compare)

    print(f"\nbest trial {best_trial}: " + ", ".join(f"{k} = {v}" for k, v in trials[best_trial].items()))
    print(f"  score search {best['search_score']:.4f}, test {best['test_score']:.4f} ({tune['objective']})")
    print(f"  F1 median: global setting {compare['f1_global_best'].median():.4f}, "
          f"best per image {compare['f1_best_per_image'].median():.4f} "
          f"(gap median {compare['gap'].median():.4f}, max {compare['gap'].max():.4f})")
    print(f"-> {out}")
    return 0


def _plot_gap(path: Path, compare: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(report.STYLE):
        fig, ax = report._figure(plt)
        for part, color, marker in (("search", report.COLORS[0], "o"), ("test", report.COLORS[1], "s")):
            sub = compare[compare["part"] == part]
            ax.scatter(sub["f1_best_per_image"], sub["f1_global_best"], s=18, color=color, marker=marker,
                       edgecolor="white", lw=0.5, label=f"{part} images", zorder=3)
        low = min(compare["f1_global_best"].min(), compare["f1_best_per_image"].min(), 0.9) - 0.02
        ax.plot([low, 1], [low, 1], color="#868e96", lw=1, ls="--", label="no gap")
        ax.set(xlim=(low, 1), ylim=(low, 1), xlabel="F1 best setting of each image",
               ylabel="F1 one setting for all", title="What one setting for all images costs")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left")
        report._save(fig, path)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
