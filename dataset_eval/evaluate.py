"""Evaluate the pore detection on a data set: micrographs + ground-truth masks (white = pore).

    python dataset_eval/evaluate.py              run the pipeline + evaluate
    python dataset_eval/evaluate.py --no-run     evaluate the last pipeline results again
    python dataset_eval/evaluate.py --config other.yaml

Settings: dataset_eval/eval_config.yaml – eval: (folders, thresholds) and pipeline: (overrides of
the project config.yaml). The pipeline is only called through its public API; nothing in
pore_detection is changed. Images and masks are paired by file name without extension.

Written to <eval.output>/:
    pipeline/<ID>_results/   output of the pipeline run (as by main.py)
    evaluation.xlsx          Images, Summary, Size classes, Filters, Error origin, Skipped
    plots/*.png              porosity, F1, recall / precision over size, sizes, area error, filters, errors
    overlays/*.jpg           worst images: original | classes on white (grey correct, orange false, blue missed)
    eval_config_used.yaml    copy of the evaluation settings
See metrics.py for how the comparison is made.
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pore_detection import Config, RunAborted, config_path, find_images, load_config, run  # noqa: E402
from pore_detection.core.io import load_image  # noqa: E402

import metrics  # noqa: E402
import report  # noqa: E402

DEFAULT_CONFIG = HERE / "eval_config.yaml"


# ---------------------------------------------------------------- settings

def merge(base, override):
    """override on top of base: dicts key by key, everything else replaced; a method block with
    another name replaces the whole block (its parameters belong to the old method)."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    if "name" in override and override["name"] != base.get("name"):
        return copy.deepcopy(override)
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(base[key], value) if key in base else copy.deepcopy(value)
    return result


def load_settings(path: Path) -> tuple[dict, Config]:
    """Evaluation settings (paths absolute) and the pipeline config."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = dict(data["eval"])
    for key in ("images", "masks", "output"):
        folder = Path(settings[key])
        settings[key] = (folder if folder.is_absolute() else path.parent / folder).resolve()
    pipeline = merge(load_config().to_dict(), data.get("pipeline") or {})
    print(f"pipeline config: {config_path().name} + overrides of {path.name}")
    return settings, Config(pipeline)


# ---------------------------------------------------------------- pairing

def pair_files(images_dir: Path, masks_dir: Path, extensions) -> tuple[list[metrics.Pair], pd.DataFrame]:
    """Image + mask with the same file name without extension; the rest goes to the skipped list."""
    image_id = images_dir.name
    skipped = []
    masks: dict[str, Path] = {}
    for path in find_images(masks_dir, extensions):
        if path.stem in masks:
            skipped.append({"file": path.name, "reason": f"second mask with the name {path.stem}"})
            continue
        masks[path.stem] = path
    pairs, seen = [], set()
    for path in find_images(images_dir, extensions):
        if path.stem in seen:
            skipped.append({"file": path.name, "reason": f"second image with the name {path.stem}"})
            continue
        seen.add(path.stem)
        if path.stem in masks:
            pairs.append(metrics.Pair(f"{image_id}_{path.stem}", path, masks[path.stem]))
        else:
            skipped.append({"file": path.name, "reason": "no mask"})
    skipped += [{"file": m.name, "reason": "no image"} for stem, m in masks.items() if stem not in seen]
    return pairs, pd.DataFrame(skipped, columns=["file", "reason"])


# ---------------------------------------------------------------- pipeline

def run_pipeline(config: Config, images_dir: Path, out_dir: Path) -> Path:
    bar = None

    def progress(index: int, total: int, stage: str, name: str):
        nonlocal bar
        if bar is None:
            bar = tqdm(total=total, unit="image", desc="pipeline")
        bar.n = index
        bar.set_postfix_str(f"{name} {stage}" if name else stage)
        bar.refresh()

    try:
        result = run(config, input_path=images_dir, output_dir=out_dir, progress=progress)
    except RunAborted as error:
        sys.exit(f"run aborted: {error}")
    finally:
        if bar is not None:
            bar.close()
    print(f"pipeline: {result.succeeded} images succeeded, {result.failed} failed, "
          f"{result.duration_s:.1f} s, device {result.device}")
    return result.results_dir


# ---------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="evaluation settings")
    parser.add_argument("--no-run", action="store_true", help="only evaluate existing pipeline results")
    args = parser.parse_args()

    config_path = args.config.resolve()
    settings, config = load_settings(config_path)
    images_dir, masks_dir, out = settings["images"], settings["masks"], settings["output"]
    for folder in (images_dir, masks_dir):
        if not folder.is_dir():
            sys.exit(f"folder not found: {folder}")
    pairs, skipped = pair_files(images_dir, masks_dir, config.input.extensions)
    print(f"{len(pairs)} image / mask pairs, {len(skipped)} files without partner")
    if not pairs:
        return 1

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "eval_config_used.yaml")
    results_dir = out / "pipeline" / f"{images_dir.name}_results"
    if not args.no_run:
        results_dir = run_pipeline(config, images_dir, out / "pipeline")
    if not (results_dir / "images.json").is_file():
        sys.exit(f"no pipeline results in {results_dir}")
    rows = {row["image"]: row for row in metrics.read_table(results_dir / "images.json").to_dict("records")}
    errors_path = results_dir / "errors.json"
    failed = metrics.read_table(errors_path) if errors_path.is_file() else pd.DataFrame(columns=["image", "message"])
    failed = dict(zip(failed["image"], failed["message"]))

    connectivity = int(config.postprocessing.labeling.to_dict().get("connectivity", 8))
    results: dict[str, metrics.ImageResult] = {}
    for pair in tqdm(pairs, unit="image", desc="evaluation"):
        folder = results_dir / pair.name
        if pair.name not in rows or not (folder / "labels.tif").is_file():
            reason = failed.get(pair.name, "no pipeline result")
            skipped.loc[len(skipped)] = [pair.image.name, f"pipeline failed: {reason}"]
            continue
        try:
            data = metrics.load(pair, folder, rows[pair.name], settings["mask_threshold"])
        except ValueError as error:
            skipped.loc[len(skipped)] = [pair.image.name, str(error)]
            continue
        result = metrics.evaluate(data, settings["match_iou"], settings["rim_px"], connectivity)
        result.values = {"image": pair.name, "file": pair.image.name,
                         "pixel_size_um": rows[pair.name].get("pixel_size_um"), **result.values}
        results[pair.name] = result
    if not results:
        print("nothing to evaluate")
        return 1

    images = pd.DataFrame([r.values for r in results.values()])
    images["pixel_size_um"] = pd.to_numeric(images["pixel_size_um"])
    images.insert(3, "roi_warning", images["gt_outside_roi_percent"] > settings["roi_warn_percent"])
    unit, factor = report.size_unit(images)
    gt, det, pairs_table = report.pore_tables(results, dict(zip(images["image"], factor)))
    edges = settings["size_classes_um"] if unit == "um" else settings["size_classes_px"]
    sizes = report.size_class_table(gt, det, pairs_table, edges, unit)
    filters = report.filter_table(results, int(images["gt_pore_px"].sum()))
    errors = report.error_table(images)
    summary = report.summary(images, det, pairs_table, settings, unit, {"skipped": len(skipped)})

    report.write_excel(out / "evaluation.xlsx", {"Summary": summary, "Images": images, "Size classes": sizes,
                                                 "Filters": filters, "Error origin": errors, "Skipped": skipped})
    report.write_plots(out / "plots", images, gt, det, pairs_table, sizes, filters, errors, unit)

    overlays = out / "overlays"
    if overlays.is_dir():
        for old in overlays.glob("*.jpg"):
            old.unlink()
    overlays.mkdir(exist_ok=True)
    by_name = {pair.name: pair for pair in pairs}
    table = images.set_index("image")
    for rank, (name, reason) in enumerate(report.worst_images(images, settings["worst_images"]), 1):
        pair, v = by_name[name], table.loc[name]
        data = metrics.load(pair, results_dir / name, rows[name], settings["mask_threshold"])
        _, original = load_image(pair.image, images_dir.name)
        title = (f"{pair.image.name}   worst by {reason}\n"
                 f"F1 {v['pixel_f1']:.3f}   porosity GT {v['gt_porosity_percent']:.3f} %  "
                 f"detected {v['det_porosity_percent']:.3f} %  ({v['porosity_error_pp']:+.3f} pp)")
        report.write_overlay(overlays / f"{rank:02d}_{pair.image.stem}.jpg", original, metrics.classes(data),
                             title, settings["overlay_max_px"])

    print("\n" + summary.to_string(index=False, max_colwidth=60))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
