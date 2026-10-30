"""Run: process the images of one ID folder one by one through all stages.

Per image: load -> sample (ROI, scale, sample frame) -> crop to the bounding box of the sample -> gray -> filters /
detector / mask steps (tileable steps on tiles in parallel threads) -> labeling -> label steps
-> evaluation modules (in parallel) -> write (plots are drawn while writing).

While an image is computed, the next one is loaded in the background (runtime.prefetch) and
the previous one is written in the background. On device "cuda", tileable steps with a GPU
path run on the GPU (see core/tiling.py).
"""

import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Callable, Iterator

import cv2
import numpy as np
import pandas as pd

from . import io, plotting, registry, tiling, writer
from .config import PROJECT_DIR, Config
from .device import gpu_name, select_device
from .imaging import to_gray
from .step import Step
from .types import EvaluationResult, FrameMissing, ImageContext, ImageInfo, SampleFrame, ScaleResult

log = logging.getLogger("pore_detection")

# progress(image index 0-based, image count, stage, image name)
Progress = Callable[[int, int, str, str], None]


class RunAborted(Exception):
    """Error that affects all images: the run stops."""


@dataclass(frozen=True)
class ImageError:
    image: str
    stage: str
    method: str
    message: str


@dataclass(frozen=True)
class RunResult:
    succeeded: int
    failed: int
    errors: list[ImageError]
    results_dir: Path
    duration_s: float
    cancelled: bool
    device: str                 # "cuda" or "cpu"


@dataclass
class _Steps:
    roi: Step
    scale: Step
    frame: Step
    filters: list[Step]         # preprocessing chain (may run on tiles)
    detection: list[Step]       # detector + mask steps (may run on tiles)
    labeler: Step
    label_steps: list[Step]
    evaluators: list[Step]
    plots: list[Step]

    @classmethod
    def from_config(cls, config: Config) -> "_Steps":
        post = config.postprocessing
        return cls(
            roi=registry.create("sample.roi", config.sample.roi),
            scale=registry.create("sample.scale", config.sample.scale),
            frame=registry.create("sample.frame", config.sample.frame),
            filters=[registry.create("preprocessing", e) for e in config.preprocessing],
            detection=[registry.create("detection", config.detection)]
                      + [registry.create("postprocessing.mask_steps", e) for e in post.mask_steps],
            labeler=registry.create("postprocessing.labeling", post.labeling),
            label_steps=[registry.create("postprocessing.label_steps", e) for e in post.label_steps],
            evaluators=[registry.create("evaluation", e) for e in config.evaluation],
            plots=[registry.create("plots", e) for e in config.plots],
        )


class _ImageFailed(Exception):
    def __init__(self, stage: str, method: str, cause: Exception):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage, self.method = stage, method


@dataclass
class _Done:
    """An image that passed all stages; its files are written in the background."""

    name: str
    row: dict
    write: Callable[[], None]
    layout: dict[str, str | None]   # its per-image columns -> sheet of the results Excel


def run(config: Config, input_path=None, output_dir=None,
        progress: Progress | None = None, cancel: Event | None = None) -> RunResult:
    """Process all images of one ID folder. Paths: None = value from config.yaml."""
    start = time.perf_counter()
    setup_logging(config)
    input_dir = _resolve(input_path if input_path is not None else config.input.path)
    output_dir = _resolve(output_dir if output_dir is not None else config.output.dir)
    if not input_dir.is_dir():
        raise RunAborted(f"input folder not found: {input_dir}")

    image_id = io.image_id_of(input_dir)
    steps = _Steps.from_config(config)
    if steps.plots and not plotting.available():
        raise RunAborted("plots are configured but matplotlib is not installed "
                         "(pip install -r requirements.txt, or remove the plots from config.yaml)")
    gpu_wanted = any(step.tileable and tiling.gpu_path(step) for step in steps.filters + steps.detection)
    device = select_device(config.runtime.device, gpu_wanted)
    gpu = tiling.Gpu(config.runtime.gpu_memory_fraction, config.runtime.tile_batch_size) if device == "cuda" else None
    views = config.input.views
    files, ignored = io.find_recordings(input_dir, config.input.extensions, views)
    threads = os.cpu_count() if config.runtime.threads == "auto" else config.runtime.threads
    log.info("ID %s: %d images in %s, device %s, %d threads", image_id, len(files), input_dir,
             gpu_name() if gpu else device, threads)
    if ignored:
        log.warning("%d files are neither view %s nor %s and are ignored: %s", len(ignored), views.sample,
                    views.pores, ", ".join(path.name for path in ignored))

    try:
        results_dir = writer.prepare_results_dir(output_dir, image_id)
        writer.write_config(results_dir, config.to_dict())
    except OSError as error:
        raise RunAborted(f"output folder not writable: {error}") from error

    rows: list[dict] = []
    layout: dict[str, str | None] = dict(CORE_EXCEL)     # column -> sheet of the results Excel
    errors: list[ImageError] = []
    cancelled = False
    id_values = None
    pool = ThreadPoolExecutor(threads, thread_name_prefix="tile")
    loader = ThreadPoolExecutor(1, thread_name_prefix="load")
    background_writer = ThreadPoolExecutor(1, thread_name_prefix="write")
    writing: tuple[_Done, Future] | None = None

    def fail(name: str, stage: str, method: str, message: str):
        errors.append(ImageError(name, stage, method, message))
        log.error("%s skipped – %s / %s: %s", name, stage, method, message)

    def finish_writing():
        """Wait for the image being written; only then it counts as succeeded."""
        nonlocal writing
        if writing is None:
            return
        done, future = writing
        writing = None
        try:
            future.result()
            rows.append(done.row)
        except Exception as error:
            fail(done.name, "write", "-", f"{type(error).__name__}: {error}")

    def load(recording: io.Recording) -> Callable[[], tuple[ImageInfo, np.ndarray, np.ndarray]]:
        if config.runtime.prefetch:
            return loader.submit(io.load_recording, recording, image_id, views).result
        return lambda: io.load_recording(recording, image_id, views)

    try:
        next_load = load(files[0]) if files else None
        for index, recording in enumerate(files):
            if cancel is not None and cancel.is_set():
                cancelled = True
                log.info("run cancelled after %d of %d images", index, len(files))
                break
            current_load = next_load
            next_load = load(files[index + 1]) if index + 1 < len(files) else None
            name = f"{image_id}_{recording.name}"
            report = (lambda stage: progress(index, len(files), stage, name)) if progress else (lambda stage: None)
            try:
                done = _process_image(config, steps, current_load, recording, results_dir, pool, gpu, report)
            except RunAborted:
                raise
            except _ImageFailed as failure:
                fail(name, failure.stage, failure.method, str(failure))
                continue
            finish_writing()
            layout.update(done.layout)
            writing = (done, background_writer.submit(done.write))
        finish_writing()
        if not cancelled and progress:
            progress(len(files), len(files), "done", "")
        if not cancelled and rows:
            id_values = _finalize(steps.evaluators, pd.DataFrame(rows), results_dir)
            write_id_plots(steps.plots, pd.DataFrame(rows), results_dir)
    finally:
        # also after an abort: everything finished so far stays readable
        finish_writing()
        loader.shutdown(wait=False, cancel_futures=True)
        background_writer.shutdown()
        pool.shutdown()
        error_table = pd.DataFrame([vars(e) for e in errors], columns=["image", "stage", "method", "message"])
        writer.write_summary(results_dir, image_id, pd.DataFrame(rows), error_table, id_values, layout)

    duration = time.perf_counter() - start
    log.info("%d of %d images succeeded, %d failed (%.1f s)", len(rows), len(files), len(errors), duration)
    return RunResult(len(rows), len(errors), errors, results_dir, duration, cancelled, device)


def _process_image(config: Config, steps: _Steps, load: Callable[[], tuple[ImageInfo, np.ndarray, np.ndarray]],
                   recording: io.Recording, results_dir: Path, pool: ThreadPoolExecutor, gpu: tiling.Gpu | None,
                   report: Callable[[str], None]) -> _Done:
    """One recording through all stages; returns its row of the per-image table and its write job.
    The sample stage runs on the sample view, everything after it on the pores view (the same image
    without input.views)."""
    stage, method = "load", "-"

    def at(new_stage: str, step: Step | None = None):
        nonlocal stage, method
        stage, method = new_stage, step.name if step else "-"
        report(stage)

    try:
        at("load")
        info, original, sample_view = load()
        drawing = sample_view if config.output.draw_on == "sample" else original     # overlays, frame picture

        # both sample methods only read the sample view: the scale runs in parallel with the ROI
        scale_future = pool.submit(steps.scale.detect_scale, info, sample_view)
        at("sample.roi", steps.roi)
        roi_mask = steps.roi.detect_roi(info, sample_view)
        del sample_view
        at("sample.scale", steps.scale)
        scale = scale_future.result()
        if scale.scale_mask is not None:
            if scale.scale_mask.shape != roi_mask.shape:
                raise ValueError(f"scale_mask has shape {scale.scale_mask.shape}, image {roi_mask.shape}")
            roi_mask = roi_mask & ~scale.scale_mask     # the scale overlay is never sample
        top, bottom, left, right = _bounding_box(roi_mask)
        at("sample.frame", steps.frame)
        frame = steps.frame.detect_frame(info, roi_mask)

        at("crop")
        ctx = ImageContext(replace(info, offset_px=(top, left)), original[top:bottom, left:right],
                           roi_mask[top:bottom, left:right], scale.pixel_size_um, frame)

        at("gray")
        gray = _to_gray_parallel(ctx.original, config.input.gray_weights, config.runtime.tile_size, pool)
        gray = _run_chain(steps.filters, ctx, gray, config, pool, gpu, at)
        mask = _run_chain(steps.detection, ctx, gray, config, pool, gpu, at)
        del gray
        mask = mask & ctx.roi_mask              # never a pore in the embedding medium

        at("postprocessing.labeling", steps.labeler)
        _check_scale(steps.labeler, ctx)
        pores = steps.labeler.label(ctx, mask)
        del mask
        for step in steps.label_steps:
            at("postprocessing.label_steps", step)
            _check_scale(step, ctx)
            pores = step.process(ctx, pores)

        at("evaluation")
        evaluators = [step.for_image(recording.key) for step in steps.evaluators]     # per-image inputs
        for step in evaluators:
            _check_scale(step, ctx)
        futures = [(step, pool.submit(step.evaluate, ctx, pores)) for step in evaluators]
        results: dict[str, EvaluationResult] = {}
        for step, future in futures:
            at("evaluation", step)
            try:
                results[step.name] = future.result()
            except FrameMissing:            # e.g. a cut-out without sample edges: only this module is left out
                log.warning("%s: evaluation %s skipped – no sample frame", info.name, step.name)
        at("evaluation")
        per_image, per_pore, images = _merge_results(results)

        full_labels = _to_full(pores.labels, info, top, left)
        full_rejected = _to_full(pores.rejected_labels, info, top, left)
    except RunAborted:
        raise
    except tiling.StepError as error:
        raise _ImageFailed(registry.stage_of(type(error.step)), error.step.name, error.__cause__) from error
    except Exception as error:
        raise _ImageFailed(stage, method, error) from error

    # values of the modules first: the order of the columns in images.json and the Excel sheets
    draw_path = str(recording.sample if config.output.draw_on == "sample" else recording.pores)
    row = {"image": info.name, "path": str(info.path), "draw_path": draw_path, **per_image, **_scale_columns(scale),
           **frame_columns(frame, scale.pixel_size_um), "manual_corrections": 0}

    def write():
        writer.write_image(results_dir, info, drawing, full_labels, full_rejected, pores.rejected, roi_mask,
                           per_pore, images, config.output.save_full_overlay)
        picture = lambda: plotting.sample_picture(drawing, full_labels, frame)      # noqa: E731 – only if needed
        writer.write_plots(results_dir / info.name, plotting.render(steps.plots, row, per_pore, picture))
    return _Done(info.name, row, write, excel_layout(steps.evaluators, results))


def _to_full(labels: np.ndarray | None, info: ImageInfo, top: int, left: int) -> np.ndarray:
    """Labels of the cropped area placed into the full image size (zeros outside / if None)."""
    full = np.zeros((info.height_px, info.width_px), dtype=np.int32)
    if labels is not None:
        full[top:top + labels.shape[0], left:left + labels.shape[1]] = labels
    return full


def _run_chain(chain: list[Step], ctx: ImageContext, data: np.ndarray, config: Config,
               pool: ThreadPoolExecutor, gpu: tiling.Gpu | None, at) -> np.ndarray:
    """Filters or detection: tileable runs on tiles, other steps on the whole image."""
    for group in tiling.groups(chain):
        for step in group:
            _check_scale(step, ctx)
        at(registry.stage_of(type(group[0])), group[0])
        if group[0].tileable:
            data = tiling.run_tiled(group, ctx, data, config.runtime.tile_size, pool, gpu)
        else:
            data = tiling.apply(group[0], ctx, data)
    return data


def _to_gray_parallel(original: np.ndarray, weights, band_px: int, pool: ThreadPoolExecutor) -> np.ndarray:
    """to_gray in horizontal bands in parallel threads (same result as one call)."""
    gray = np.empty(original.shape[:2], dtype=np.float32)
    bands = [(y, min(y + band_px, len(gray))) for y in range(0, len(gray), band_px)]
    for future in [pool.submit(to_gray, original[a:b], weights, gray[a:b]) for a, b in bands]:
        future.result()
    return gray


def _bounding_box(roi_mask: np.ndarray) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) of the sample; raises if the mask is empty."""
    left, top, width, height = cv2.boundingRect(roi_mask.view(np.uint8))
    if width == 0 or height == 0:
        raise ValueError("no sample found (roi_mask is empty)")
    return top, top + height, left, left + width


def _scale_columns(scale: ScaleResult) -> dict:
    """Scale columns of images.json: value, what was read, where (boxes as [top, bottom, left, right])."""
    box = None
    if scale.scale_mask is not None and scale.scale_mask.any():
        left, top, width, height = cv2.boundingRect(scale.scale_mask.view(np.uint8))
        box = [top, top + height, left, left + width]
    return {
        "pixel_size_um": scale.pixel_size_um,
        "scale_label": scale.label,
        "scale_bar_px": list(scale.bar_px) if scale.bar_px is not None else None,
        "scale_box_px": box,
    }


# sheet of the results Excel for the columns of core (None = JSON only); see excel_layout
CORE_EXCEL: dict[str, str | None] = {
    "image": writer.PORES_SHEET, "path": None, "draw_path": None,
    "pixel_size_um": writer.DETECTION_SHEET, "scale_label": writer.DETECTION_SHEET,
    "scale_bar_px": None, "scale_box_px": None,
    "frame_origin_px": None, "frame_angle_deg": writer.DETECTION_SHEET, "frame_wedge_deg": writer.DETECTION_SHEET,
    "frame_thickness_px": writer.PORES_SHEET, "frame_thickness_um": writer.PORES_SHEET,
    "frame_length_px": writer.PORES_SHEET, "frame_length_um": writer.PORES_SHEET,
    "manual_corrections": writer.PORES_SHEET,
}


def excel_layout(evaluators: list[Step], results: dict[str, EvaluationResult]) -> dict[str, str | None]:
    """Sheet of the results Excel for every per-image column: core columns as in CORE_EXCEL, the
    values of an evaluation module on its excel_sheet (default: its name), excel_hidden ones in
    JSON only."""
    layout = dict(CORE_EXCEL)
    for step in evaluators:
        result = results.get(step.name)
        if result is None:
            continue
        sheet = step.excel_sheet or step.name
        for key in result.per_image:
            layout[key] = None if key in step.excel_hidden else sheet
    return layout


FRAME_COLUMNS = ("frame_origin_px", "frame_angle_deg", "frame_thickness_px", "frame_thickness_um",
                 "frame_length_px", "frame_length_um", "frame_wedge_deg")


def frame_columns(frame: SampleFrame | None, pixel_size_um: float | None) -> dict:
    """Sample frame columns of images.json (all None without a frame; _um only with a scale)."""
    if frame is None:
        return dict.fromkeys(FRAME_COLUMNS)
    um = (lambda px: px * pixel_size_um) if pixel_size_um is not None else (lambda px: None)
    return {
        "frame_origin_px": [frame.origin_px[0], frame.origin_px[1]],
        "frame_angle_deg": frame.angle_deg,
        "frame_thickness_px": frame.thickness_px,
        "frame_thickness_um": um(frame.thickness_px),
        "frame_length_px": frame.length_px,
        "frame_length_um": um(frame.length_px),
        "frame_wedge_deg": frame.wedge_deg,
    }


def frame_from_columns(row: dict) -> SampleFrame | None:
    """The sample frame back from a row of images.json (e.g. for manual corrections)."""
    origin = row.get("frame_origin_px")
    if origin is None or (isinstance(origin, float) and np.isnan(origin)):
        return None
    return SampleFrame((float(origin[0]), float(origin[1])), float(row["frame_angle_deg"]),
                       float(row["frame_thickness_px"]), float(row["frame_length_px"]),
                       float(row["frame_wedge_deg"]))


def _check_scale(step: Step, ctx: ImageContext) -> None:
    """A step with a micrometre parameter needs a known scale."""
    if ctx.pixel_size_um is None and any(key.endswith(("_um", "_um2")) for key in vars(step)):
        raise tiling.StepError(step) from ValueError("scale required but not found")


def _merge_results(results: dict[str, EvaluationResult]):
    """Combine the results of all evaluation modules; duplicate names abort the run."""
    per_image: dict = {}
    images: dict = {}
    per_pore: pd.DataFrame | None = None
    owner: dict[str, str] = {}

    def claim(key: str, module: str, kind: str):
        if key in owner:
            raise RunAborted(f"evaluation modules '{owner[key]}' and '{module}' both return {kind} '{key}'")
        owner[key] = module

    for module, result in results.items():
        for key in result.per_image:
            claim(key, module, "per-image column")
        per_image.update(result.per_image)
        for key in result.images:
            claim(f"image:{key}", module, "image")
        images.update(result.images)
        if result.per_pore is not None:
            for key in result.per_pore.columns:
                if key != "pore_id":
                    claim(f"pore:{key}", module, "per-pore column")
            per_pore = result.per_pore if per_pore is None else per_pore.merge(result.per_pore, on="pore_id")

    for key, value in per_image.items():
        if isinstance(value, np.generic):
            per_image[key] = value.item()
    if per_pore is None:
        per_pore = pd.DataFrame({"pore_id": pd.Series(dtype=np.int32)})
    return per_image, per_pore, images


def _finalize(evaluators: list[Step], per_image: pd.DataFrame, results_dir: Path) -> dict:
    def per_pore_tables() -> Iterator[pd.DataFrame]:
        for name in per_image["image"]:
            yield writer.read_table(results_dir / name / writer.PORES_JSON)

    values: dict = {}
    owner: dict[str, str] = {}
    for step in evaluators:
        if not hasattr(step, "finalize"):
            continue
        for key, value in step.finalize(per_image, per_pore_tables()).items():
            if key in owner:
                raise RunAborted(f"evaluation modules '{owner[key]}' and '{step.name}' both return ID value '{key}'")
            owner[key] = step.name
            values[key] = value
    return values


def write_id_plots(plots: list[Step], per_image: pd.DataFrame, results_dir: Path) -> None:
    """Plots over the whole ID (plot_id of the plot modules) -> <ID>_results/plots/."""
    if any(hasattr(step, "plot_id") for step in plots):
        pores = plotting.id_pores(per_image, writer.read_table, results_dir)
        writer.write_plots(results_dir, plotting.render_id(plots, per_image, pores))


def resolve_path(path) -> Path:
    """Relative paths in config.yaml are relative to the project folder."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


_resolve = resolve_path

LOG_FORMAT = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")


def setup_logging(config: Config) -> None:
    """Console (if there is one) and optional log file. Handlers added by others (e.g. the
    launcher's log window) are kept."""
    log.setLevel(config.logging.level)
    for handler in list(log.handlers):
        if getattr(handler, "_pore_detection", False):
            log.removeHandler(handler)
            handler.close()
    handlers = [logging.StreamHandler()] if sys.stderr is not None else []  # pythonw: no console
    if config.logging.file:
        handlers.append(logging.FileHandler(_resolve(config.logging.file), encoding="utf-8"))
    for handler in handlers:
        handler._pore_detection = True
        handler.setFormatter(LOG_FORMAT)
        log.addHandler(handler)
