"""Tiling: run consecutive tileable steps tile by tile, in parallel threads or in GPU batches.

A tile is a core area of tile_size x tile_size px plus the summed overlap of all steps of the
run on every side. Inside the image the overlap is real data; at the border of the image it is
mirrored (cv2.BORDER_REFLECT_101, the default border of OpenCV filters). Every tile runs all
steps of the run one after another; only its core area is kept, so the overlap absorbs the
border effects of all steps together.

Tiles whose core area contains no sample pixel are not computed: filters leave the image
unchanged there, steps that produce a mask set it to False.

GPU (device "cuda", at least one step of the run has a GPU path): the free GPU memory is
measured; if the whole image fits, it is computed in one piece, otherwise the tiles are passed
in batches. Steps without a GPU path run on the CPU in between (tiles in parallel). On
out-of-memory the batch is halved and retried.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np

from . import device
from .step import Detector, Filter, Step
from .types import ImageContext

# Estimated GPU memory per pixel of a tile: float32 input, padded copy, intermediate, output.
# Only a starting point for the batch size; too large batches are halved on out-of-memory.
GPU_BYTES_PER_PIXEL = 16


class StepError(Exception):
    """A step failed; carries the step so the error can name stage and method."""

    def __init__(self, step: Step):
        super().__init__(step.name)
        self.step = step


@dataclass(frozen=True)
class Gpu:
    """GPU settings from config.runtime."""

    memory_fraction: float      # max. share of the free GPU memory
    batch_size: int | str       # tiles per GPU call, "auto" = from free memory


def apply(step: Step, ctx: ImageContext, item: np.ndarray) -> np.ndarray:
    """Call the CPU path of a filter, detector or mask step."""
    try:
        return step.detect(ctx, item) if isinstance(step, Detector) else step.process(ctx, item)
    except Exception as error:
        raise StepError(step) from error


def gpu_path(step: Step):
    """The bound GPU path of a step (process_gpu / detect_gpu) or None."""
    return getattr(step, "detect_gpu" if isinstance(step, Detector) else "process_gpu", None)


def groups(steps: list[Step]) -> list[list[Step]]:
    """Split a chain into runs: consecutive tileable steps together, every other step alone."""
    result: list[list[Step]] = []
    for step in steps:
        if step.tileable and result and result[-1][-1].tileable:
            result[-1].append(step)
        else:
            result.append([step])
    return result


def run_tiled(steps: list[Step], ctx: ImageContext, data: np.ndarray, tile_size: int,
              pool: ThreadPoolExecutor, gpu: Gpu | None = None) -> np.ndarray:
    """Run a chain of tileable steps on all tiles of `data`. gpu=None: CPU only."""
    if gpu is not None and not any(gpu_path(step) for step in steps):
        gpu = None
    height, width = data.shape
    if gpu is not None:
        try:
            return _run_gpu(steps, ctx, data, tile_size, pool, gpu)
        finally:
            device.release()
    if height <= tile_size and width <= tile_size:  # one tile = the whole image, no padding needed
        return _run_batch(steps, [ctx], [data], pool, gpu=False)[0]

    tiles = _Tiles(steps, ctx, data, tile_size)

    def work(y0: int, x0: int) -> None:
        tile_ctx, item = tiles.cut(y0, x0)
        for step in steps:
            item = apply(step, tile_ctx, item)
        tiles.paste(y0, x0, item)

    for future in [pool.submit(work, y, x) for y, x in tiles.starts]:
        future.result()  # re-raises the first error of a tile
    return tiles.result


def _run_gpu(steps, ctx, data, tile_size, pool, gpu: Gpu) -> np.ndarray:
    height, width = data.shape
    budget = device.free_gpu_memory_bytes() * gpu.memory_fraction
    if gpu.batch_size == "auto" and height * width * GPU_BYTES_PER_PIXEL <= budget:
        try:
            return _run_batch(steps, [ctx], [data], pool, gpu=True)[0]
        except Exception as error:
            if not device.is_out_of_memory(error):
                raise
            device.release()  # estimate too optimistic -> tiles

    tiles = _Tiles(steps, ctx, data, tile_size)
    if gpu.batch_size == "auto":
        tile_px = (tile_size + 2 * tiles.overlap) ** 2
        batch_size = max(1, int(budget // (tile_px * GPU_BYTES_PER_PIXEL)))
    else:
        batch_size = gpu.batch_size

    pending = list(tiles.starts)
    while pending:
        batch, rest = pending[:batch_size], pending[batch_size:]
        cut = [tiles.cut(y0, x0) for y0, x0 in batch]
        try:
            items = _run_batch(steps, [c for c, _ in cut], [i for _, i in cut], pool, gpu=True)
        except Exception as error:
            if not device.is_out_of_memory(error) or batch_size == 1:
                raise
            device.release()
            batch_size = max(1, len(batch) // 2)
            continue  # same tiles again, smaller batch
        for (y0, x0), item in zip(batch, items):
            tiles.paste(y0, x0, item)
        pending = rest
    return tiles.result


def _run_batch(steps, ctxs: list[ImageContext], items: list[np.ndarray], pool: ThreadPoolExecutor,
               gpu: bool) -> list[np.ndarray]:
    """All steps on a batch of items: GPU path where available, otherwise CPU in parallel."""
    for step in steps:
        path = gpu_path(step) if gpu else None
        if path is not None:
            try:
                items = path(ctxs, items)
            except Exception as error:
                if device.is_out_of_memory(error):
                    raise
                raise StepError(step) from error
        elif len(items) == 1:
            items = [apply(step, ctxs[0], items[0])]
        else:
            items = list(pool.map(lambda c, i, s=step: apply(s, c, i), ctxs, items))
    return items


class _Tiles:
    """Tile grid of one run: cut tiles with overlap, paste their core areas into the result."""

    def __init__(self, steps: list[Step], ctx: ImageContext, data: np.ndarray, tile_size: int):
        self.ctx, self.data, self.tile_size = ctx, data, tile_size
        self.overlap = sum(step.tile_overlap_px for step in steps)
        self.to_mask = not isinstance(steps[-1], Filter)
        height, width = data.shape
        self.result = np.empty((height, width), dtype=bool if self.to_mask else data.dtype)
        self.starts = []
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                y1, x1 = self._end(y0, x0)
                if ctx.roi_mask[y0:y1, x0:x1].any():
                    self.starts.append((y0, x0))
                else:  # no sample in this tile: not computed
                    self.result[y0:y1, x0:x1] = False if self.to_mask else data[y0:y1, x0:x1]

    def _end(self, y0: int, x0: int) -> tuple[int, int]:
        return min(y0 + self.tile_size, self.data.shape[0]), min(x0 + self.tile_size, self.data.shape[1])

    def cut(self, y0: int, x0: int) -> tuple[ImageContext, np.ndarray]:
        y1, x1 = self._end(y0, x0)
        o = self.overlap
        box = (y0 - o, y1 + o, x0 - o, x1 + o)
        offset_y, offset_x = self.ctx.info.offset_px
        tile_ctx = ImageContext(
            info=replace(self.ctx.info, offset_px=(offset_y + y0 - o, offset_x + x0 - o)),
            original=_cut(self.ctx.original, box),
            roi_mask=_cut(self.ctx.roi_mask, box),
            pixel_size_um=self.ctx.pixel_size_um,
        )
        return tile_ctx, _cut(self.data, box)

    def paste(self, y0: int, x0: int, item: np.ndarray) -> None:
        y1, x1 = self._end(y0, x0)
        o = self.overlap
        self.result[y0:y1, x0:x1] = item[o:o + y1 - y0, o:o + x1 - x0]


def _cut(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Region (top, bottom, left, right) of `array`; parts outside are mirrored.

    Inside the image: a read-only view (no copy). At the border: a mirrored copy.
    """
    top, bottom, left, right = box
    height, width = array.shape[:2]
    tile = array[max(0, top):min(bottom, height), max(0, left):min(right, width)]
    pads = (max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width))
    if not any(pads):
        tile = tile.view()
        tile.flags.writeable = False
        return tile
    if tile.dtype == bool:
        return cv2.copyMakeBorder(tile.view(np.uint8), *pads, cv2.BORDER_REFLECT_101).view(bool)
    return cv2.copyMakeBorder(tile, *pads, cv2.BORDER_REFLECT_101)
