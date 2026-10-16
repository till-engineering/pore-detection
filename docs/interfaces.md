# Interfaces

> Binding specification of all data passed between modules. Interface version: 0.5 (draft)
>
> Rule: a module only receives and returns the types defined here. If a new method needs
> data that is not defined here, this document and `core/types.py` are changed first.
> There are no untyped "extra" fields.
>
> Changes 0.1 → 0.2: NumPy instead of torch, `ImageContext`, new stage `sample`,
> postprocessing produces `LabeledPores`, removed `probability` and `warnings`.
> Changes 0.2 → 0.3: gray working image, `LabeledPores.properties` + `get()`,
> mandatory CPU path + optional GPU path, read-only context, tiles are copies.
> Changes 0.3 → 0.4: `ScaleDetector` returns `ScaleResult` (scale + mask of the scale overlay).
> Changes 0.4 → 0.5: `LabeledPores` carries rejected pores (`rejected_labels`, `rejected`); pore ids are
> stable and may have gaps; `LabelStep.changes_pores`.

---

## 1. Conventions

| Topic | Convention |
|-------|------------|
| Array library | `numpy.ndarray` everywhere, `pandas.DataFrame` for tables. No torch tensors at interfaces |
| Working image | gray, `(H, W)`, `float32`, range `0.0 … 1.0` (independent of 8/16 bit source) |
| Original image | color, `(H, W, 3)`, RGB, source dtype (`uint8` / `uint16`) |
| Masks | `(H, W)`, `bool`, `True` = feature present |
| Labels | `(H, W)`, `int32`, `0` = background, `1 … count` = pore id |
| Coordinates | origin top-left, `y` = row (down), `x` = column (right); sizes given as `(height, width)` |
| Lengths | pixels (`_px`) and/or micrometres (`_um`); the unit is always part of the name |
| Areas | `_px2` / `_um2` |
| Scale | `pixel_size_um` = micrometres per pixel, square pixels; `None` = unknown |
| Mutability | types are frozen dataclasses; a step returns **new** data and never modifies its input. Arrays in `ImageContext` are read-only (`writeable=False`) |
| Self-contained | every method works with whatever a previous stage may deliver in the documented format |
| Gray conversion | done once by `core` with `input.gray_weights`; methods needing color use `ctx.original` |

---

## 2. Data types (`core/types.py`)

### 2.1 `ImageInfo` – metadata of the source image

| Field | Type | Meaning |
|-------|------|---------|
| `path` | `Path` | source file |
| `image_id` | `str` | name of the input folder (the sample ID) |
| `name` | `str` | `<image_id>_<file stem>`, used in all outputs |
| `height_px` | `int` | full image height |
| `width_px` | `int` | full image width |
| `bit_depth` | `int` | 8 or 16 |
| `offset_px` | `tuple[int, int]` | `(y, x)` of the current data inside the full image (crop to ROI, tile) |

### 2.2 `ImageContext` – read-only context, passed to every step from `preprocessing` on

| Field | Type | Shape | Meaning |
|-------|------|-------|---------|
| `info` | `ImageInfo` | – | |
| `original` | `ndarray uint8/uint16` | `(H, W, 3)` | original color image as loaded, never modified |
| `roi_mask` | `ndarray bool` | `(H, W)` | `True` = sample, `False` = embedding medium / background |
| `pixel_size_um` | `float \| None` | – | scale |

Created by `core` after the `sample` stage and cropped to the bounding box of `roi_mask`.
A tileable step receives a context cropped to its tile (incl. overlap) with its own
`info.offset_px` (position of the tile's top-left pixel incl. overlap). Where the overlap reaches
beyond the cropped area, `original`, `roi_mask` and the image are mirrored there, and
`offset_px` may be negative.

### 2.3 `LabeledPores` – output of postprocessing, input of evaluation

| Field | Type | Shape | Meaning |
|-------|------|-------|---------|
| `labels` | `ndarray int32` | `(H, W)` | `0` = no counted pore, else `pore_id` |
| `count` | `int` | – | number of counted pores |
| `properties` | `DataFrame` | `count` rows | one row per counted pore, column `pore_id` (ascending) + any columns added by label steps; after labeling it contains only `pore_id` |
| `rejected_labels` | `ndarray int32 \| None` | `(H, W)` | `pore_id` of rejected pores (filters, by hand); `None` = none |
| `rejected` | `DataFrame` | one row per rejected pore | `pore_id`, `filter` (registered name, `manual` = by hand), `reason` (text) |

Counted and rejected pores share **one id space**. Ids are unique and ascending but may have
gaps: a rejected pore keeps its id, so it is the same pore when it is counted again; new pores
get `next_id`. Evaluation modules only see the counted pores.

Helpers for label steps (`core/types.py`, `core/geometry.py`):

```python
pores.reject(ids, filter_name, reasons) -> LabeledPores   # move counted pores to the rejected ones
pores.relabeled(labels) -> LabeledPores                    # new counted pores (merge, split, …); rejected stay
pores.ids, pores.next_id, pores.pixels                     # ids, first free id, cached pore pixel list
geometry.areas / boxes / shapes / renumber                 # shape values only as decision aids, not output
```

```python
pores.get("area_px2") -> ndarray   # one value per pore, ordered by pore_id
                                   # raises KeyError with a clear message if the column is missing
```

- The pore mask is `labels > 0`.
- Label steps that compute values (e.g. `pore_properties`) return a new `LabeledPores` with
  additional columns in `properties`.
- **Rule (not checked):** steps that change `labels` (merge, filters) must come **before** steps
  that compute `properties` in `config.yaml`, otherwise values are stale.

### 2.4 `EvaluationResult` – output of one evaluation module for one image

| Field | Type | Meaning |
|-------|------|---------|
| `per_image` | `dict[str, float \| int \| str]` | one value per column of the per-image results |
| `per_pore` | `DataFrame \| None` | one row per pore, must contain column `pore_id` |
| `images` | `dict[str, ndarray]` | images to save (e.g. overlay), key = output name |

Column and image names are chosen by the module. If two modules return the same name, the run
is aborted at the first successful image with an error naming both modules.

Modules never write files. The writer in `core` writes all results according to
`docs/output_format.md`.

### 2.5 `ScaleResult` – output of `sample.scale`

| Field | Type | Shape | Meaning |
|-------|------|-------|---------|
| `pixel_size_um` | `float \| None` | – | micrometres per pixel, `None` = unknown |
| `scale_mask` | `ndarray bool \| None` | `(H, W)` | `True` = scale overlay (bar, label, box), never sample; `None` = nothing found (default) |
| `bar_px` | `tuple[int, int, int, int] \| None` | – | measured bar `(top, bottom, left, right)`, for the optical check; `None` = none |
| `label` | `str \| None` | – | text read from the scale label, for the optical check; `None` = none |

`core` removes `scale_mask` from `roi_mask` before cropping and writes `pixel_size_um`, `label`,
`bar_px` and the bounding box of `scale_mask` to `images.json` (see `docs/output_format.md`).
A method returns mask and bar even if it found the overlay but could not read its label.

---

## 3. Stage interfaces (`core/step.py`)

```python
class Step:
    name: str                 # registered name, used in config.yaml
    params: dict[str, str]    # parameter name -> description (for list_methods)
    tileable: bool            # local operation, may run on tiles
    tile_overlap_px: int      # required border around each tile
    def __init__(self, **params): ...   # params exactly as in config.yaml

# --- sample: always on the original image, whole image, not tileable ---
class RoiDetector(Step):
    def detect_roi(self, info: ImageInfo, original: ndarray) -> ndarray: ...          # (H, W) bool

class ScaleDetector(Step):
    def detect_scale(self, info: ImageInfo, original: ndarray) -> ScaleResult: ...

# --- preprocessing: optional filters on the gray working image ---
class Filter(Step):
    def process(self, ctx: ImageContext, image: ndarray) -> ndarray: ...   # (H, W) float32 0..1

# --- detection: exactly one, delivers everything that may be a pore ---
class Detector(Step):
    def detect(self, ctx: ImageContext, image: ndarray) -> ndarray: ...    # gray in, (H, W) bool out

# --- postprocessing ---
class MaskStep(Step):
    def process(self, ctx: ImageContext, mask: ndarray) -> ndarray: ...    # (H, W) bool

class Labeler(Step):                                                       # not tileable
    def label(self, ctx: ImageContext, mask: ndarray) -> LabeledPores: ...

class LabelStep(Step):                                                     # not tileable
    changes_pores: bool = True   # False = only adds values per pore (run again after a manual correction)
    def process(self, ctx: ImageContext, pores: LabeledPores) -> LabeledPores: ...

# --- evaluation: whole image, modules of one image run in parallel ---
class Evaluator(Step):
    def evaluate(self, ctx: ImageContext, pores: LabeledPores) -> EvaluationResult: ...
    def finalize(self, per_image: DataFrame,
                 per_pore: Iterator[DataFrame]) -> dict[str, float | int | str]: ...   # optional
```

### 3.1 CPU and GPU path

- The methods above are the **CPU path – mandatory** for every method.
- Tileable steps may additionally implement a **GPU path** that receives a batch of tiles:
  `process_gpu(ctxs: list[ImageContext], items: list[ndarray]) -> list[ndarray]`
  (`detect_gpu` for detectors). Inputs and outputs are NumPy; the method moves data with
  `core.device.to_gpu()` / `to_numpy()`.
- `core` uses the GPU path if it exists and a GPU is available, otherwise the CPU path.
  One name in the config, same config on every machine.

### 3.2 `finalize`

Called once at the end of a complete run with the results of all images of the ID:
`per_image` from memory (one row per image), `per_pore` read lazily from the written JSON files.
Returns values for the whole ID.

### 3.3 Helpers offered by `core`

| Helper | Purpose |
|--------|---------|
| `core.imaging.to_gray(original, weights)` | same gray conversion as `core` uses |
| `core.imaging.downscale(image, factor)` | work on an overview, e.g. embedding medium detection on panoramas |
| `core.device.to_gpu(array)` / `to_numpy(tensor)` | for GPU paths |

---

## 4. Rules applied by `core`

| Rule | Detail |
|------|--------|
| Gray | after loading, `original` is converted once to the gray working image (`input.gray_weights`) |
| Scale overlay | `roi_mask & ~scale_mask`; `RoiDetector` and `ScaleDetector` run in parallel (both only read the original) |
| Crop | after `sample`, all data is cropped to the bounding box of `roi_mask` |
| Skip tiles | tiles without any ROI pixel are not computed (filters: unchanged, masks: `False`) |
| ROI before labeling | mask pixels outside `roi_mask` are set to `False` right before the `Labeler` |
| Tiling | tileable steps get one tile (a copy); consecutive tileable steps stay tiled, overlaps are summed; borders padded with `reflect`; tiles run in parallel threads |
| Scale | a step with a `_um` parameter and `pixel_size_um is None` raises → image is skipped |

## 5. Data flow per image

```
file ──load──► ImageInfo + original (color)
     ──RoiDetector──► roi_mask
     ──ScaleDetector──► ScaleResult (pixel_size_um + scale_mask)   (parallel to RoiDetector)
     ──core: roi_mask & ~scale_mask
     ──core: crop + gray──► ImageContext + gray working image (float32)
     ──[Filter]*──► gray image
     ──Detector──► mask
     ──[MaskStep]*──► mask
     ──core: mask & roi_mask──► Labeler ──► LabeledPores
     ──[LabelStep]*──► LabeledPores (+ properties)
     ──[Evaluator]* (parallel)──► EvaluationResult per module
     ──core writer──► <ID>_results/<ID>_<image>/
end of run: Evaluator.finalize ──► <ID>_results/ (ID values)
```

`[...]*` = chain of zero or more steps, order as in `config.yaml`.
