# Interfaces

> Binding specification of all data passed between modules. Interface version: 0.12 (draft)
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
> Changes 0.5 → 0.6: new stage `sample.frame` (`FrameDetector`) and type `SampleFrame` (coordinate
> system of the sample); `ImageContext.frame`, `require_frame()`, `sample_coords()`.
> Changes 0.6 → 0.7: new stage `plots` (`Plot`): pictures of the finished values of one image.
> Changes 0.7 → 0.8: optional `Step.choices` (allowed values of a parameter, shown as a selection
> in the launcher).
> Changes 0.8 → 0.9: optional `Plot.variant` (second view of a plot, drawn too, switch in the viewer).
> Changes 0.9 → 0.10: optional `Evaluator.excel_sheet`, `Evaluator.excel_hidden` (results Excel by topic).
> Changes 0.10 → 0.11: optional `Plot.plot_id` (the same plot over the whole ID).
> Changes 0.11 → 0.12: optional `Plot.needs_picture` (plot() gets `picture=`: the straightened sample,
> `core.plotting.SamplePicture` with pore centroids, or None without frame) and `Plot.size_in`.

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
| `frame` | `SampleFrame \| None` | – | coordinate system of the sample, `None` = `sample.frame` is `none` |

```python
ctx.require_frame() -> SampleFrame              # raises (image skipped) if frame is None
ctx.sample_coords() -> (x_px, z_px)             # float32 (H, W) each: every pixel of this context's
                                                # area (crop / tile) in the sample frame
```

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

### 2.6 `SampleFrame` – output of `sample.frame`

Right-angled coordinate system of a rectangular section, in pixel coordinates of the **full
original image** (pixel index = pixel centre), so it stays valid after cropping and on tiles.
Fixed convention: the **top edge is the outer surface**, the **bottom edge the inner surface**.

| Field | Type | Meaning |
|-------|------|---------|
| `origin_px` | `tuple[float, float]` | `(y, x)` origin: inner (bottom) edge meets the left edge |
| `angle_deg` | `float` | direction of x, counter-clockwise from the image x axis as seen on screen |
| `thickness_px` | `float` | inner edge (`z = 0`) to outer edge (`z = thickness_px`) |
| `length_px` | `float` | left edge to right edge, along the inner edge |
| `wedge_deg` | `float` | angle between the fitted inner and outer edge – only for checking, not used |

```python
frame.x_axis -> (dy, dx)        # unit vector along the outer edge, to the right
frame.z_axis -> (dy, dx)        # unit vector perpendicular to it, upwards (outer side)
frame.to_sample(y_px, x_px) -> (x_px, z_px)   # image points (full image coordinates) -> frame
```

- x runs along the outer edge, z perpendicular to it; z is in px, the relative height is
  `z_px / thickness_px` (0 = inner, 1 = outer surface).
- A step that needs the frame calls `ctx.require_frame()` or `ctx.sample_coords()`; without a
  frame the image fails with a clear message (like `_um` parameters without a scale).

---

## 3. Stage interfaces (`core/step.py`)

```python
class Step:
    name: str                 # registered name, used in config.yaml
    params: dict[str, str]    # parameter name -> description (for list_methods)
    choices: dict[str, list[str]]   # optional: parameter -> allowed values (launcher selection)
    tileable: bool            # local operation, may run on tiles
    tile_overlap_px: int      # required border around each tile
    def __init__(self, **params): ...   # params exactly as in config.yaml

# --- sample: always on the original image, whole image, not tileable ---
class RoiDetector(Step):
    def detect_roi(self, info: ImageInfo, original: ndarray) -> ndarray: ...          # (H, W) bool

class ScaleDetector(Step):
    def detect_scale(self, info: ImageInfo, original: ndarray) -> ScaleResult: ...

class FrameDetector(Step):        # on the roi_mask (full image, scale overlay removed), after roi + scale
    def detect_frame(self, info: ImageInfo, roi_mask: ndarray) -> SampleFrame | None: ...  # None = no frame

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
    excel_sheet: str | None = None    # optional: sheet of the results Excel (None = module name)
    excel_hidden: set[str] = set()    # optional: per-image values written to JSON only
    def finalize(self, per_image: DataFrame,
                 per_pore: Iterator[DataFrame]) -> dict[str, float | int | str]: ...   # optional
```

```python
# --- plots: pictures of the finished values of one image, any number ---
class Plot(Step):
    columns: list[str] = []         # per-image values needed (missing -> plot skipped, logged)
    pore_columns: list[str] = []    # per-pore values needed (missing -> plot skipped, logged)
    variant: str = ""               # optional second view, e.g. "normalized"
    def plot(self, ax, image: dict, pores: DataFrame) -> None: ...   # draw into a matplotlib Axes
    # optional: the same plot over the whole ID -> <ID>_results/plots/ (end of a complete run, after corrections)
    def plot_id(self, ax, images: DataFrame, pores: DataFrame) -> None: ...   # pores: all images + column image
```

- With `variant` set, `core` draws the plot twice: `plot(ax, image, pores)` (normal view) and
  `plot(ax, image, pores, variant=<variant>)`. Both are saved (`plots/<name>.png`,
  `plots/<name>.<variant>.png`); the viewer shows a checkbox named after the variant to switch.

- `image` is the row of the image as in `images.json`, `pores` the table as in `pores.json` –
  a plot only knows column names, never an evaluation module.
- The plot only draws into `ax`. Figure, size, look (`core.plotting.STYLE`, `COLORS`) and saving
  as PNG are done by `core` (`core/plotting.py`). Import matplotlib only inside `plot()`.
- Something else missing → `raise core.plotting.MissingData(reason)`. A plot that cannot be
  drawn is skipped and logged; the image does **not** fail.
- Plots are drawn while the image is written (background) and again after every manual correction.

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
| Sample frame | `FrameDetector` runs on that `roi_mask` (full image); the result is `ctx.frame` and is written to `images.json` |
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
     ──FrameDetector──► SampleFrame | None
     ──core: crop + gray──► ImageContext + gray working image (float32)
     ──[Filter]*──► gray image
     ──Detector──► mask
     ──[MaskStep]*──► mask
     ──core: mask & roi_mask──► Labeler ──► LabeledPores
     ──[LabelStep]*──► LabeledPores (+ properties)
     ──[Evaluator]* (parallel)──► EvaluationResult per module
     ──core writer──► <ID>_results/<ID>_<image>/
     ──[Plot]* (while writing)──► <ID>_results/<ID>_<image>/plots/<name>.png
end of run: Evaluator.finalize ──► <ID>_results/ (ID values)
```

`[...]*` = chain of zero or more steps, order as in `config.yaml`.
