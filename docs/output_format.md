# Output Format

> Binding specification of everything the program writes to disk.
> Status: folder structure, formats, file names and JSON layout defined; run metadata open.

The writer (`core/writer.py`) is a fixed part of `core` (not a plug-in, not configurable apart
from `output.save_full_overlay`). It is generic: it writes every column and image returned by the
evaluation modules without knowing them. Evaluation modules never write files themselves.

## 1. Output folder structure

One run processes one ID (= name of the input folder).

```
<output dir>/
└── <ID>_results/
    ├── <ID>_<image>/
    │   ├── pores.json              per-pore table
    │   ├── mask.tif                pore mask, full resolution
    │   ├── labels.tif              pore id per pixel (int32), full resolution
    │   ├── roi_mask.tif            sample mask (255 = sample), full resolution
    │   ├── rejected_labels.tif     pore id of rejected pores per pixel (int32), full resolution
    │   ├── rejected.json           rejected pores: pore_id, filter (manual = by hand), reason
    │   ├── corrections.json        manual corrections (only if there are any, see PLAN.md 7.3)
    │   ├── overlay.tif             only with output.save_full_overlay: true
    │   └── <name>.tif              images returned by evaluation modules (key = name)
    ├── images.json                 one row per image
    ├── id_values.json              values over the whole ID (Evaluator.finalize)
    ├── errors.json                 skipped images
    ├── <ID>_results.xlsx           sheets: images, id_values, errors
    └── config_used.yaml            copy of the config used for this run
```

- `<image>` = file name without extension. Images always lie directly in the ID folder.
- An image folder is written only after the image passed all stages; if writing fails, the files
  written so far are removed again.
- `images.json`, `errors.json` and the Excel file are written at the end of the run, also after a
  cancel or abort. `id_values.json` (and the `id_values` sheet) only after a complete run.
- A new run **overwrites** `<ID>_results/`. Only files created by the program are deleted (the
  names above, `*.tif` inside image folders); other files are left untouched, image folders are
  removed only if empty.

## 2. Formats

| Content | Format | Read by | Written |
|---------|--------|---------|---------|
| Tables (per pore, per image, ID values, errors) | JSON | viewer, `finalize` | always |
| Per image, ID values, errors | Excel (`xlsxwriter`) | humans only – never read back by the program | always |
| Per-pore tables | JSON only, no Excel (may exceed Excel's row limit) | | |
| Previews | – (the viewer computes its overview from the full-resolution files) | | never |
| Mask, full resolution | 8-bit TIFF (0 / 255), PackBits | further analysis | always |
| Labels (pore id per pixel), full resolution | 32-bit TIFF, PackBits | viewer | always |
| ROI mask (sample = 255), full resolution | 8-bit TIFF, PackBits | viewer | always |
| Original, full resolution | – (stays in the input folder) | viewer zoom reads it from there | never |
| Overlay, full resolution | TIFF, PackBits, source bit depth | reports | only if `output.save_full_overlay: true` |
| Images of evaluation modules | TIFF, PackBits | | when returned |

- Full overlay: pore pixels blended 50 % with red over the original.
- Preview files of older versions (`preview_*.jpg/png`) are deleted when a run overwrites the folder.
- Large images are written with fast settings (PackBits) – PNG with default compression takes
  tens of seconds for panoramas.
- Mask and full overlay always have the size of the original image (`height_px × width_px`).

## 3. JSON layout

Tables are **column-oriented**: one key per column, each value a list with one entry per row.
`NaN` is written as `null`. Read with `pd.DataFrame(json.load(file))` (`writer.read_table`).

```json
{"pore_id": [1, 2, 3], "area_px2": [901, 113, 441], "area_um2": [null, null, null]}
```

`id_values.json` is a flat object: `{"porosity_percent": 5.13}`.

## 4. Per-image results

`images.json` / sheet `images`: one row per successful image, in processing order.

| Column | Meaning |
|--------|---------|
| `image` | `<ID>_<image>` (= name of the image folder) |
| `path` | source file |
| `pixel_size_um` | scale in µm per px, `null` = unknown |
| `scale_label` | text read from the scale label, `null` = none |
| `scale_bar_px` | measured scale bar `[top, bottom, left, right]` (bottom/right exclusive), `null` = none |
| `scale_box_px` | scale overlay `[top, bottom, left, right]` = bounding box of `scale_mask`, `null` = none |
| `frame_origin_px` | origin of the sample frame `[y, x]` (full image), `null` = no frame |
| `frame_angle_deg` | direction of the frame's x axis, counter-clockwise from the image x axis (as seen) |
| `frame_thickness_px` / `frame_thickness_um` | inner edge to outer edge (`_um` only with a scale) |
| `frame_length_px` / `frame_length_um` | left edge to right edge along the inner edge |
| `frame_wedge_deg` | angle between the fitted inner and outer edge (check only) |
| `manual_corrections` | number of manual corrections of this image (0 after a run) |
| … | all `per_image` values of the evaluation modules |

In Excel the list columns (boxes, `frame_origin_px`) are written as text (`"[top, bottom, left, right]"`).

`pores.json`: column `pore_id` + all `per_pore` columns of the evaluation modules, joined on
`pore_id`.

## 5. ID values

`id_values.json` / sheet `id_values`: all values returned by `finalize` of the evaluation
modules. No per-pore listing.

## 6. Error list

`errors.json` / sheet `errors`, one row per skipped image:

| Column | Meaning |
|--------|---------|
| `image` | `<ID>_<image>` |
| `stage` | e.g. `load`, `detection`, `evaluation`, `write` |
| `method` | registered method name, `-` for core stages |
| `message` | `<exception type>: <message>` |

## 7. Run metadata

TODO: software version, device, timings.

## 8. Manual corrections

Made in the viewer (PLAN.md 7.3) and written straight into the current results: the image folder
(`labels.tif`, `rejected_labels.tif`, `rejected.json`, `pores.json`, `mask.tif`), `images.json`, the
Excel file and `id_values.json`. A new run starts again from the automatic result and deletes them.

`corrections.json` (column-free object, coordinates in pixels of the full image):

```json
{"image": "<ID>_<image>",
 "corrections": [{"action": "remove", "x": 3715.0, "y": 1829.0},
                 {"action": "draw", "x": 3715.5, "y": 1829.5, "points": [[3662.0, 1804.0], …]}],
 "added": [7, 369], "removed": [6], "drawn": [369]}
```

`action`: `restore` (count a rejected pore), `remove` (reject a counted pore / delete a drawn one),
`draw` (outline becomes a pore), `area` (counted pores completely inside are rejected).
`added`: pore ids counted by hand, `removed`: rejected by hand, `drawn`: drawn by hand.
