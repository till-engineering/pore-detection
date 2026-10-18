"""Writer: writes all results of a run. Binding specification: docs/output_format.md.

Generic: writes every column and image the evaluation modules return without knowing them.
Tables in JSON are column-oriented ({column: [values]}), NaN is written as null.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from .types import ImageInfo

OVERLAY_COLOR = (255, 0, 0)              # RGB, pores in overlays
OVERLAY_ALPHA = 0.5                      # share of the overlay color on pore pixels
TIFF_FAST = [cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_PACKBITS]
JPEG_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 90]
# images of evaluation modules: <name>.tif, or <name>.jpg if the key ends with .jpg (pictures for humans)
MODULE_IMAGE_SUFFIXES = {".tif", ".jpg"}

# file names, see docs/output_format.md
PORES_JSON = "pores.json"
MASK_TIF = "mask.tif"
LABELS_TIF = "labels.tif"
ROI_MASK_TIF = "roi_mask.tif"
REJECTED_LABELS_TIF = "rejected_labels.tif"
REJECTED_JSON = "rejected.json"
CORRECTIONS_JSON = "corrections.json"
OVERLAY_TIF = "overlay.tif"
IMAGES_JSON = "images.json"
ID_VALUES_JSON = "id_values.json"
ERRORS_JSON = "errors.json"
CONFIG_USED = "config_used.yaml"
PLOTS_DIR = "plots"                     # in the image folder: <plot name>.png
# sheets of the results Excel: Pores first, Detection, ID and Errors last, module sheets in between
PORES_SHEET, DETECTION_SHEET, ID_SHEET, ERRORS_SHEET = "Pores", "Detection", "ID", "Errors"
OTHER_SHEET = "Other"                   # per-image values without a sheet in the layout
# previews of older program versions are still deleted when a run overwrites their folder
_OLD_FILES = {"preview_original.jpg", "preview_overlay.jpg", "preview_mask.png", "preview_roi.png",
              "preview_contrast.jpg"}
_IMAGE_FILES = {PORES_JSON, MASK_TIF, LABELS_TIF, ROI_MASK_TIF, REJECTED_LABELS_TIF, REJECTED_JSON,
                CORRECTIONS_JSON, OVERLAY_TIF} | _OLD_FILES


def excel_name(image_id: str) -> str:
    return f"{image_id}_results.xlsx"


def prepare_results_dir(output_dir: Path, image_id: str) -> Path:
    """Create <output_dir>/<ID>_results/ and delete the files a previous run created there.

    Only program files are deleted: the known top-level files and, inside the image folders
    (<ID>_*), the known result files plus *.tif / *.jpg (images of evaluation modules).
    Empty image folders are removed afterwards; anything else is left untouched.
    """
    results_dir = Path(output_dir) / f"{image_id}_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in (IMAGES_JSON, ID_VALUES_JSON, ERRORS_JSON, CONFIG_USED, excel_name(image_id)):
        (results_dir / name).unlink(missing_ok=True)
    for folder in results_dir.glob(f"{image_id}_*"):
        if folder.is_dir():
            _remove_program_files(folder)
    write_plots(results_dir, {})            # plots over the ID
    return results_dir


def _remove_program_files(folder: Path) -> None:
    """Delete the result files of one image folder, and the folder if it is empty then."""
    for file in folder.iterdir():
        if file.is_file() and (file.name in _IMAGE_FILES or file.suffix in MODULE_IMAGE_SUFFIXES):
            file.unlink()
    write_plots(folder, {})
    if not any(folder.iterdir()):
        folder.rmdir()


def write_config(results_dir: Path, config_dict: dict) -> None:
    with open(results_dir / CONFIG_USED, "w", encoding="utf-8") as file:
        yaml.safe_dump(config_dict, file, sort_keys=False, allow_unicode=True)


def write_image(results_dir: Path, info: ImageInfo, original: np.ndarray, labels: np.ndarray,
                rejected_labels: np.ndarray, rejected: pd.DataFrame, roi_mask: np.ndarray,
                per_pore: pd.DataFrame, images: dict[str, np.ndarray], save_full_overlay: bool) -> None:
    """Write all results of one image. `original`, `labels`, `rejected_labels` and `roi_mask` in
    full image size; `rejected`: pore_id, filter, reason of the rejected pores.

    If writing fails, the files written so far are removed again (never a half-written image).
    """
    folder = results_dir / info.name
    folder.mkdir(exist_ok=True)  # may still hold files of the user
    try:
        _imwrite(folder / ROI_MASK_TIF, roi_mask.view(np.uint8) * np.uint8(255), TIFF_FAST)
        write_pores(folder, original, labels, rejected_labels, rejected, per_pore, images, save_full_overlay)
    except BaseException:
        _remove_program_files(folder)
        raise


def write_pores(folder: Path, original: np.ndarray, labels: np.ndarray, rejected_labels: np.ndarray,
                rejected: pd.DataFrame, per_pore: pd.DataFrame, images: dict[str, np.ndarray],
                save_full_overlay: bool) -> None:
    """Everything of one image that depends on the pores (all but the ROI mask) – also rewritten
    after a manual correction."""
    write_table(folder / PORES_JSON, per_pore)
    mask = labels > 0
    _imwrite(folder / MASK_TIF, mask.view(np.uint8) * np.uint8(255), TIFF_FAST)
    _imwrite(folder / LABELS_TIF, labels, TIFF_FAST)
    _imwrite(folder / REJECTED_LABELS_TIF, rejected_labels, TIFF_FAST)
    write_table(folder / REJECTED_JSON, rejected)
    if save_full_overlay:
        _imwrite(folder / OVERLAY_TIF, overlay(original, mask), TIFF_FAST)
    for name, image in images.items():
        if name.endswith(".jpg"):
            _imwrite(folder / name, image, JPEG_QUALITY)
        else:
            _imwrite(folder / f"{name}.tif", image, TIFF_FAST)


def write_plots(folder: Path, plots: dict[str, bytes]) -> None:
    """plots/<name>.png of one image (PNG bytes per plot name); old plot images are replaced, an
    empty plots folder is removed."""
    plots_dir = folder / PLOTS_DIR
    if plots_dir.is_dir():
        for file in plots_dir.glob("*.png"):
            file.unlink()
    if plots:
        plots_dir.mkdir(exist_ok=True)
        for name, png in plots.items():
            (plots_dir / f"{name}.png").write_bytes(png)
    elif plots_dir.is_dir() and not any(plots_dir.iterdir()):
        plots_dir.rmdir()


def write_corrections(folder: Path, image: str, corrections: list[dict], added: list[int], removed: list[int],
                      drawn: list[int]) -> None:
    """corrections.json of one image; removed when there are no corrections (any more)."""
    path = folder / CORRECTIONS_JSON
    if not corrections:
        path.unlink(missing_ok=True)
        return
    data = {"image": image, "corrections": corrections, "added": added, "removed": removed, "drawn": drawn}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def write_summary(results_dir: Path, image_id: str, per_image: pd.DataFrame, errors: pd.DataFrame,
                  id_values: dict | None, layout: dict[str, str | None]) -> None:
    """Tables over the whole ID: JSON for the program (complete), one Excel file for humans.

    The Excel has one sheet per topic: `layout` maps every per-image column to its sheet, None =
    JSON only (see pipeline.excel_layout). Every sheet starts with the column image.
    id_values is None after a cancelled run (then no ID values are written).
    """
    write_table(results_dir / IMAGES_JSON, per_image)
    write_table(results_dir / ERRORS_JSON, errors)
    if id_values is not None:
        with open(results_dir / ID_VALUES_JSON, "w", encoding="utf-8") as file:
            json.dump(_json_value_dict(id_values), file, indent=2, ensure_ascii=False)

    sheets: dict[str, list[str]] = {}
    for column in per_image.columns:
        sheet = layout.get(column, OTHER_SHEET)
        if column != "image" and sheet is not None:
            sheets.setdefault(sheet, []).append(column)
    order = ([PORES_SHEET] if PORES_SHEET in sheets else []) \
        + [s for s in sheets if s not in (PORES_SHEET, DETECTION_SHEET)] \
        + ([DETECTION_SHEET] if DETECTION_SHEET in sheets else [])
    with pd.ExcelWriter(results_dir / excel_name(image_id), engine="xlsxwriter") as excel:
        for sheet in order:
            _excel_sheet(excel, sheet, per_image[["image", *sheets[sheet]]])
        if id_values is not None:
            _excel_sheet(excel, ID_SHEET, pd.DataFrame([id_values]))
        _excel_sheet(excel, ERRORS_SHEET, errors)


def _excel_sheet(excel: pd.ExcelWriter, name: str, table: pd.DataFrame) -> None:
    """One sheet: list cells as text (Excel has no list type), columns as wide as their header,
    header row frozen."""
    table.map(lambda v: str(v) if isinstance(v, list) else v).to_excel(excel, sheet_name=name, index=False)
    sheet = excel.sheets[name]
    for i, column in enumerate(table.columns):
        sheet.set_column(i, i, max(10, len(str(column)) + 2))
    sheet.freeze_panes(1, 0)


def write_table(path: Path, table: pd.DataFrame) -> None:
    columns = {column: _json_list(table[column]) for column in table.columns}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(columns, file, ensure_ascii=False)


def read_table(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pores in OVERLAY_COLOR, blended over the image (same dtype as rgb)."""
    color = np.array(OVERLAY_COLOR, dtype=np.float32) * (np.iinfo(rgb.dtype).max / 255)
    result = rgb.copy()
    result[mask] = (rgb[mask] * (1 - OVERLAY_ALPHA) + color * OVERLAY_ALPHA).astype(rgb.dtype)
    return result


def _imwrite(path: Path, image: np.ndarray, options: list[int]) -> None:
    """Write RGB or single-channel images; works with non-ASCII paths."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(path.suffix, image, options)
    if not ok:
        raise OSError(f"cannot encode {path}")
    data.tofile(path)


def _json_list(column: pd.Series) -> list:
    values = column.tolist()
    if column.dtype.kind == "f":
        values = [None if value != value else value for value in values]  # NaN -> null
    return values


def _json_value_dict(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value != value:
            value = None
        result[key] = value
    return result
