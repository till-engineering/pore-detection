"""Geometry of numbered pores for postprocessing steps: bounding boxes, renumbering, shape values.

The shape values here are decision aids for filters only (definitions as in Pore_Detection_V2,
which used scikit-image regionprops); they are not written to the output. Output values per pore
come from label steps / evaluation modules.
"""

import math

import cv2
import numpy as np
import pandas as pd

# below this pixel count every shape value is dominated by the pixel grid and says nothing
MIN_PIXELS_FOR_SHAPE = 25


def pixels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices (ascending) and ids of all pore pixels. Pores cover a small part of the image,
    so working on these lists is much cheaper than passes over the whole image."""
    flat = np.flatnonzero(labels)
    return flat, labels.ravel()[flat]


def ids_of(labels: np.ndarray) -> np.ndarray:
    """All ids > 0 present in `labels`, ascending."""
    return np.flatnonzero(np.bincount(pixels(labels)[1]))


Pixels = tuple[np.ndarray, np.ndarray]      # result of pixels(); pass it on to avoid a new pass


def boxes(labels: np.ndarray, ids: np.ndarray | None = None,
          pixel_list: Pixels | None = None) -> dict[int, tuple[int, int, int, int]]:
    """Bounding box (top, bottom, left, right; bottom/right exclusive) per pore id (all ids > 0
    if `ids` is None)."""
    flat, values = pixel_list if pixel_list is not None else pixels(labels)
    if flat.size == 0:
        return {}
    row, col = np.divmod(flat, labels.shape[1])
    size = int(values.max()) + 1
    # pixels come in row order: first / last occurrence of an id = its top / bottom row
    top, bottom = np.full(size, -1, dtype=np.int64), np.full(size, -1, dtype=np.int64)
    top[values[::-1]] = row[::-1]               # with repeated indices the last assignment wins
    bottom[values] = row
    left, right = np.full(size, labels.shape[1], dtype=np.int64), np.full(size, -1, dtype=np.int64)
    np.minimum.at(left, values, col)
    np.maximum.at(right, values, col)
    wanted = np.flatnonzero(top >= 0) if ids is None else np.asarray(ids, dtype=np.int64)
    return {int(i): (int(top[i]), int(bottom[i]) + 1, int(left[i]), int(right[i]) + 1) for i in wanted}


def renumber(labels: np.ndarray, start: int = 1) -> np.ndarray:
    """Ids 0 stays 0, all other ids become start, start + 1, … (in ascending order)."""
    flat, values = pixels(labels)
    present = np.bincount(values) > 0
    present[0] = False
    new = (np.cumsum(present) + (start - 1)).astype(np.int32)
    out = np.zeros(labels.shape, dtype=np.int32)
    out.ravel()[flat] = new[values]
    return out


def areas(labels: np.ndarray, ids: np.ndarray, pixel_list: Pixels | None = None) -> np.ndarray:
    """Pixel count per id in `ids`."""
    values = (pixel_list if pixel_list is not None else pixels(labels))[1]
    return np.bincount(values, minlength=int(np.max(ids, initial=0)) + 1)[ids]


def equivalent_diameter(area_px2: np.ndarray) -> np.ndarray:
    """Diameter of the circle with the same area."""
    return np.sqrt(4.0 * np.asarray(area_px2, dtype=np.float64) / math.pi)


def shapes(labels: np.ndarray, ids: np.ndarray, pixel_list: Pixels | None = None) -> pd.DataFrame:
    """Shape values per pore: pore_id, area_px2, major_axis_px, minor_axis_px (ellipse with the
    same second moments), solidity (area / convex hull area), feret_max_px (largest extent over
    the pixel corners), aspect_ratio, roundness (4 A / (pi feret²), 1 for a circle)."""
    rows = []
    for pore_id, (top, bottom, left, right) in boxes(labels, ids, pixel_list).items():
        mask = (labels[top:bottom, left:right] == pore_id).view(np.uint8)
        rows.append((pore_id, *_shape(mask)))
    table = pd.DataFrame(rows, columns=["pore_id", "area_px2", "major_axis_px", "minor_axis_px", "solidity",
                                        "feret_max_px"])
    with np.errstate(divide="ignore", invalid="ignore"):
        table["aspect_ratio"] = np.where(table["minor_axis_px"] > 0,
                                         table["major_axis_px"] / table["minor_axis_px"], np.inf)
        table["roundness"] = 4.0 * table["area_px2"] / (math.pi * table["feret_max_px"] ** 2)
    return table


def _shape(mask: np.ndarray) -> tuple[int, float, float, float, float]:
    area = int(mask.sum())
    m = cv2.moments(mask, binaryImage=True)
    # covariance of the pixel coordinates -> ellipse axes 4 * sqrt(eigenvalue) (skimage definition)
    a, b, c = m["mu20"] / area, m["mu11"] / area, m["mu02"] / area
    root = math.sqrt(max(0.0, ((a - c) / 2) ** 2 + b * b))
    major = 4.0 * math.sqrt(max(0.0, (a + c) / 2 + root))
    minor = 4.0 * math.sqrt(max(0.0, (a + c) / 2 - root))

    # convex hull as in skimage: over the pixel edges (diamond offsets), a pixel belongs to the
    # convex image if its centre lies inside; computed row by row (first / last pixel per row)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rows, first, last = _convex_rows(_hull(np.concatenate(contours).reshape(-1, 2)))
    solidity = area / max(int((last - first + 1).sum()), 1)

    # Feret: largest distance across the convex image, measured at its pixel edges
    outline = _hull(np.concatenate([np.stack([first, rows], 1), np.stack([last, rows], 1)]))
    difference = outline[:, None, :] - outline[None, :, :]
    feret = float(np.sqrt((difference ** 2).sum(-1)).max())
    return area, major, minor, solidity, feret


_DIAMOND = np.array([[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]])


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull (vertices, in order) of the pixel centres `points` widened by half a pixel."""
    widened = (points[:, None, :] + _DIAMOND[None]).reshape(-1, 2).astype(np.float32)
    return cv2.convexHull(widened).reshape(-1, 2).astype(np.float64)


def _convex_rows(hull: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels whose centre lies inside the convex polygon `hull`: per row y the first and last x."""
    start, end = hull, np.roll(hull, -1, axis=0)
    rows = np.arange(math.ceil(hull[:, 1].min() - eps), math.floor(hull[:, 1].max() + eps) + 1)
    y0, y1 = start[None, :, 1], end[None, :, 1]
    y = rows[:, None].astype(np.float64)
    on_edge = (y >= np.minimum(y0, y1) - eps) & (y <= np.maximum(y0, y1) + eps)
    flat = np.abs(y1 - y0) < eps
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(flat, start[None, :, 0], start[None, :, 0] + (y - y0) / (y1 - y0) * (end[None, :, 0] - start[None, :, 0]))
    x_low = np.where(on_edge, np.where(flat, np.minimum(start[None, :, 0], end[None, :, 0]), x), np.inf).min(1)
    x_high = np.where(on_edge, np.where(flat, np.maximum(start[None, :, 0], end[None, :, 0]), x), -np.inf).max(1)
    first, last = np.ceil(x_low - eps).astype(np.int64), np.floor(x_high + eps).astype(np.int64)
    keep = last >= first
    return rows[keep], first[keep], last[keep]
