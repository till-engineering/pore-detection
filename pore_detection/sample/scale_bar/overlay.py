"""Find burnt-in scale overlays: a black bar with a text label in a pure white box (ImageJ style).

Ported from Pore_Detection_V2 (poren/massstab/kasten.py, geteilter_kasten.py), with one change
for speed: V2 filled the holes of the white areas in the whole image (12 s of 14 s on a 99 MP
panorama). Here this happens only in a window around each bar candidate, as large as the box is
allowed to be – the result is the same, the cost no longer depends on the image size.

The search starts from the bar, not from the box: a solid, wide, horizontal black block with
white above and below does not occur in a micrograph by chance. From the bar the box is found,
because the box is pure white (255) and microstructure practically never is:

- regular case: the bar is a hole in the white box; with holes filled the box is a solid
  rectangle, and the white component containing the bar is the box
- split case: the bar runs from box wall to box wall and cuts the box in two halves; the box is
  put together from the white areas directly above and below it (both carry its full width
  and are flush)
"""

from dataclasses import dataclass

import cv2
import numpy as np

# bar
MIN_BAR_PX = 10                 # shortest bar
MAX_BAR_HEIGHT_PX = 40          # highest bar
MAX_BAR_HEIGHT_FRAC = 0.05      # highest bar as share of the image height
MIN_BAR_ASPECT = 3.0            # bar at least this much wider than high
MIN_BAR_FILL = 0.85             # filled share of its bounding box (solid, not a scratch)
ENV_WHITE_MIN = 250             # gray level from which the bar's surrounding counts as white
MIN_ENV_WHITE = 0.5             # share of white directly above and below the bar
MAX_CANDIDATES = 8              # bar candidates checked per image
# box
MAX_BOX_WIDTH_FACTOR = 8.0      # box at most this much wider than the bar
MAX_BOX_HEIGHT_FACTOR = 40.0    # box at most this much higher than the bar
MAX_BOX_AREA_FRAC = 0.35        # box at most this share of the image
MIN_BOX_RECTANGULARITY = 0.70   # filled share of the box's bounding rectangle
SPLIT_PROBE_PX = 4              # split case: rows above / below the bar searched for the halves
MIN_SPLIT_COVER = 0.9           # split case: each half covers this share of the bar width
MAX_SPLIT_OFFSET = 0.15         # split case: max. sideways offset of the halves (share of bar width)
SPLIT_PENALTY = 0.9             # split case: weaker shape argument, lower score
# label
MIN_LABEL_HEIGHT_PX = 3         # smallest character height


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in pixels."""

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_corners(cls, x0: int, y0: int, x1: int, y1: int) -> "Box":
        return cls(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.y2), slice(self.x, self.x2)

    def shifted(self, dx: int, dy: int) -> "Box":
        return Box(self.x + dx, self.y + dy, self.w, self.h)

    def union(self, other: "Box") -> "Box":
        return Box.from_corners(min(self.x, other.x), min(self.y, other.y),
                                max(self.x2, other.x2), max(self.y2, other.y2))

    def padded(self, pad: int, width: int, height: int) -> "Box":
        """Grown by `pad` on every side (negative = shrunk), clipped to the image."""
        return Box.from_corners(max(0, self.x - pad), max(0, self.y - pad),
                                min(width, self.x2 + pad), min(height, self.y2 + pad))

    def contains(self, other: "Box") -> bool:
        return self.x <= other.x and self.y <= other.y and self.x2 >= other.x2 and self.y2 >= other.y2


@dataclass(frozen=True)
class Candidate:
    """A scale overlay found by its shape, not read yet."""

    bar: Box                    # reference length = bar width
    box: Box                    # white box around bar and label
    label: Box | None           # text line, None = no text found
    score: float                # 0..1, how much it looks like an overlay (shape only)
    notes: tuple[str, ...]      # doubts, logged if this candidate gives the scale


def find_candidates(gray: np.ndarray, ink_max: int, white_levels: tuple[int, ...]) -> list[Candidate]:
    """All overlay candidates of an 8-bit gray image, best first.

    `white_levels`: gray levels from which the box counts as white, strictest first; the next
    one is only tried for a bar whose box was not found with the previous one (e.g. JPEG).
    """
    candidates = []
    for bar, bar_score, bar_notes in _bars(gray, ink_max):
        for white_min in white_levels:
            found = _box_for_bar(gray, bar, white_min)
            if found is None:
                continue
            box, notes, penalty = found
            label = _find_label(gray, box, bar, ink_max)
            notes = bar_notes + notes
            if white_min != white_levels[0]:
                notes += (f"box only found with lowered white level {white_min}",)
            if label is None:
                notes += ("no label in the box",)
            candidates.append(Candidate(bar, box, label, penalty * _score(bar, box, label, bar_score), notes))
            break
    candidates.sort(key=lambda c: -c.score)
    return candidates


# --- step 1: the bar ---

def _bars(gray: np.ndarray, ink_max: int) -> list[tuple[Box, float, tuple[str, ...]]]:
    """Solid, wide, horizontal dark blocks with white above and below; best first."""
    height, width = gray.shape
    ink = (gray <= ink_max).view(np.uint8)
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    x, y, w, h, area = (stats[1:, i] for i in range(5))   # row 0 = background

    # shape filter vectorised: a micrograph has tens of thousands of dark pores
    max_bar_h = min(MAX_BAR_HEIGHT_PX, max(1, int(MAX_BAR_HEIGHT_FRAC * height)))
    fill = area / np.maximum(w * h, 1)
    keep = ((w >= MIN_BAR_PX) & (h <= max_bar_h) & (w >= MIN_BAR_ASPECT * h) & (fill >= MIN_BAR_FILL)
            & (w < width) & (h < height))

    found = []
    for i in np.flatnonzero(keep):
        bar = Box(int(x[i]), int(y[i]), int(w[i]), int(h[i]))
        white = _environment_white(gray, bar)
        if white < MIN_ENV_WHITE:
            continue
        # solid (fill), very wide (aspect) and surrounded by white
        score = 0.5 * white + 0.3 * float(fill[i]) + 0.2 * min(1.0, bar.w / bar.h / 10.0)
        notes = (f"only {white:.0%} white around the bar",) if white < 0.75 else ()
        found.append((bar, score, notes))
    found.sort(key=lambda item: (-item[1], -item[0].w))
    return found[:MAX_CANDIDATES]


def _environment_white(gray: np.ndarray, bar: Box) -> float:
    """Share of white directly above and below the bar."""
    pad = max(2, bar.h)
    above = gray[max(0, bar.y - pad):bar.y, bar.x:bar.x2]
    below = gray[bar.y2:bar.y2 + pad, bar.x:bar.x2]
    if above.size == 0 or below.size == 0:
        return 0.0
    return float(((above >= ENV_WHITE_MIN).sum() + (below >= ENV_WHITE_MIN).sum()) / (above.size + below.size))


# --- step 2: the box ---

def _box_for_bar(gray: np.ndarray, bar: Box, white_min: int) -> tuple[Box, tuple[str, ...], float] | None:
    """(box, notes, score factor) for one bar, or None. Works in a window around the bar."""
    height, width = gray.shape
    max_w = min(width, max(bar.w + 4, int(MAX_BOX_WIDTH_FACTOR * bar.w)))
    max_h = min(height, max(bar.h + 4, int(MAX_BOX_HEIGHT_FACTOR * max(bar.h, 1))))
    # every allowed box containing the bar lies inside; +1 px, so touching the window border
    # means: larger than allowed
    window = Box.from_corners(max(0, bar.x2 - max_w - 1), max(0, bar.y2 - max_h - 1),
                              min(width, bar.x + max_w + 1), min(height, bar.y + max_h + 1))
    labels = _filled_white_components(gray[window.slices()], white_min)
    local_bar = bar.shifted(-window.x, -window.y)

    if labels[local_bar.slices()].any():
        found = _box_around_hole(labels, local_bar, max_w, max_h)
        penalty = 1.0
    else:
        found = _box_from_halves(labels, local_bar, max_w, max_h)
        penalty = SPLIT_PENALTY
    if found is None:
        return None
    box, notes = found
    box = box.shifted(window.x, window.y)
    if box.area > MAX_BOX_AREA_FRAC * gray.size or box.w <= bar.w or not box.contains(bar):
        return None             # the box must enclose the bar, not be it
    return box, notes, penalty


def _filled_white_components(gray: np.ndarray, white_min: int) -> np.ndarray:
    """Label image (int32) of the white areas with their holes (bar, text) filled, 4-connected."""
    white = gray >= white_min
    # background reachable from the border via a 1 px frame is outside, everything else filled
    outside = cv2.copyMakeBorder((~white).view(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(outside, None, (0, 0), 2, flags=4)
    filled = (outside[1:-1, 1:-1] != 2).view(np.uint8)
    _count, labels = cv2.connectedComponents(filled, connectivity=4)
    return labels


def _component_box(labels: np.ndarray, component: int) -> tuple[Box, int]:
    """Bounding box and pixel count of one component."""
    ys, xs = np.nonzero(labels == component)
    return Box.from_corners(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), len(xs)


def _box_around_hole(labels: np.ndarray, bar: Box, max_w: int, max_h: int) -> tuple[Box, tuple[str, ...]] | None:
    """Regular case: the filled white component the bar lies in."""
    ids, counts = np.unique(labels[bar.slices()], return_counts=True)
    counts[ids == 0] = 0
    box, pixels = _component_box(labels, int(ids[np.argmax(counts)]))

    rectangularity = pixels / max(box.area, 1)
    if rectangularity < MIN_BOX_RECTANGULARITY:
        return None
    notes = ()
    if rectangularity < 0.95:
        notes += (f"box only {rectangularity:.0%} rectangular, probably merged with white image areas",)

    # box ran into equally white image areas: limit it to a plausible window around the bar
    if box.w > max_w or box.h > max_h:
        height, width = labels.shape
        cx, cy = bar.center
        x0 = int(max(0, min(bar.x, cx - max_w / 2)))
        y0 = int(max(0, min(bar.y, cy - max_h / 2)))
        box = Box.from_corners(x0, y0, min(width, max(bar.x2, x0 + max_w)), min(height, max(bar.y2, y0 + max_h)))
        notes += ("box limited to a window around the bar",)
    return box, notes


def _box_from_halves(labels: np.ndarray, bar: Box, max_w: int, max_h: int) -> tuple[Box, tuple[str, ...]] | None:
    """Split case: box put together from the white areas above and below the bar."""
    height = labels.shape[0]
    upper = _half(labels, bar, max(0, bar.y - SPLIT_PROBE_PX), bar.y)
    lower = _half(labels, bar, bar.y2, min(height, bar.y2 + SPLIT_PROBE_PX))
    if upper is None or lower is None:
        return None
    (upper_id, upper_box), (lower_id, lower_box) = upper, lower

    # both halves carry the bar over its full width …
    for half in (upper_box, lower_box):
        if min(half.x2, bar.x2) - max(half.x, bar.x) < MIN_SPLIT_COVER * bar.w:
            return None
    # … and are flush: two random bright spots framing the bar do not end flush with it
    tolerance = max(2.0, MAX_SPLIT_OFFSET * bar.w)
    if abs(upper_box.x - lower_box.x) > tolerance or abs(upper_box.x2 - lower_box.x2) > tolerance:
        return None

    box = upper_box.union(lower_box).union(bar)
    occupied = int(np.isin(labels[box.slices()], [upper_id, lower_id]).sum()) + bar.area
    rectangularity = occupied / max(box.area, 1)
    # here the box is hardly wider than the bar; if it is too large, the halves are no box
    if rectangularity < MIN_BOX_RECTANGULARITY or box.w > max_w or box.h > max_h:
        return None
    notes = ("bar reaches the box walls, box put together from the areas above and below",)
    if rectangularity < 0.95:
        notes += (f"box only {rectangularity:.0%} rectangular",)
    return box, notes


def _half(labels: np.ndarray, bar: Box, y0: int, y1: int) -> tuple[int, Box] | None:
    """Dominant white component in a band of rows over the bar width (a band, not one row:
    an anti-aliased transition row often lies between bar and box white)."""
    if y1 <= y0:
        return None
    ids, counts = np.unique(labels[y0:y1, bar.x:bar.x2], return_counts=True)
    counts[ids == 0] = 0
    if not counts.any():
        return None
    component = int(ids[np.argmax(counts)])
    return component, _component_box(labels, component)[0]


# --- step 3: the label ---

def _find_label(gray: np.ndarray, box: Box, bar: Box, ink_max: int) -> Box | None:
    """Text line inside the box, bar excluded. Strictly inside the box: no microstructure."""
    ink = (gray[box.slices()] <= ink_max).view(np.uint8).copy()
    local = bar.shifted(-box.x, -box.y)
    ink[local.slices()] = 0                    # the bar is the reference length, not text

    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    bar_center_y = bar.center[1]
    sides: dict[str, list[tuple[Box, int]]] = {"above": [], "below": []}
    for x, y, w, h, area in stats[1:]:
        if h < MIN_LABEL_HEIGHT_PX or h >= box.h or w >= box.w:
            continue
        part = Box(box.x + int(x), box.y + int(y), int(w), int(h))
        sides["above" if part.center[1] < bar_center_y else "below"].append((part, int(area)))

    parts = max(sides.values(), key=lambda items: sum(area for _, area in items))
    if not parts:
        return None
    # only the line of the largest character; otherwise bright structure inside an oversized
    # box ends up in the same bounding box as the text
    seed = max(parts, key=lambda item: item[1])[0]
    top, bottom = seed.y - 0.6 * seed.h, seed.y2 + 0.6 * seed.h
    label = seed
    for part, _area in parts:
        if top <= part.center[1] <= bottom:
            label = label.union(part)
    return label


def _score(bar: Box, box: Box, label: Box | None, bar_score: float) -> float:
    """How much this looks like a scale overlay, from the shape alone."""
    score = bar_score
    if label is not None:
        score += 0.25
        offset = abs(label.center[0] - bar.center[0]) / max(bar.w, 1)   # label centred over the bar
        score += 0.1 * max(0.0, 1.0 - offset)
    tightness = bar.area / max(box.area, 1)                             # box tight around its content
    score += 0.05 * min(1.0, tightness * 4.0)
    return float(min(1.0, score))
