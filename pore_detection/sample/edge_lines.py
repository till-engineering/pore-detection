"""Sample frame of a rectangular section from straight lines fitted to its edges.

The whole sample is in the image and lies roughly horizontal (tilt well below 45°). The top edge
is the outer surface, the bottom edge the inner one (PLAN.md 7.5). A sample cut by the image
border makes the image fail.

1. Outline of the sample (largest outer contour of the mask); every outline point gets its
   outward normal from the neighbouring points (+-normal_window_px along the outline).
2. Each point is assigned to the side its normal points to: top, bottom, left, right (first by
   the image axes, then once more by the axes of the fitted top edge).
3. Corners are left out: only the middle part of every side counts (corner_trim at both ends).
4. Top edge: robust line fit (Huber, cv2.fitLine) – it defines x and the outer surface exactly.
5. Bottom edge: parallel to the top edge, at the median distance of the bottom points – breakouts
   and the shrinkage gap do not shift it. Its own line fit only gives the wedge angle (check).
6. Left and right edge: robust line fits; origin = bottom edge meets left edge, length = distance
   to where the bottom edge meets the right edge.
"""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import FrameDetector
from ..core.types import SampleFrame

TOP, BOTTOM, LEFT, RIGHT = range(4)
SIDE_NAMES = ("top", "bottom", "left", "right")
MIN_POINTS = 10                 # fewer points on a side: no line can be fitted


@register("edge_lines")
class EdgeLines(FrameDetector):
    """Right-angled frame from lines fitted to the edges: x along the outer (top) edge, origin bottom left."""

    params = {
        "normal_window_px": "outline points before / after a point used for its edge direction",
        "corner_trim": "share of every side left out at both ends (rounded corners), e.g. 0.15",
    }

    def detect_frame(self, info, roi_mask):
        points, normals = self._outline(roi_mask)

        # sides first by the image axes, then by the axes of the fitted top edge
        side = _sides(normals, (0.0, 1.0))
        x_axis = _direction(_line(points[_trimmed(points, side, TOP, (0.0, 1.0), self.corner_trim)], "top"))
        side = _sides(normals, x_axis)
        top = _line(points[_trimmed(points, side, TOP, x_axis, self.corner_trim)], "top")
        x_axis = _direction(top)
        z_axis = (-x_axis[1], x_axis[0])            # (dy, dx): x turned by 90° upwards (as seen)

        # bottom edge: parallel to the top edge at the median distance of its points
        bottom_points = points[_trimmed(points, side, BOTTOM, x_axis, self.corner_trim)]
        if len(bottom_points) < MIN_POINTS:
            raise ValueError("too few points on the bottom edge")
        height = (bottom_points[:, 0] - top[0][0]) * z_axis[0] + (bottom_points[:, 1] - top[0][1]) * z_axis[1]
        thickness = -float(np.median(height))
        if thickness <= 0:
            raise ValueError("bottom edge not below the top edge")
        bottom_origin = (top[0][0] - thickness * z_axis[0], top[0][1] - thickness * z_axis[1])
        bottom = (bottom_origin, x_axis)
        wedge = _angle_between(x_axis, _direction(_line(bottom_points, "bottom")))

        left = _line(points[_trimmed(points, side, LEFT, z_axis, self.corner_trim)], "left")
        right = _line(points[_trimmed(points, side, RIGHT, z_axis, self.corner_trim)], "right")
        origin = _intersection(bottom, left)
        end = _intersection(bottom, right)
        length = (end[0] - origin[0]) * x_axis[0] + (end[1] - origin[1]) * x_axis[1]
        if length <= 0:
            raise ValueError("right edge not right of the left edge")

        angle = float(np.degrees(np.arctan2(-x_axis[0], x_axis[1])))   # counter-clockwise as seen
        return SampleFrame((float(origin[0]), float(origin[1])), angle, thickness, float(length), wedge)

    def _outline(self, roi_mask):
        """Outline points (y, x) of the largest sample part and their outward unit normals (dy, dx)."""
        contours, _ = cv2.findContours(roi_mask.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("no sample outline (roi_mask is empty)")
        contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)   # (x, y)
        if len(contour) < 4 * MIN_POINTS:
            raise ValueError("sample outline too short for a frame")
        height, width = roi_mask.shape
        on_border = ((contour[:, 0] == 0) | (contour[:, 0] == width - 1)
                     | (contour[:, 1] == 0) | (contour[:, 1] == height - 1))
        if on_border.sum() >= MIN_POINTS:
            raise ValueError("sample touches the image border – the frame needs the whole sample in the image")
        k = self.normal_window_px
        tangent = np.roll(contour, -k, axis=0) - np.roll(contour, k, axis=0)       # (dx, dy)
        normal_xy = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        normal_xy /= np.maximum(np.linalg.norm(normal_xy, axis=1, keepdims=True), 1e-9)
        if np.sum(normal_xy * (contour - contour.mean(axis=0))) < 0:   # turning direction of the outline
            normal_xy = -normal_xy                                      # -> normals point outwards
        points = contour[:, ::-1]                                                  # (y, x)
        normals = normal_xy[:, ::-1]                                               # (dy, dx)
        return points, normals


def _sides(normals: np.ndarray, x_axis) -> np.ndarray:
    """Side of every outline point: the frame axis its outward normal is closest to."""
    z_axis = (-x_axis[1], x_axis[0])
    along_x = normals @ np.asarray(x_axis)
    along_z = normals @ np.asarray(z_axis)
    return np.where(np.abs(along_z) >= np.abs(along_x),
                    np.where(along_z > 0, TOP, BOTTOM),
                    np.where(along_x > 0, RIGHT, LEFT))


def _trimmed(points: np.ndarray, side: np.ndarray, which: int, axis, trim: float) -> np.ndarray:
    """Points of one side without both ends (corners): position along `axis` inside the middle part."""
    selected = side == which
    if selected.sum() < MIN_POINTS:
        raise ValueError(f"too few points on the {SIDE_NAMES[which]} edge")
    position = points @ np.asarray(axis)
    low, high = np.quantile(position[selected], [trim, 1 - trim])
    return selected & (position >= low) & (position <= high)


def _line(points: np.ndarray, name: str):
    """Robust line fit: (point (y, x), unit direction (dy, dx))."""
    if len(points) < MIN_POINTS:
        raise ValueError(f"too few points on the {name} edge")
    vx, vy, x0, y0 = cv2.fitLine(points[:, ::-1].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return (float(y0), float(x0)), (float(vy), float(vx))


def _direction(line) -> tuple[float, float]:
    """Direction of a nearly horizontal line, pointing to the right (dx > 0)."""
    dy, dx = line[1]
    return (dy, dx) if dx >= 0 else (-dy, -dx)


def _angle_between(a, b) -> float:
    """Angle in degrees between two directions (lines: 0 … 90°)."""
    cos = abs(a[0] * b[0] + a[1] * b[1])
    return float(np.degrees(np.arccos(min(1.0, cos))))


def _intersection(line_a, line_b) -> tuple[float, float]:
    """Intersection (y, x) of two lines given as (point, direction)."""
    (ay, ax), (ady, adx) = line_a
    (by, bx), (bdy, bdx) = line_b
    det = adx * bdy - ady * bdx
    if abs(det) < 1e-9:
        raise ValueError("edges are parallel – no corner")
    t = ((bx - ax) * bdy - (by - ay) * bdx) / det
    return ay + t * ady, ax + t * adx
