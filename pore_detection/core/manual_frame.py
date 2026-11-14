"""Sample frame set by hand in the viewer (PLAN.md 7.11): five clicks, always a right angle.

Clicks in order, (x, y) in pixels of the full image:
    1, 2  two points on one long edge (top or bottom) -> line, direction of x
    3     a point on the other long edge -> parallel line, thickness
    4, 5  a point on each short edge (left / right in any order) -> perpendicular lines, length

As in edge_lines: the upper long edge (as seen) is the outer edge, x points to the right along it,
z upwards; origin = inner (bottom) edge meets the left edge. Rounded corners do not matter, only
lines are clicked. There is no own line of the inner edge, so the wedge angle is NaN.
(frame_from_clicks is the same in the Fatigue_crack_labeler, core/manual_frame.py – keep both in step.)
"""

import math

import numpy as np

from .types import SampleFrame

CLICKS = 5
MIN_SIZE_PX = 2.0               # shorter line, thickness or length: the clicks are not accepted


def frame_from_clicks(points) -> SampleFrame:
    """SampleFrame from the five clicks; ValueError with a short reason if they give no rectangle."""
    points = [(float(x), float(y)) for x, y in points]
    if len(points) != CLICKS:
        raise ValueError(f"a sample frame needs {CLICKS} clicks, got {len(points)}")
    (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5) = points
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm < MIN_SIZE_PX:
        raise ValueError("the first two clicks are too close – click two points along the edge")
    if dx < 0 or (dx == 0 and dy > 0):             # x points to the right (vertical: upwards)
        dx, dy = -dx, -dy
    x_axis = (dy / norm, dx / norm)                 # (dy, dx)
    z_axis = (-x_axis[1], x_axis[0])                # x turned by 90° upwards (as seen)

    # third click: the other long edge; the upper one of both is the outer edge
    height = (y3 - y1) * z_axis[0] + (x3 - x1) * z_axis[1]
    thickness = abs(height)
    if thickness < MIN_SIZE_PX:
        raise ValueError("the third click lies on the first line – click on the other long edge")
    top = (y1, x1) if height < 0 else (y1 + height * z_axis[0], x1 + height * z_axis[1])
    bottom = (top[0] - thickness * z_axis[0], top[1] - thickness * z_axis[1])

    # fourth and fifth click: left and right edge, perpendicular to the long edges
    along = sorted((y - bottom[0]) * x_axis[0] + (x - bottom[1]) * x_axis[1] for x, y in ((x4, y4), (x5, y5)))
    length = along[1] - along[0]
    if length < MIN_SIZE_PX:
        raise ValueError("the last two clicks give no length – click on the left and on the right edge")
    origin = (bottom[0] + along[0] * x_axis[0], bottom[1] + along[0] * x_axis[1])
    angle = math.degrees(math.atan2(-x_axis[0], x_axis[1]))     # counter-clockwise as seen
    return SampleFrame((float(origin[0]), float(origin[1])), float(angle), float(thickness), float(length),
                       float(np.nan))
