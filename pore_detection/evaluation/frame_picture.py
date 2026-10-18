"""Frame picture: the sample with its coordinate system (sample frame) and both dimensions.

Saved as frame_picture.jpg in the image folder – a picture for a rough look by a human, without
the viewer. Drawn as the "Sample frame" layer of the viewer, in its color: axes x / z with arrow
heads at the origin, the frame rectangle dashed, labels "outer" / "inner". In addition, length
and thickness are dimensioned outside the rectangle: in mm with a scale, else in px.

Without a sample frame (sample.frame: none) no picture is made; the image does not fail.
"""

import cv2
import numpy as np

from ..core import imaging
from ..core.registry import register
from ..core.step import Evaluator
from ..core.types import EvaluationResult

COLOR = (240, 140, 0)           # RGB, as the sample frame in the viewer (gui/viewer/style.py FRAME)
OUTLINE = (0, 0, 0)             # dark rim around texts, readable on light and dark ground
MAX_PX = 4000                   # long edge of the saved picture in px
MARGIN = 0.12                   # free space around the sample for the dimensions, share of its long side
ARROW = 0.25                    # axis arrows, share of the smaller side (as in the viewer)
DIMENSION_GAP = 0.05            # dimension line distance from the rectangle, share of its long side
LINE_SHARE = 0.0012             # line width, share of the picture's long edge
TEXT_SHARE = 0.0007             # cv2 font scale per px of the picture's long edge
DASH_PX = (14, 9)               # dash and gap of the rectangle in picture px (at LINE_SHARE 1)


@register("frame_picture")
class FramePicture(Evaluator):
    """Sample with frame (axes, rectangle) and dimensioned length / thickness -> frame_picture.jpg."""

    params = {}

    def evaluate(self, ctx, pores):
        frame = ctx.frame
        if frame is None:
            return EvaluationResult(per_image={}, per_pore=None, images={})
        length, thickness = frame.length_px, frame.thickness_px
        margin = round(MARGIN * max(length, thickness))
        top, left = ctx.info.offset_px

        original = ctx.original
        if original.dtype != np.uint8:              # JPG: 8 bit only
            original = (original >> 8).astype(np.uint8)
        padded = cv2.copyMakeBorder(original, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        scale = min(1.0, MAX_PX / max(padded.shape[:2]))
        picture = imaging.downscale(padded, 1 / scale) if scale < 1 else padded.copy()
        scale = picture.shape[1] / padded.shape[1]

        (xy, xx), (zy, zx) = frame.x_axis, frame.z_axis
        oy, ox = frame.origin_px

        def point(x: float, z: float) -> tuple[float, float]:
            """Sample frame (px) -> picture px (x, y)."""
            return (scale * (ox + x * xx + z * zx - left + margin), scale * (oy + x * xy + z * zy - top + margin))

        width = max(1, round(LINE_SHARE * max(picture.shape[:2])))
        font = TEXT_SHARE * max(picture.shape[:2])
        draw = _Drawing(picture, width, font)

        corners = [point(0, 0), point(length, 0), point(length, thickness), point(0, thickness)]
        for i in range(4):
            draw.dashed(corners[i], corners[(i + 1) % 4])
        arrow = ARROW * min(length, thickness)
        draw.arrow(point(0, 0), point(arrow, 0))
        draw.arrow(point(0, 0), point(0, arrow))
        draw.text(point(arrow * 1.15, 0), "x", "left")
        draw.text(point(0, arrow * 1.15), "z", "above")
        draw.text(point(length / 2, thickness), "outer", "above")
        draw.text(point(length / 2, 0), "inner", "above")

        # dimensions: length below the inner edge, thickness right of the right edge
        gap = DIMENSION_GAP * max(length, thickness)
        draw.dimension(point(0, 0), point(length, 0), point(0, -gap), point(length, -gap),
                       _length_text(length, ctx.pixel_size_um), "below")
        draw.dimension(point(length, 0), point(length, thickness), point(length + gap, 0),
                       point(length + gap, thickness), _length_text(thickness, ctx.pixel_size_um), "right")
        return EvaluationResult(per_image={}, per_pore=None, images={"frame_picture.jpg": picture})


class _Drawing:
    """Lines, arrows and texts in COLOR on an RGB uint8 picture (positions in picture px, float)."""

    def __init__(self, picture: np.ndarray, width: int, font: float):
        self.picture, self.width, self.font = picture, width, font

    def line(self, a, b) -> None:
        cv2.line(self.picture, _px(a), _px(b), COLOR, self.width, cv2.LINE_AA)

    def dashed(self, a, b) -> None:
        a, b = np.array(a), np.array(b)
        total = np.linalg.norm(b - a)
        dash, space = DASH_PX[0] * self.width, DASH_PX[1] * self.width
        start = 0.0
        while start < total:
            stop = min(start + dash, total)
            self.line(a + (b - a) * start / total, a + (b - a) * stop / total)
            start = stop + space

    def arrow(self, a, b, head: float = 0.18) -> None:
        """Line a -> b with an arrow head at b (head: share of the line, as in the viewer)."""
        a, b = np.array(a), np.array(b)
        direction = b - a
        normal = np.array([-direction[1], direction[0]])
        self.line(a, b)
        self.line(b, b - head * direction + head / 2 * normal)
        self.line(b, b - head * direction - head / 2 * normal)

    def dimension(self, a, b, a_out, b_out, text: str, side: str) -> None:
        """Dimension of a -> b: extension lines to the dimension line a_out -> b_out, arrows at both
        ends, text at its middle."""
        a, b, a_out, b_out = (np.array(p) for p in (a, b, a_out, b_out))
        out = (a_out - a) / max(np.linalg.norm(a_out - a), 1e-9)
        overshoot = 4 * self.width * out
        self.line(a + 2 * self.width * out, a_out + overshoot)
        self.line(b + 2 * self.width * out, b_out + overshoot)
        half = np.linalg.norm(b_out - a_out) / 2
        head = min(0.5, 12 * self.width / max(half, 1e-9))     # share of the half line
        middle = (a_out + b_out) / 2
        self.arrow(middle, a_out, head)
        self.arrow(middle, b_out, head)
        self.text(tuple(middle + 3 * self.width * out), text, side)

    def text(self, anchor, text: str, side: str) -> None:
        """Text next to anchor: side = above | below | left | right of it (left = text starts there)."""
        face, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(self.width * 1.2))
        (w, h), _ = cv2.getTextSize(text, face, self.font, thickness)
        x, y = anchor
        x, y = {"above": (x - w / 2, y - h / 2), "below": (x - w / 2, y + 1.5 * h),
                "left": (x, y + h / 2), "right": (x, y + h / 2)}[side]
        position = _px((x, y))
        cv2.putText(self.picture, text, position, face, self.font, OUTLINE, thickness + 2 * self.width, cv2.LINE_AA)
        cv2.putText(self.picture, text, position, face, self.font, COLOR, thickness, cv2.LINE_AA)


def _length_text(length_px: float, pixel_size_um: float | None) -> str:
    """With a scale in mm (the cv2 font has no µ), else in px."""
    if pixel_size_um is None:
        return f"{length_px:.0f} px"
    mm = length_px * pixel_size_um / 1000
    return f"{mm:.2f} mm" if mm >= 1 else f"{mm:.3f} mm"


def _px(point) -> tuple[int, int]:
    return round(float(point[0])), round(float(point[1]))
