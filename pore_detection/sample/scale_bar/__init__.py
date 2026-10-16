"""Scale from the burnt-in scale bar (black bar + label in a white box, ImageJ style).

Ported from the scale detection of Pore_Detection_V2 and made faster (99 MP panorama: 14 s in
V2). Files:
    overlay.py   find bar, box and text line from their shape
    reading.py   read the label: patch preparation, OCR engines, majority vote
    units.py     "151.7241 µm" -> micrometres

Rules: a scale only comes from a plausible shape AND a readable label; an unreadable label gives
no scale (None). There are no limits on the value: whatever is read is used, an unusual value
only logs a warning – the viewer shows bar and text for the optical check. Without a scale the
best overlay is still returned (mask, bar): the black bar is never a pore.

Self-contained (imports only core). CPU only; sample methods are not tiled.
"""

import logging

import cv2
import numpy as np

from ...core.registry import register
from ...core.step import ScaleDetector
from ...core.types import ScaleResult
from .overlay import find_candidates
from .reading import load_engines, read_label

log = logging.getLogger("pore_detection")


@register("scale_bar")
class ScaleBar(ScaleDetector):
    """Reads the burnt-in scale bar (bar length + OCR of its label); mask of the overlay."""

    params = {
        "ocr_engines": "OCR engines asked in this order: rapidocr (fast) and/or tesseract (slow, 2nd opinion)",
        "ink_max": "gray level 0-255 up to which a pixel is bar or text",
        "box_white_min": "gray level 0-255 from which a pixel is box white (strict, 255)",
        "box_white_fallback": "lowered box white level, tried if the strict one finds no box (JPEG)",
        "expected_min_um_per_px": "outside the expected range a warning is logged (the value is kept)",
        "expected_max_um_per_px": "upper end of the expected range",
        "exclusion_pad_px": "margin around the overlay box that is also excluded from the sample",
    }

    def __init__(self, **params):
        super().__init__(**params)
        self.engines = load_engines(self.ocr_engines)      # fails at start if an engine is missing

    def detect_scale(self, info, original):
        gray = _gray8(original)
        candidates = find_candidates(gray, self.ink_max, (self.box_white_min, self.box_white_fallback))
        if not candidates:
            log.info("%s: no scale found (no bar in a white box)", info.name)
            return ScaleResult(None)
        reasons = []
        for candidate in candidates:
            found = self._read(gray, candidate, info.name, reasons)
            if found is not None:
                return self._result(*found, candidate, gray.shape)
        log.info("%s: no scale – %s", info.name, "; ".join(reasons))
        return self._result(None, None, candidates[0], gray.shape)

    def _result(self, um_per_px, text, candidate, shape) -> ScaleResult:
        bar = candidate.bar
        return ScaleResult(um_per_px, self._mask(candidate.box, shape), (bar.y, bar.y2, bar.x, bar.x2), text)

    def _read(self, gray, candidate, name, reasons) -> tuple[float, str] | None:
        """(µm per px, text read), or None if the label is missing or not readable."""
        where = f"bar at x={candidate.bar.x}, y={candidate.bar.y}"
        if candidate.label is None:
            reasons.append(f"{where}: no label")
            return None
        vote = read_label(gray, candidate, self.engines)
        if isinstance(vote, str):
            reasons.append(f"{where}: {vote}")
            return None
        length, text = vote.winner.length, vote.winner.text
        um_per_px = length.value_um / candidate.bar.w

        doubts = list(candidate.notes)
        if length.correction:
            doubts.append(length.correction)
        if vote.dissent:
            doubts.append(f"outvoted readings: {vote.dissent} (agreement {vote.agreement:.0%})")
        if not self.expected_min_um_per_px <= um_per_px <= self.expected_max_um_per_px:
            doubts.append("outside the expected range – please check")
        log.info("%s: scale %s / %d px = %.6g µm/px (%s, read %r)", name, length, candidate.bar.w, um_per_px,
                 vote.winner.engine, vote.winner.text)
        for doubt in doubts:
            log.warning("%s: scale – %s", name, doubt)
        return um_per_px, text

    def _mask(self, box, shape) -> np.ndarray:
        height, width = shape
        mask = np.zeros(shape, dtype=bool)
        mask[box.padded(self.exclusion_pad_px, width, height).slices()] = True
        return mask


def _gray8(original: np.ndarray) -> np.ndarray:
    """8-bit gray image; the overlay is pure black / white, so the channel weights do not matter."""
    rgb8 = original if original.dtype == np.uint8 else (original >> 8).astype(np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
