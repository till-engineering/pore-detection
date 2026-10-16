"""Read the label of a scale overlay: prepare the text line, OCR engines, majority vote.

Ported from Pore_Detection_V2 (poren/massstab/ocr.py, erkennung.py) with one change for speed:
RapidOCR runs its recognition model only (~10 ms instead of ~0.8 s per reading). The text line
is already known from the overlay geometry, so its text detection model is not needed – and
reading the known line directly was also more reliable on the V2 test images.

Patch preparation matters more than the engine:
- the patch stays inside the white box (no microstructure read as characters)
- the bar is painted white (a line over the text is read as a character: "47 km" -> "471 km")
- the text line is scaled to a fixed height, cut out tightly, with a small white quiet zone

Every engine reads every variant of the patch until one value has enough votes without any
dissent; the normal case costs three readings, only difficult labels cost more.
"""

import logging
import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np

from .overlay import Box, Candidate
from .units import Length, parse_length

log = logging.getLogger("pore_detection")

VARIANT_HEIGHTS_PX = (48, 40, 64)   # target heights of the text line, in reading order
BINARIZED_VARIANT = True            # additionally read an Otsu-binarised version of the first
MAX_UPSCALE = 8.0
# tight crop: the recognition model scales the patch to 48 px height; with a wide margin the text
# gets too small and repeated digits merge ("1000.7645" -> "100.7645" at 0.6 / 8 px)
PAD_FACTOR = 0.3                    # margin around the text line (x line height), inside the box
QUIET_ZONE_PX = 4                   # white frame around the patch
ENGINE_WEIGHTS = {"rapidocr": 1.0, "tesseract": 0.85}
MIN_CONSENSUS_WEIGHT = 2.5          # votes for one value without dissent: stop reading (~3 readings)


@dataclass(frozen=True)
class Reading:
    """One OCR reading of one patch variant."""

    length: Length
    text: str
    engine: str
    confidence: float

    @property
    def vote(self) -> float:
        """An uncertain reading counts less, but never nothing."""
        return ENGINE_WEIGHTS.get(self.engine, 0.7) * (0.5 + 0.5 * min(1.0, max(0.0, self.confidence)))


@dataclass(frozen=True)
class Vote:
    """Result of reading one label."""

    winner: Reading
    agreement: float            # share of all votes for the winning value
    dissent: str                # outvoted values, "" = none


def read_label(gray: np.ndarray, candidate: Candidate, engines: list) -> Vote | str:
    """Majority vote over all engines and patch variants; a reason string if nothing parses."""
    readings: list[Reading] = []
    texts: list[str] = []
    for patch in _patch_variants(gray, candidate):
        for engine in engines:
            text, confidence = engine.read(patch)
            texts.append(text)
            length = parse_length(text)
            if length is not None:
                readings.append(Reading(length, text, engine.name, confidence))
        if _consensus(readings):
            break
    if not readings:
        seen = ", ".join(sorted({t for t in texts if t})) or "nothing"
        return f"label not readable (read: {seen!r})"

    groups = sorted(_group(readings), key=lambda g: -sum(r.vote for r in g))
    total = sum(r.vote for r in readings)
    winner = max(groups[0], key=lambda r: r.confidence)
    dissent = "; ".join(f"{g[0].length} ({', '.join(sorted({r.engine for r in g}))})" for g in groups[1:])
    return Vote(winner, sum(r.vote for r in groups[0]) / total, dissent)


def _group(readings: list[Reading]) -> list[list[Reading]]:
    """Readings with the same value (0.1 % tolerance)."""
    groups: list[list[Reading]] = []
    for reading in readings:
        for group in groups:
            a, b = group[0].length.value_um, reading.length.value_um
            if abs(a - b) <= 0.001 * max(a, b):
                group.append(reading)
                break
        else:
            groups.append([reading])
    return groups


def _consensus(readings: list[Reading]) -> bool:
    """One value with enough votes and not a single dissenting reading."""
    groups = _group(readings)
    return len(groups) == 1 and sum(r.vote for r in groups[0]) >= MIN_CONSENSUS_WEIGHT


# --- patch preparation ---

def _patch_variants(gray: np.ndarray, candidate: Candidate):
    """The text line in several versions, in reading order (normal case first)."""
    first = None
    for height in VARIANT_HEIGHTS_PX:
        patch = _prepare_patch(gray, candidate, height)
        first = patch if first is None else first
        yield patch
    if BINARIZED_VARIANT:
        yield cv2.threshold(first, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def _prepare_patch(gray: np.ndarray, candidate: Candidate, text_height_px: int) -> np.ndarray:
    """Text line cut out inside the box, bar painted white, scaled, with a white quiet zone."""
    label, box, bar = candidate.label, candidate.box, candidate.bar
    pad = round(PAD_FACTOR * label.h)
    x0, y0 = max(box.x + 1, label.x - pad), max(box.y + 1, label.y - pad)     # 1 px off the box wall
    x1, y1 = min(box.x2 - 1, label.x2 + pad), min(box.y2 - 1, label.y2 + pad)
    if x1 <= x0 or y1 <= y0:
        return np.full((1, 1), 255, dtype=np.uint8)
    patch = gray[y0:y1, x0:x1].copy()
    # paint the bar white instead of cutting it off: the OCR needs the margin
    bx0, by0 = max(0, bar.x - x0 - 1), max(0, bar.y - y0 - 1)
    bx1, by1 = max(0, bar.x2 - x0 + 1), max(0, bar.y2 - y0 + 1)
    patch[by0:by1, bx0:bx1] = 255

    factor = min(MAX_UPSCALE, max(1.0, text_height_px / max(label.h, 1)))
    if factor > 1.0:
        patch = cv2.resize(patch, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(patch, QUIET_ZONE_PX, QUIET_ZONE_PX, QUIET_ZONE_PX, QUIET_ZONE_PX,
                              cv2.BORDER_CONSTANT, value=255)


# --- engines ---

class RapidOcr:
    """RapidOCR (rapidocr-onnxruntime), recognition model only. Model loads on first use."""

    name = "rapidocr"

    def __init__(self):
        self._recognizer = None

    def load(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        self._recognizer = RapidOCR().text_recognizer

    def read(self, patch: np.ndarray) -> tuple[str, float]:
        result, _elapsed = self._recognizer([cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)])
        text, confidence = result[0]
        return str(text).strip(), float(confidence)


class Tesseract:
    """Tesseract via pytesseract – a slower second opinion (~0.3 s per reading).

    Needs the Tesseract program (on PATH or in the environment variable TESSERACT_CMD). No
    character whitelist on purpose: with one, "151.7241 µm" was read as "nm" in V2.
    """

    name = "tesseract"

    def load(self) -> None:
        import pytesseract
        binary = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if not binary:
            raise RuntimeError("Tesseract program not found (PATH or TESSERACT_CMD)")
        pytesseract.pytesseract.tesseract_cmd = binary
        pytesseract.get_tesseract_version()

    def read(self, patch: np.ndarray) -> tuple[str, float]:
        import pytesseract
        for config in ("--psm 7", "--psm 6"):        # one line, else one block
            text = pytesseract.image_to_string(patch, config=config).strip()
            if text:
                return text, 0.6                     # no usable confidence without TSV output
        return "", 0.0


ENGINES = {"rapidocr": RapidOcr, "tesseract": Tesseract}
_loaded: dict[str, object] = {}


def load_engines(names) -> list:
    """The named engines in this order, each loaded once per process (the model is large)."""
    engines = []
    for name in names:
        if name not in ENGINES:
            raise ValueError(f"unknown OCR engine '{name}' (available: {', '.join(ENGINES)})")
        if name not in _loaded:
            engine = ENGINES[name]()
            engine.load()
            _loaded[name] = engine
        engines.append(_loaded[name])
    return engines
