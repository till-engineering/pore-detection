"""Parse scale labels like "151.7241 µm" into micrometres (ported from Pore_Detection_V2, einheiten.py).

The label comes from OCR and is the least reliable link of the chain: "mm" read instead of "nm"
falsifies every measurement by a factor of a million without anyone noticing. Therefore:
nothing is guessed (unclear -> None), and every repair of a known OCR confusion is reported.
"""

import re
from dataclasses import dataclass

UNIT_FACTORS_UM = {
    # no picometre on purpose: it never appears on a scale bar, but "pm" is one of the most
    # frequent misreadings of "µm" – so it is repaired to µm instead of being off by 10^6
    "angstrom": 1e-4,
    "nm": 1e-3,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "dm": 1e5,
    "m": 1e6,
    "km": 1e9,
    "mil": 25.4,
    "in": 25_400.0,
    "inch": 25_400.0,
}
_SPELLED = {"micron": "um", "microns": "um", "micrometer": "um", "micrometre": "um", "mikrometer": "um",
            "nanometer": "nm", "nanometre": "nm", "millimeter": "mm", "millimetre": "mm"}
_MICRO_CHARS = "µμρᵘ"          # micro sign, greek mu, rho (misreading), superscript u
# known OCR confusions, only as second attempt, longer patterns first; all except rn/nun are
# misreadings of µ (its descender makes it p, y, j or 1, its bow u, v or i)
_CONFUSIONS = (("rn", "m"), ("nun", "mm"), ("yum", "um"), ("jum", "um"), ("pum", "um"), ("pm", "um"),
               ("jm", "um"), ("im", "um"), ("lm", "um"), ("vm", "um"), ("ym", "um"))
# no space inside the number on purpose: "151 1.7241 µm" must not become 1511.7241
_LABEL_RE = re.compile(r"(?P<num>[0-9]+(?:[.,][0-9]+)*)\s*(?P<unit>[^\s0-9]{1,12})")


@dataclass(frozen=True)
class Length:
    """A parsed length label."""

    value: float
    unit: str                   # key of UNIT_FACTORS_UM
    value_um: float
    correction: str | None      # repaired OCR confusion, None = read directly

    def __str__(self) -> str:
        return f"{self.value:.10g} {self.unit}"


def parse_length(text: str) -> Length | None:
    """First complete "<number> <unit>" in the text, or None."""
    for match in _LABEL_RE.finditer(text.replace("\n", " ")):
        value = _parse_number(match.group("num"))
        if value is None or value <= 0:
            continue
        unit, correction = _normalize_unit(match.group("unit"))
        if unit is not None:
            return Length(value, unit, value * UNIT_FACTORS_UM[unit], correction)
    return None


def _normalize_unit(raw: str) -> tuple[str | None, str | None]:
    """Unit text -> (key of UNIT_FACTORS_UM or None, applied correction or None)."""
    text = raw.strip().strip(".,:;()[]{}")
    for char in _MICRO_CHARS:
        text = text.replace(char, "u")
    text = re.sub(r"[^a-z]", "", text.replace("Â", "").lower())   # Â: rest of mis-decoded UTF-8
    if text in UNIT_FACTORS_UM:
        return text, None
    if text in _SPELLED:
        return _SPELLED[text], None
    fixed = text
    for wrong, right in _CONFUSIONS:
        fixed = fixed.replace(wrong, right)
    if text and fixed != text and fixed in UNIT_FACTORS_UM:
        return fixed, f"unit {text!r} read as {fixed!r}"
    return None, None


def _parse_number(raw: str) -> float | None:
    """Number text -> float; decimal comma and thousands separators allowed.

    With both point and comma the last one is the decimal separator. A single comma is a decimal
    comma (ImageJ writes no thousands separators, and "86,5" read as 865 is the worse error).
    """
    text = re.sub(r"[\s']", "", raw)
    if "," in text and "." in text:
        decimal = "," if text.rfind(",") > text.rfind(".") else "."
        text = text.replace("." if decimal == "," else ",", "").replace(decimal, ".")
    elif "," in text:
        text = text.replace(",", ".") if text.count(",") == 1 else text.replace(",", "")
    elif text.count(".") > 1:
        groups = text.split(".")
        # all groups after the first with 3 digits: thousands ("1.234.567"), else last = decimal
        text = "".join(groups) if all(len(g) == 3 for g in groups[1:]) else "".join(groups[:-1]) + "." + groups[-1]
    try:
        return float(text)
    except ValueError:
        return None
