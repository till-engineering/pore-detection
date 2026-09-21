"""Umrechnung px <-> physikalische Einheiten, Parsen von Angaben wie "2.0 mm".

Der Maßstabsbalken ist beschriftet, und diese Beschriftung kommt aus einer OCR. Sie ist
damit die unzuverlässigste Stelle der ganzen Kette: ein falsch gelesenes ``mm`` statt
``nm`` verfälscht jede spätere Messung um den Faktor eine Million, ohne dass es auffällt.

Deshalb zwei Regeln in diesem Modul:

1. **Geraten wird nicht.** Was sich nicht eindeutig auflösen lässt, liefert ``None``.
2. **Jede Korrektur wird protokolliert.** Bekannte OCR-Verwechslungen (``rn`` statt ``m``)
   werden behoben, aber die Korrektur steht in ``corrections`` und senkt später die
   Konfidenz. Ein Wert, der nur nach einer Reparatur zustande kam, ist weniger wert als
   einer, der direkt gelesen wurde.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Einheiten
# --------------------------------------------------------------------------------------

#: Umrechnungsfaktoren in Mikrometer. Mikrometer ist die interne Basiseinheit, weil sich
#: metallographische Gefügemerkmale darin ohne Exponenten ausdrücken lassen.
UNIT_FACTORS_UM: dict[str, float] = {
    # Pikometer fehlt hier bewusst. Es ist eine gültige SI-Einheit, aber auf einem
    # Maßstabsbalken kommt sie nicht vor - ein Pixel wäre tausendfach kleiner als ein
    # Atom. Dafür ist "pm" eine der häufigsten Fehllesungen von "µm". Die Einheit
    # wegzulassen heißt: "pm" fällt in die Verwechslungstabelle und wird zu "µm"
    # repariert, statt eine Messung um den Faktor 10^6 zu verfälschen.
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

#: Zeichen, die je nach Schriftart und Encoding für "mikro" stehen.
_MICRO_CHARS = "µμρᵘ"  # MICRO SIGN, GREEK MU, rho (Fehllesung), hoch-u

#: Bekannte OCR-Verwechslungen, nur als zweiter Versuch angewandt.
#: Reihenfolge ist relevant: längere Muster zuerst.
#: Alles außer "rn"/"nun" sind Fehllesungen des µ-Zeichens: sein Abstrich nach unten
#: macht es zu p, y, j oder 1, sein Bogen zu u, v oder i.
_CONFUSIONS: tuple[tuple[str, str], ...] = (
    ("rn", "m"),      # "rn" wird in serifenlosen Schriften als "m" gesetzt
    ("nun", "mm"),
    ("yum", "um"),
    ("jum", "um"),
    ("pum", "um"),
    ("pm", "um"),
    ("jm", "um"),
    ("im", "um"),
    ("lm", "um"),
    ("vm", "um"),
    ("ym", "um"),
)
# Ziffernartige Fehllesungen ("1m", "0m") stehen bewusst nicht hier. Die Einheit wird aus
# einer Zeichengruppe **ohne Ziffern** gewonnen - sie können also gar nicht ankommen. Aus
# "1m" wird "1" plus "m", und das ist ein Meter. Eine Regel, die daraus µm machte, wäre
# nicht erreichbar und im Einzelfall schlicht falsch.


def normalize_unit(raw: str) -> tuple[str | None, list[str]]:
    """Einheitentext -> kanonischer Schlüssel aus :data:`UNIT_FACTORS_UM`.

    Liefert ``(None, [...])``, wenn sich die Einheit nicht eindeutig zuordnen lässt -
    lieber kein Maßstab als ein falscher. Die zweite Rückgabe listet die angewandten
    Korrekturen; sie ist leer, wenn direkt gelesen werden konnte.
    """
    corrections: list[str] = []
    s = raw.strip().strip(".,:;()[]{}")
    for ch in _MICRO_CHARS:
        s = s.replace(ch, "u")
    s = s.replace("Â", "")  # Mojibake-Rest aus falsch dekodiertem UTF-8
    s = s.lower()
    s = re.sub(r"[^a-z]", "", s)

    if not s:
        return None, corrections
    if s in UNIT_FACTORS_UM:
        return s, corrections

    # Zweiter Versuch: bekannte Verwechslungen beheben.
    fixed = s
    for wrong, right in _CONFUSIONS:
        if wrong in fixed:
            fixed = fixed.replace(wrong, right)
    if fixed != s and fixed in UNIT_FACTORS_UM:
        corrections.append(f"Einheit {s!r} als {fixed!r} gelesen")
        return fixed, corrections

    # "micron"/"microns" als ausgeschriebene Form.
    if s in {"micron", "microns", "micrometer", "micrometre", "mikrometer"}:
        return "um", corrections
    if s in {"nanometer", "nanometre"}:
        return "nm", corrections
    if s in {"millimeter", "millimetre"}:
        return "mm", corrections

    return None, corrections


# --------------------------------------------------------------------------------------
# Zahlen
# --------------------------------------------------------------------------------------

def parse_number(raw: str) -> float | None:
    """Zahltext -> float, mit Dezimalkomma und Tausendertrennung.

    Liegen Punkt und Komma gemeinsam vor, ist das *letzte* der beiden das
    Dezimaltrennzeichen. Steht nur ein Komma, gilt es als Dezimaltrennzeichen - ImageJ
    schreibt keine Tausenderpunkte, und "86,5" als 865 zu lesen wäre der schlimmere
    Fehler.
    """
    s = re.sub(r"[\s']", "", raw)
    if not s:
        return None

    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thousands = "." if dec == "," else ","
        s = s.replace(thousands, "").replace(dec, ".")
    elif "," in s:
        s = s.replace(",", ".", 1) if s.count(",") == 1 else s.replace(",", "")
    elif s.count(".") > 1:
        # Mehrere Punkte. Sind alle Gruppen dahinter dreistellig, ist durchgehend
        # Tausendertrennung gemeint ("1.234.567"); sonst ist der letzte Punkt das
        # Dezimaltrennzeichen ("1.234.5").
        groups = s.split(".")
        if all(len(g) == 3 for g in groups[1:]):
            s = "".join(groups)
        else:
            s = "".join(groups[:-1]) + "." + groups[-1]

    try:
        value = float(s)
    except ValueError:
        return None
    return value


# --------------------------------------------------------------------------------------
# Gesamtangabe
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedLength:
    """Eine gelesene Längenangabe, zerlegt und nach Mikrometer umgerechnet."""

    value: float
    unit: str
    value_um: float
    raw: str
    corrections: tuple[str, ...] = field(default_factory=tuple)

    @property
    def was_corrected(self) -> bool:
        return bool(self.corrections)

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}"


#: Die Zahl darf **kein** Leerzeichen enthalten. Das ist bewusst streng: liest die OCR
#: aus "151.7241 µm" versehentlich "151 1.7241 µm", würde eine tolerante Regel daraus
#: 1511.7241 zusammensetzen - eine Zahl, die nie im Bild stand. Lieber ein Teiltreffer,
#: der in der Abstimmung überstimmt wird, als ein erfundener Wert.
_LABEL_RE = re.compile(
    r"(?P<num>[0-9]+(?:[.,][0-9]+)*)\s*(?P<unit>[^\s0-9]{1,12})",
)


def parse_length(text: str) -> ParsedLength | None:
    """Beschriftung wie ``"151.7241 µm"`` -> :class:`ParsedLength`.

    Gibt ``None`` zurück, wenn Zahl oder Einheit fehlen oder die Einheit unbekannt ist.
    Von mehreren Treffern im Text gewinnt der erste vollständig auflösbare.
    """
    if not text:
        return None

    cleaned = text.replace("\n", " ").strip()
    for match in _LABEL_RE.finditer(cleaned):
        value = parse_number(match.group("num"))
        if value is None or value <= 0:
            continue
        unit, corrections = normalize_unit(match.group("unit"))
        if unit is None:
            continue
        return ParsedLength(
            value=value,
            unit=unit,
            value_um=value * UNIT_FACTORS_UM[unit],
            raw=match.group(0).strip(),
            corrections=tuple(corrections),
        )
    return None


# --------------------------------------------------------------------------------------
# Formatierung
# --------------------------------------------------------------------------------------

#: Für die Ausgabe: absteigende Schwellen in Mikrometer.
_DISPLAY_UNITS: tuple[tuple[float, str], ...] = (
    (1e9, "km"),
    (1e6, "m"),
    (1e3, "mm"),
    (1.0, "um"),
    (1e-3, "nm"),
)

#: Für die Anzeige gegenüber Menschen - im Dateisystem und in CSV bleibt es bei "um".
UNIT_SYMBOLS = {"um": "µm"}


def format_um(value_um: float, digits: int = 4) -> str:
    """Mikrometerwert menschenlesbar, in der jeweils passenden Einheit."""
    if value_um == 0:
        return "0 µm"
    magnitude = abs(value_um)
    factor, unit = next(
        ((f, u) for f, u in _DISPLAY_UNITS if magnitude >= f),
        (1e-3, "nm"),
    )
    shown = value_um / factor
    symbol = UNIT_SYMBOLS.get(unit, unit)
    return f"{shown:.{digits}g} {symbol}"
