"""Parsen der Maßstabsbeschriftung.

Das ist die gefährlichste Stelle des Moduls: ein falsch gelesenes ``mm`` statt ``nm``
verfälscht jede spätere Messung um den Faktor eine Million - und zwar unauffällig, weil
das Ergebnis weiterhin plausibel aussieht.
"""

from __future__ import annotations

import pytest

from poredet.core.units import (
    UNIT_FACTORS_UM,
    format_um,
    normalize_unit,
    parse_length,
    parse_number,
)


class TestZahlen:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("151.7241", 151.7241),
            ("2.0", 2.0),
            ("1", 1.0),
            ("86,5", 86.5),          # Dezimalkomma
            ("1.234.567", 1234567.0),  # durchgehend Tausenderpunkte
            ("1.234,5", 1234.5),     # deutsche Schreibweise
            ("1,234.5", 1234.5),     # englische Schreibweise
        ],
    )
    def test_liest_zahlen(self, text: str, expected: float) -> None:
        assert parse_number(text) == pytest.approx(expected)

    def test_unlesbares_gibt_none(self) -> None:
        assert parse_number("") is None
        assert parse_number("abc") is None


class TestEinheiten:
    @pytest.mark.parametrize("raw", ["µm", "μm", "um", "UM", " µm ", "micron"])
    def test_mikrometer_in_allen_schreibweisen(self, raw: str) -> None:
        unit, corrections = normalize_unit(raw)
        assert unit == "um"
        assert corrections == []

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("pm", "um"), ("ym", "um"), ("yum", "um"), ("jm", "um"), ("lm", "um")],
    )
    def test_repariert_fehllesungen_des_mu(self, raw: str, expected: str) -> None:
        """Alle diese Formen entstehen aus dem µ-Zeichen - und werden protokolliert."""
        unit, corrections = normalize_unit(raw)
        assert unit == expected
        assert corrections, "eine Reparatur muss vermerkt werden"

    def test_pikometer_ist_kein_gueltiger_massstab(self) -> None:
        """Ein Balken in Pikometern kommt nicht vor, eine Fehllesung von µm dagegen oft."""
        assert "pm" not in UNIT_FACTORS_UM

    def test_unbekanntes_wird_nicht_geraten(self) -> None:
        unit, _ = normalize_unit("xyz")
        assert unit is None


class TestBeschriftung:
    @pytest.mark.parametrize(
        ("text", "value_um"),
        [
            ("151.7241 µm", 151.7241),
            ("2.0 mm", 2_000.0),
            ("1 µm", 1.0),
            ("1.035 m", 1_035_000.0),
            ("47 km", 47e9),
            ("20m", 20e6),
            ("86.5 mm", 86_500.0),
            ("500 nm", 0.5),
        ],
    )
    def test_liest_vollstaendige_angaben(self, text: str, value_um: float) -> None:
        parsed = parse_length(text)
        assert parsed is not None
        assert parsed.value_um == pytest.approx(value_um)

    def test_ohne_einheit_kein_ergebnis(self) -> None:
        """Eine Zahl allein ist kein Maßstab - hier wird nichts unterstellt."""
        assert parse_length("151.7241") is None

    def test_ohne_zahl_kein_ergebnis(self) -> None:
        assert parse_length("µm") is None

    def test_leerzeichen_in_der_zahl_erfindet_keinen_wert(self) -> None:
        """Aus "151 1.7241 µm" darf nie 1511.7241 werden - diese Zahl stand nie im Bild."""
        parsed = parse_length("151 1.7241 µm")
        assert parsed is None or parsed.value_um != pytest.approx(1511.7241)

    def test_reparatur_wird_vermerkt(self) -> None:
        parsed = parse_length("151.7241 pm")
        assert parsed is not None
        assert parsed.unit == "um"
        assert parsed.was_corrected


class TestFormatierung:
    @pytest.mark.parametrize(
        ("value_um", "contains"),
        [(0.5, "nm"), (200.0, "µm"), (2_000.0, "mm"), (1_035_000.0, "m")],
    )
    def test_waehlt_passende_einheit(self, value_um: float, contains: str) -> None:
        assert contains in format_um(value_um)
