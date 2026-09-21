"""Die Schnittstelle, mit der alle späteren Module arbeiten.

``ScaleInfo`` ist der einzige Ort, an dem aus Pixeln Mikrometer werden. Wenn hier etwas
schief geht, ist jede Porengröße im Bericht falsch - und zwar konsistent falsch, was am
schwersten auffällt.
"""

from __future__ import annotations

import numpy as np
import pytest

from poredet.core.models import BBox, ScaleInfo, ScaleSource
from poredet.scale.exclusion import excluded_area_px, overlay_mask
from poredet.scale.manual import from_line, from_um_per_px


@pytest.fixture
def scale() -> ScaleInfo:
    return ScaleInfo.from_bar(
        value_um=200.0,
        bar_length_px=100.0,
        source=ScaleSource.OVERLAY_BAR,
        box=BBox(400, 300, 120, 30),
    )


class TestUmrechnung:
    def test_laenge(self, scale: ScaleInfo) -> None:
        assert scale.um_per_px == pytest.approx(2.0)
        assert scale.to_um(50) == pytest.approx(100.0)
        assert scale.to_mm(500) == pytest.approx(1.0)

    def test_flaeche_skaliert_quadratisch(self, scale: ScaleInfo) -> None:
        """Der häufigste Fehler überhaupt: den Faktor bei Flächen nur einmal anwenden."""
        assert scale.to_um2(100) == pytest.approx(400.0)
        assert scale.to_mm2(1_000_000) == pytest.approx(4.0)

    def test_rueckrichtung(self, scale: ScaleInfo) -> None:
        """Filtergrenzen werden physikalisch angegeben und müssen in Pixel zurück."""
        assert scale.to_px(100.0) == pytest.approx(50.0)
        assert scale.area_to_px(400.0) == pytest.approx(100.0)

    def test_hin_und_zurueck(self, scale: ScaleInfo) -> None:
        assert scale.to_px(scale.to_um(37.5)) == pytest.approx(37.5)


class TestUngueltigeWerte:
    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_unmoeglicher_massstab_wird_abgelehnt(self, value: float) -> None:
        with pytest.raises(ValueError):
            ScaleInfo(um_per_px=value, source=ScaleSource.MANUAL)

    def test_balken_ohne_laenge(self) -> None:
        with pytest.raises(ValueError):
            ScaleInfo.from_bar(value_um=200.0, bar_length_px=0.0,
                               source=ScaleSource.OVERLAY_BAR)


class TestAusschluss:
    def test_maske_deckt_den_kasten(self, scale: ScaleInfo) -> None:
        mask = overlay_mask((600, 800), scale)

        assert mask.sum() == 120 * 30
        assert mask[310, 450]
        assert not mask[100, 100]

    def test_rand_vergroessert_die_maske(self, scale: ScaleInfo) -> None:
        assert overlay_mask((600, 800), scale, pad=4).sum() == 128 * 38

    def test_ohne_massstab_wird_nichts_ausgeschlossen(self) -> None:
        mask = overlay_mask((600, 800), None)

        assert mask.shape == (600, 800)
        assert not mask.any()

    def test_maske_wird_am_bildrand_gestutzt(self) -> None:
        """Ein Overlay in der Ecke darf keinen Index aus dem Bild heraus erzeugen."""
        scale = ScaleInfo(um_per_px=1.0, source=ScaleSource.OVERLAY_BAR,
                          box=BBox(760, 570, 120, 30))
        mask = overlay_mask((600, 800), scale, pad=10)

        assert mask.shape == (600, 800)
        assert mask[-1, -1]

    def test_ausgeschlossene_flaeche_wird_berichtet(self, scale: ScaleInfo) -> None:
        """Die Zahl verkleinert später die Bezugsfläche der Porosität."""
        assert excluded_area_px((600, 800), scale) == 120 * 30


class TestManuell:
    def test_aus_gezogener_linie(self) -> None:
        scale = from_line((100, 100), (300, 100), "400 µm")

        assert scale.um_per_px == pytest.approx(2.0)
        assert scale.source is ScaleSource.MANUAL

    def test_schraege_linie_zaehlt_ihre_wahre_laenge(self) -> None:
        scale = from_line((0, 0), (30, 40), "100 µm")  # Länge 50

        assert scale.um_per_px == pytest.approx(2.0)

    def test_unverstaendliche_angabe_wird_abgelehnt(self) -> None:
        with pytest.raises(ValueError):
            from_line((0, 0), (100, 0), "zweihundert")

    def test_zu_kurze_linie_wird_abgelehnt(self) -> None:
        with pytest.raises(ValueError):
            from_line((10, 10), (10, 10), "200 µm")

    def test_direkte_vorgabe(self) -> None:
        scale = from_um_per_px(0.5, note="Objektiv 50x")

        assert scale.um_per_px == 0.5
        assert scale.warnings == ("Objektiv 50x",)


class TestHerkunft:
    def test_warnung_senkt_konfidenz(self, scale: ScaleInfo) -> None:
        gewarnt = scale.with_warning("unsicher", confidence_factor=0.5)

        assert gewarnt.warnings == ("unsicher",)
        assert gewarnt.confidence == pytest.approx(0.5)
        assert scale.warnings == (), "das Original bleibt unberührt"

    def test_vertrauenswuerdig_nur_ohne_warnung(self, scale: ScaleInfo) -> None:
        assert scale.is_trustworthy
        assert not scale.with_warning("hm").is_trustworthy

    def test_beschreibung_nennt_die_herkunft(self, scale: ScaleInfo) -> None:
        text = scale.describe()

        assert "µm/px" in text
        assert str(ScaleSource.OVERLAY_BAR) in text


def test_maske_laesst_sich_mit_anderen_masken_verrechnen(scale: ScaleInfo) -> None:
    """So nutzt es das Porenmodul später: Probenmaske minus Overlay."""
    from poredet.scale.exclusion import exclude

    specimen = np.ones((600, 800), dtype=bool)
    usable = exclude(specimen, scale)

    assert usable.sum() == 600 * 800 - 120 * 30
