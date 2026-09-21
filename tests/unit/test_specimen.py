"""Trennung Probe / Einbettmittel, geprüft an synthetischen Schliffen.

Hier ist die Wahrheit vorgegeben: Lage und Größe des Saums, Lage der Poren. Damit lässt
sich prüfen, worauf es ankommt - und das ist **nicht** in erster Linie, ob das Harz
gefunden wird. Entscheidend ist, was dabei *nicht* passiert:

* Poren sind genauso dunkel und glatt wie Harz. Ein Verfahren, das sie mitentfernt, wäre
  unbrauchbar - die Poren sind ja das Messobjekt.
* Ohne Einbettmittel im Bild darf keines erfunden werden. Ein erfundener Saum schneidet
  Probenmaterial weg und verfälscht die Bezugsfläche der Porosität.
"""

from __future__ import annotations

import numpy as np
import pytest

from poredet.config.schema import SpecimenConfig
from poredet.dev.synthetic import make_specimen_image
from poredet.specimen import get_segmenter, segment
from poredet.specimen.full_frame import FullFrameSegmenter
from poredet.specimen.manual_roi import from_polygon


@pytest.fixture
def cfg() -> SpecimenConfig:
    return SpecimenConfig()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


class TestSaumFinden:
    @pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
    def test_findet_den_saum_an_jeder_seite(self, cfg, side: str) -> None:
        truth = make_specimen_image(resin_sides=(side,), resin_frac=0.25)

        mask = segment(truth.gray, cfg)

        assert mask.has_resin, f"Saum an der Seite {side} nicht gefunden"
        assert _iou(mask.resin, truth.resin) > 0.9

    def test_findet_zwei_gegenueberliegende_baender(self, cfg) -> None:
        """Die Probe durchquert das Bild - der Saum zerfällt in zwei Randbänder."""
        truth = make_specimen_image(resin_sides=("top", "bottom"), resin_frac=0.2)

        mask = segment(truth.gray, cfg)

        assert _iou(mask.resin, truth.resin) > 0.9

    def test_schmaler_saum(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.06)

        mask = segment(truth.gray, cfg)

        assert mask.has_resin
        assert _iou(mask.resin, truth.resin) > 0.8

    def test_luftblasen_gehoeren_zum_harz(self, cfg) -> None:
        """Blasen sind hell und rund - pixelweise kein Harz, sachlich schon."""
        truth = make_specimen_image(
            resin_sides=("left",), resin_frac=0.3,
            bubbles=((60, 100, 25), (90, 250, 30), (50, 380, 20)),
        )

        mask = segment(truth.gray, cfg)

        assert _iou(mask.resin, truth.resin) > 0.85


class TestPorenBleibenErhalten:
    """Der wichtigste Test des Moduls."""

    def test_poren_werden_nicht_entfernt(self, cfg) -> None:
        pores = ((300, 150, 18), (420, 300, 14), (350, 380, 22))
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.2, pores=pores)

        mask = segment(truth.gray, cfg)

        assert truth.pores.any()
        kept = (mask.specimen & truth.pores).sum() / truth.pores.sum()
        assert kept > 0.95, f"nur {kept:.0%} der Porenfläche blieb in der Probenmaske"

    def test_angeschnittene_pore_am_rand_bleibt_probe(self, cfg) -> None:
        """Sie ist randoffen wie Harz - aber klein und nur kurz am Rand entlang."""
        truth = make_specimen_image(
            resin_sides=("left",), resin_frac=0.2, pores=((595, 220, 26),)
        )

        mask = segment(truth.gray, cfg)

        kept = (mask.specimen & truth.pores).sum() / truth.pores.sum()
        assert kept > 0.9

    def test_poren_zaehlen_zur_bezugsflaeche(self, cfg) -> None:
        """Porosität ist Porenfläche / Probenfläche - der Nenner schließt die Poren ein."""
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.2,
                                    pores=((300, 150, 25),))

        mask = segment(truth.gray, cfg)

        assert (mask.specimen & truth.pores).sum() > 0


class TestKeinHarzErfinden:
    def test_bild_ohne_einbettmittel(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=(), resin_frac=0.0)

        mask = segment(truth.gray, cfg)

        assert not mask.has_resin
        assert mask.specimen_frac == pytest.approx(1.0)

    def test_dunkleres_gefuege_am_rand_ist_kein_harz(self, cfg) -> None:
        """Ein Randbereich desselben Werkstoffs, nur dunkler - der Kontrast fehlt."""
        truth = make_specimen_image(
            resin_sides=("left",), resin_frac=0.25,
            specimen_gray=150, resin_gray=125, resin_texture=16.0,
        )

        mask = segment(truth.gray, cfg)

        assert not mask.has_resin
        assert any("Graustufen" in w for w in mask.warnings)

    def test_reines_rauschen(self, cfg) -> None:
        rng = np.random.default_rng(5)
        gray = np.clip(rng.normal(120, 40, size=(400, 500)), 0, 255).astype(np.uint8)

        mask = segment(gray, cfg)

        assert mask.specimen_frac > 0.9


class TestKennzahlen:
    def test_probe_und_harz_schliessen_einander_aus(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.2)

        mask = segment(truth.gray, cfg)

        assert not (mask.specimen & mask.resin).any()

    def test_maske_hat_die_bildgroesse(self, cfg) -> None:
        truth = make_specimen_image(width=640, height=480)

        mask = segment(truth.gray, cfg)

        assert mask.shape == (480, 640)
        assert mask.specimen.dtype == np.bool_

    def test_grosse_bilder_werden_verkleinert_gerechnet(self) -> None:
        """Die Maske kommt trotzdem in voller Auflösung zurück."""
        truth = make_specimen_image(width=2400, height=1800, resin_sides=("left",))
        cfg = SpecimenConfig(work_max_px=600)

        mask = segment(truth.gray, cfg)

        assert mask.shape == (1800, 2400)
        assert _iou(mask.resin, truth.resin) > 0.85

    def test_freistellen(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.2)
        mask = segment(truth.gray, cfg)

        cut = mask.apply(truth.gray, fill=0)

        assert (cut[mask.resin] == 0).all()


class TestWeitereVerfahren:
    def test_full_frame_nimmt_alles(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.3)

        mask = FullFrameSegmenter().segment(truth.gray, cfg)

        assert mask.specimen.all()
        assert not mask.has_resin

    def test_verfahren_ueber_die_konfiguration_waehlbar(self) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.3)

        mask = segment(truth.gray, SpecimenConfig(method="full_frame"))

        assert mask.method == "full_frame"

    def test_unbekanntes_verfahren_ist_ein_fehler(self) -> None:
        with pytest.raises(KeyError):
            get_segmenter("gibtsnicht")

    def test_polygon_aus_der_gui(self) -> None:
        mask = from_polygon((200, 300), [(50, 50), (250, 50), (250, 150), (50, 150)])

        assert mask.method == "manual_roi"
        assert mask.specimen[100, 150]
        assert not mask.specimen[10, 10]

    def test_polygon_braucht_drei_punkte(self) -> None:
        with pytest.raises(ValueError):
            from_polygon((100, 100), [(0, 0), (10, 10)])
