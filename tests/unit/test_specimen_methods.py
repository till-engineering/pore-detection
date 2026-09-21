"""Die alternativen Trennverfahren - dieselben Anforderungen an alle.

Was hier geprüft wird, gilt unabhängig vom Weg: der Saum muss gefunden, die Poren müssen
verschont und ohne Einbettmittel darf keines erfunden werden. Ein Verfahren, das eines
davon nicht leistet, ist für die Porenmessung unbrauchbar - egal wie elegant es arbeitet.

Die Schwellen sind absichtlich locker. Sie sollen Konstruktionsfehler fangen, nicht
Verfahren gegeneinander ausspielen; das leistet der Vergleich an den echten Bildern
besser als eine Zahl im Test.
"""

from __future__ import annotations

import numpy as np
import pytest

from poredet.config.schema import SpecimenConfig
from poredet.dev.synthetic import make_specimen_image
from poredet.specimen import COMPARISON_METHODS, get_segmenter
from poredet.specimen import seeds as seeding
from poredet.specimen import signals as sig

#: Alle Verfahren, die im Vergleich antreten.
METHODS = list(COMPARISON_METHODS)


@pytest.fixture
def cfg() -> SpecimenConfig:
    return SpecimenConfig()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


class TestAlleVerfahren:
    @pytest.mark.parametrize("method", METHODS)
    def test_ist_registriert_und_baubar(self, method: str) -> None:
        segmenter = get_segmenter(method)

        assert segmenter.name == method
        assert hasattr(segmenter, "segment")

    @pytest.mark.parametrize("method", METHODS)
    def test_findet_den_saum(self, cfg, method: str) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.25)

        mask = get_segmenter(method).segment(truth.gray, cfg)

        assert mask.has_resin, f"{method} findet kein Einbettmittel"
        assert _iou(mask.resin, truth.resin) > 0.75

    @pytest.mark.parametrize("method", METHODS)
    def test_verschont_die_poren(self, cfg, method: str) -> None:
        """Poren sind so dunkel und glatt wie Harz - sie dürfen trotzdem nicht wegfallen."""
        pores = ((300, 150, 18), (420, 300, 14), (350, 380, 22))
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.2, pores=pores)

        mask = get_segmenter(method).segment(truth.gray, cfg)

        kept = (mask.specimen & truth.pores).sum() / truth.pores.sum()
        assert kept > 0.9, f"{method} entfernt {1 - kept:.0%} der Porenfläche"

    @pytest.mark.parametrize("method", METHODS)
    def test_erfindet_kein_harz(self, cfg, method: str) -> None:
        """Ohne Einbettmittel im Bild darf keines gemeldet werden."""
        truth = make_specimen_image(resin_sides=(), resin_frac=0.0)

        mask = get_segmenter(method).segment(truth.gray, cfg)

        assert mask.resin_frac < 0.05, f"{method} erfindet {mask.resin_frac:.0%} Harz"

    @pytest.mark.parametrize("method", METHODS)
    def test_probe_und_harz_ueberschneiden_sich_nicht(self, cfg, method: str) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.2)

        mask = get_segmenter(method).segment(truth.gray, cfg)

        assert not (mask.specimen & mask.resin).any()

    @pytest.mark.parametrize("method", METHODS)
    def test_maske_kommt_in_voller_aufloesung_zurueck(self, cfg, method: str) -> None:
        """Alle Verfahren rechnen verkleinert - das Ergebnis muss trotzdem passen."""
        truth = make_specimen_image(width=1600, height=1200, resin_sides=("left",))

        mask = get_segmenter(method).segment(truth.gray, cfg)

        assert mask.shape == (1200, 1600)

    @pytest.mark.parametrize("method", METHODS)
    def test_nennt_sich_im_ergebnis(self, cfg, method: str) -> None:
        truth = make_specimen_image(resin_sides=("left",))

        assert get_segmenter(method).segment(truth.gray, cfg).method == method


class TestGemeinsameSignale:
    def test_alle_verfahren_teilen_dieselbe_grundlage(self, cfg) -> None:
        """Der Vergleich misst die Verfahren nur dann, wenn die Vorarbeit dieselbe ist."""
        truth = make_specimen_image(resin_sides=("left",))
        signals = sig.compute(truth.gray, cfg)

        for method in METHODS:
            segmenter = get_segmenter(method)
            assert hasattr(segmenter, "from_signals"), f"{method} kann keine Signale teilen"
            mask = segmenter.from_signals(signals, cfg)
            assert mask.shape == truth.gray.shape

    def test_signale_sind_reproduzierbar(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=("left",))

        first = sig.compute(truth.gray, cfg)
        second = sig.compute(truth.gray, cfg)

        assert first.dark_threshold == second.dark_threshold
        assert first.texture_threshold == second.texture_threshold


class TestSaat:
    def test_harzsaat_liegt_im_harz(self, cfg) -> None:
        """Die Saat muss nicht vollständig sein - aber sie muss stimmen."""
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.25)
        signals = sig.compute(truth.gray, cfg)

        seeds = seeding.from_signals(signals, cfg.seeds)
        resin_truth = sig.upscale(truth.resin, signals.shape)

        assert seeds.resin.any()
        correct = (seeds.resin & resin_truth).sum() / seeds.resin.sum()
        assert correct > 0.95, f"nur {correct:.0%} der Harzsaat liegt wirklich im Harz"

    def test_probensaat_liegt_in_der_probe(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.25)
        signals = sig.compute(truth.gray, cfg)

        seeds = seeding.from_signals(signals, cfg.seeds)
        specimen_truth = sig.upscale(truth.specimen, signals.shape)

        assert seeds.specimen.any()
        correct = (seeds.specimen & specimen_truth).sum() / seeds.specimen.sum()
        assert correct > 0.95

    def test_dunkles_band_in_der_probe_wird_nicht_zur_saat(self, cfg) -> None:
        """Der Fehlerfall des Schwellverfahrens - die Saat kann ihn gar nicht erst machen.

        Ein dunkler, glatter Streifen mitten in der Probe, der oben und unten den Bildrand
        berührt. Er liegt außerhalb des Randstreifens und kommt deshalb nicht in die Saat.
        """
        truth = make_specimen_image(resin_sides=("left",), resin_frac=0.15)
        gray = truth.gray.copy()
        middle = gray.shape[1] // 2
        gray[:, middle - 12:middle + 12] = 40         # dunkel und glatt, quer durchs Bild

        signals = sig.compute(gray, cfg)
        seeds = seeding.from_signals(signals, cfg.seeds)

        height, width = signals.shape
        band = sig.border_band((height, width), max(2, int(0.10 * min(height, width))))
        column = width // 2
        inner_stripe = np.zeros((height, width), dtype=bool)
        inner_stripe[:, column - 2:column + 2] = True
        inner_stripe &= ~band

        assert not (seeds.resin & inner_stripe).any(), (
            "das innere Band ist in die Harzsaat geraten"
        )

    def test_ohne_dunklen_rand_keine_saat(self, cfg) -> None:
        truth = make_specimen_image(resin_sides=(), resin_frac=0.0)
        signals = sig.compute(truth.gray, cfg)

        seeds = seeding.from_signals(signals, cfg.seeds)

        assert not seeds.usable or seeds.resin.mean() < 0.02
