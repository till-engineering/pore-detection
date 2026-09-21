"""Geometrie der Maßstabserkennung, geprüft an synthetischen Bildern.

Hier ist die Wahrheit exakt bekannt: Balkenlänge, Kastenlage und Beschriftung sind
vorgegeben. Damit lässt sich absolute Korrektheit prüfen - bei echten Bildern ließe sich
nur feststellen, dass sich nichts verändert hat.

Diese Tests brauchen **keine OCR**. Sie prüfen den Teil, der die Messgenauigkeit
bestimmt: die gemessene Balkenlänge. Ein Pixel Fehler bei einem 50-px-Balken sind zwei
Prozent auf jede spätere Porengröße.
"""

from __future__ import annotations

import numpy as np
import pytest

from poredet.config.schema import ScaleConfig
from poredet.core.models import BBox
from poredet.dev.synthetic import make_overlay_image
from poredet.scale.detectors.box_overlay import BoxOverlayDetector


@pytest.fixture
def detector() -> BoxOverlayDetector:
    return BoxOverlayDetector()


@pytest.fixture
def cfg() -> ScaleConfig:
    return ScaleConfig()


class TestBalken:
    @pytest.mark.parametrize("bar_length", [20, 47, 88, 126, 300])
    def test_misst_die_balkenlaenge_exakt(self, detector, cfg, bar_length: int) -> None:
        """Die Balkenlänge ist die Referenz - sie muss auf das Pixel stimmen."""
        truth = make_overlay_image(bar_length_px=bar_length)
        candidates = detector.detect(truth.gray, cfg)

        assert candidates, "Overlay wurde nicht gefunden"
        assert candidates[0].bar_length_px == bar_length

    @pytest.mark.parametrize("background", ["texture", "bright", "dark"])
    def test_findet_overlay_auf_jedem_untergrund(self, detector, cfg, background: str) -> None:
        """"bright" ist der harte Fall: heller Untergrund, der dem Kasten gleicht."""
        truth = make_overlay_image(background=background, bar_length_px=100)
        candidates = detector.detect(truth.gray, cfg)

        assert candidates, f"Untergrund {background}: nichts gefunden"
        assert candidates[0].bar_length_px == 100

    @pytest.mark.parametrize("position", ["bottom_right", "top_left", "center"])
    def test_findet_overlay_an_jeder_stelle(self, detector, cfg, position: str) -> None:
        """Es gibt keinen festen Suchbereich - das Overlay saß in den Testbildern überall."""
        truth = make_overlay_image(position=position, bar_length_px=90)
        candidates = detector.detect(truth.gray, cfg)

        assert candidates
        assert candidates[0].bar == truth.bar

    @pytest.mark.parametrize("label_below", [True, False])
    def test_findet_beschriftung_ober_und_unterhalb(self, detector, cfg, label_below) -> None:
        truth = make_overlay_image(label_below=label_below)
        candidates = detector.detect(truth.gray, cfg)

        assert candidates
        candidate = candidates[0]
        assert candidate.label is not None
        assert candidate.label_side == ("below" if label_below else "above")


class TestKasten:
    def test_kasten_umschliesst_balken_und_beschriftung(self, detector, cfg) -> None:
        truth = make_overlay_image(bar_length_px=110)
        candidate = detector.detect(truth.gray, cfg)[0]

        assert candidate.box.contains(candidate.bar)
        assert candidate.label is not None
        assert candidate.box.contains(candidate.label)

    def test_kasten_trifft_die_wahrheit(self, detector, cfg) -> None:
        truth = make_overlay_image(bar_length_px=110)
        candidate = detector.detect(truth.gray, cfg)[0]

        assert candidate.box == truth.box

    def test_beschriftung_enthaelt_den_balken_nicht(self, detector, cfg) -> None:
        """Sonst liest die OCR den Balken als Zeichen mit."""
        truth = make_overlay_image(bar_length_px=110)
        candidate = detector.detect(truth.gray, cfg)[0]

        assert candidate.label is not None
        overlap_y = min(candidate.label.y2, candidate.bar.y2) - max(
            candidate.label.y, candidate.bar.y
        )
        assert overlap_y <= 0


class TestKeineFalschenFunde:
    def test_bild_ohne_overlay_liefert_nichts(self, detector, cfg) -> None:
        rng = np.random.default_rng(3)
        gray = np.clip(rng.normal(150, 30, size=(400, 500)), 0, 254).astype(np.uint8)

        assert detector.detect(gray, cfg) == []

    def test_kratzer_ist_kein_balken(self, detector, cfg) -> None:
        """Ein langer dunkler Strich im Gefüge - aber ohne weißen Kasten drumherum."""
        rng = np.random.default_rng(4)
        gray = np.clip(rng.normal(150, 20, size=(400, 500)), 0, 254).astype(np.uint8)
        gray[200:203, 100:300] = 10

        assert detector.detect(gray, cfg) == []

    def test_schwarze_flaeche_ist_kein_balken(self, detector, cfg) -> None:
        """Massiv und dunkel, aber nicht langgestreckt."""
        rng = np.random.default_rng(5)
        gray = np.clip(rng.normal(200, 15, size=(400, 500)), 0, 254).astype(np.uint8)
        gray[150:250, 150:250] = 0

        assert detector.detect(gray, cfg) == []


class TestMehrereKandidaten:
    def test_der_echte_balken_gewinnt(self, detector, cfg) -> None:
        """Ein zweiter dunkler Strich ohne Kasten darf den Maßstab nicht verdrängen."""
        truth = make_overlay_image(bar_length_px=90, position="bottom_right")
        gray = truth.gray.copy()
        gray[60:64, 40:340] = 0  # längerer Strich, aber frei im Gefüge

        candidates = detector.detect(gray, cfg)

        assert candidates
        assert candidates[0].bar == truth.bar


def test_bbox_rechnet_richtig() -> None:
    box = BBox(10, 20, 30, 40)

    assert (box.x2, box.y2) == (40, 60)
    assert box.area == 1200
    assert box.center == (25.0, 40.0)
    assert box.contains(BBox(12, 22, 5, 5))
    assert not box.contains(BBox(38, 22, 5, 5))
