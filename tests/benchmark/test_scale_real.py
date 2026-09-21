"""Die Maßstabserkennung gegen die echten Testbilder.

Die Wahrheit kommt aus den ImageJ-Tags der TIFFs (``io.metadata``), nicht aus einer im
Test gepflegten Tabelle - sonst prüft der Test irgendwann nur noch die Tabelle.

Der Produktivpfad rührt diese Tags nicht an. Er liest ausschließlich den eingebrannten
Balken; die Metadaten sagen hier nur, was dabei herauskommen müsste.

Erwartete Genauigkeit: der Balken wird in ganzen Pixeln gezeichnet. Bei ``1.035 m`` und
97 px/m müsste er 100,4 px lang sein, ImageJ zeichnet 100 px - 0,4 % Abweichung, die im
Bild stecken und nicht in der Messung. Ein Pixel auf den kürzesten Balken des Satzes
(20 px) sind 5 %; die Schranke liegt deshalb bei 1 %, was für alle 14 Bilder gilt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poredet.io.image_reader import read_image
from poredet.io.metadata import read_imagej_calibration

pytestmark = pytest.mark.benchmark

#: Toleranz gegenüber der ImageJ-Referenz.
MAX_DEVIATION_PCT = 1.0


@pytest.fixture(scope="module")
def results(scale_images: list[Path], resolver) -> dict[str, tuple]:
    """Einmal alle Bilder auswerten - das OCR-Modell lädt sonst je Test neu."""
    out = {}
    for path in scale_images:
        image = read_image(path)
        outcome = resolver.resolve(image.gray, path)
        calibration = read_imagej_calibration(path)
        out[path.name] = (outcome, calibration)
    return out


def test_jedes_bild_liefert_einen_massstab(results) -> None:
    missing = [
        f"{name}: {outcome.rejections[0] if outcome.rejections else 'kein Grund vermerkt'}"
        for name, (outcome, _cal) in results.items()
        if outcome.scale is None
    ]

    assert not missing, "Maßstab nicht erkannt:\n  " + "\n  ".join(missing)


def test_massstab_stimmt_mit_der_imagej_referenz(results) -> None:
    deviations = {}
    for name, (outcome, calibration) in results.items():
        if outcome.scale is None or calibration is None:
            continue
        reference = calibration.um_per_px
        deviations[name] = 100.0 * (outcome.scale.um_per_px - reference) / reference

    assert deviations, "keine Referenzwerte vorhanden"
    too_far = {n: d for n, d in deviations.items() if abs(d) > MAX_DEVIATION_PCT}

    detail = ", ".join(f"{n} {d:+.3f} %" for n, d in too_far.items())
    assert not too_far, f"Abweichung über {MAX_DEVIATION_PCT:.1f} %: {detail}"


def test_balken_ist_die_referenzlaenge_nicht_der_kasten(results) -> None:
    """Der häufigste stille Fehler: die Kastenbreite als Maßstab zu nehmen."""
    for name, (outcome, _cal) in results.items():
        scale = outcome.scale
        if scale is None:
            continue
        assert scale.bar_box is not None and scale.box is not None, name
        assert scale.bar_length_px == scale.bar_box.w, name
        assert scale.bar_box.w < scale.box.w, name


def test_ausschlussbereich_ist_bekannt(results) -> None:
    """Ohne ihn zählt der schwarze Balken später als größte Pore des Bildes."""
    for name, (outcome, _cal) in results.items():
        scale = outcome.scale
        if scale is None:
            continue
        box = scale.exclusion_box(pad=4)
        assert box is not None and box.area > 0, name
        assert box.contains(scale.bar_box), name


def test_jedes_ergebnis_nennt_seine_herkunft(results) -> None:
    for name, (outcome, _cal) in results.items():
        scale = outcome.scale
        if scale is None:
            continue
        assert scale.label_text, name
        assert scale.engine, name
        assert 0.0 <= scale.confidence <= 1.0, name


def test_konfidenz_bleibt_brauchbar(results) -> None:
    """Ein durchweg unsicheres Ergebnis wäre in der Praxis wertlos, auch wenn es stimmt."""
    scales = [o.scale for o, _ in results.values() if o.scale is not None]
    weak = [s for s in scales if s.confidence < 0.5]

    assert not weak, "zu unsicher: " + ", ".join(str(s.label_text) for s in weak)
