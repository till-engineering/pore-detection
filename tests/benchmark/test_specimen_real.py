"""Die Trennung Probe / Einbettmittel gegen die echten Schliffe.

Für diese Bilder gibt es keine von Hand gezeichneten Masken, also auch keine Wahrheit,
gegen die sich rechnen ließe. Geprüft wird deshalb zweierlei:

* **Eigenschaften**, die unabhängig von der genauen Grenze gelten müssen - der Saum liegt
  am Bildrand, Probe und Harz überschneiden sich nicht, die Probe bleibt der größere Teil.
* **Die Gegenprobe**: dieselbe Aufnahmeserie ohne Einbettmittel. Dort darf keines
  gefunden werden. Das ist der schärfere der beiden Tests - ein erfundener Saum schneidet
  Probenmaterial weg, und niemand sieht es der Porositätszahl später an.

Geprüft wird das **Standardverfahren** aus der Konfiguration, nicht ein fest verdrahtetes:
Was ausgeliefert wird, muss getestet sein. Daneben hält
:func:`test_gegenprobe_je_verfahren` den Stand aller Verfahren fest, damit die Schwäche
des Standards nicht aus dem Blick gerät.

Die Zahlen sind **kein** Sollwert, sondern ein Regressionsschutz: sie halten den Stand
fest, der am Kontrollbild geprüft wurde.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poredet.config.schema import SpecimenConfig
from poredet.io.image_reader import read_image
from poredet.specimen import COMPARISON_METHODS, get_segmenter, segment

pytestmark = pytest.mark.benchmark

#: Harzanteil des Standardverfahrens, am Kontrollbild geprüft. Toleranz: 6 Prozentpunkte.
EXPECTED: dict[str, float] = {
    "asdasd.jpg": 0.487,
    "bsp_maßbalken.png": 0.410,
    "metallographyofsteels-img07.jpg": 0.339,
    "Mitte_Nital_4s_1,25x.png": 0.617,
    "Mitte_Nital_4s_1,25xsss.png": 0.303,
    "Mitte_Nital_4s_10sssx.png": 0.162,
    "Mitte_Nital_4s_10x.png": 0.152,
    "Mitte_Nital_4s_20x.png": 0.065,
    "Mitte_Nital_4s_50sssx.png": 0.515,
    "Mitte_Nital_4s_50x.png": 0.262,
    "Mitte_Nital_4s_5x.png": 0.224,
    "Schweissnaht_Stumpfstoss_V-Naht_4.jpg": 0.257,
}

TOLERANCE = 0.06

#: Ab diesem Anteil zählt ein Fund in einem Bild ohne Einbettmittel als Falschfund.
#: Darunter sind es einzelne Flecken, die der Abschluss ohnehin wegräumt.
FALSE_POSITIVE_THRESHOLD = 0.01

#: Falschfunde je Verfahren an den 14 Aufnahmen **ohne** Einbettmittel, gemessen.
#:
#: Das ist die Kehrseite der Verfahrenswahl und gehört festgehalten: grabcut deckt den
#: Saum am vollständigsten ab, erfindet ihn aber auch am häufigsten. Eines der 14 Bilder
#: ist polarisiert aufgenommen und hat echte schwarze Ecken außerhalb des Sichtfelds -
#: dort ist ein Fund sachlich richtig, weshalb selbst das beste Verfahren bei 1 landet.
EXPECTED_FALSE_POSITIVES: dict[str, int] = {
    "lasso": 1,
    "random_walker": 3,
    "watershed": 4,
    "grabcut": 5,
    "chan_vese": 4,
    "boundary_path": 1,
}


@pytest.fixture(scope="module")
def masks(resin_images: list[Path]) -> dict[str, object]:
    """Alle Bilder mit dem Standardverfahren, so wie die CLI es auch aufruft."""
    out = {}
    for path in resin_images:
        image = read_image(path)
        out[path.name] = segment(image.gray, color=image.color)
    return out


def test_das_standardverfahren_ist_gesetzt() -> None:
    """Die Verfahrenswahl steht in der Konfiguration, nicht im Aufrufcode."""
    assert SpecimenConfig().method == "grabcut"


def test_in_jedem_bild_wird_einbettmittel_gefunden(masks) -> None:
    missing = [name for name, mask in masks.items() if not mask.has_resin]

    assert not missing, "kein Einbettmittel gefunden in: " + ", ".join(missing)


def test_harzanteil_bleibt_stabil(masks) -> None:
    drifted = {
        name: (mask.resin_frac, EXPECTED[name])
        for name, mask in masks.items()
        if name in EXPECTED and abs(mask.resin_frac - EXPECTED[name]) > TOLERANCE
    }

    detail = ", ".join(f"{n}: {got:.1%} statt {want:.1%}" for n, (got, want) in drifted.items())
    assert not drifted, f"Harzanteil abgewandert - {detail}"


def test_probe_und_harz_ueberschneiden_sich_nicht(masks) -> None:
    for name, mask in masks.items():
        assert not (mask.specimen & mask.resin).any(), name


def test_die_probe_bleibt_der_groessere_teil(masks) -> None:
    """Ein Schliffbild zeigt eine Probe, keinen Einbettling."""
    too_small = {n: m.specimen_frac for n, m in masks.items() if m.specimen_frac < 0.3}

    assert not too_small, f"Probenanteil unplausibel klein: {too_small}"


def test_der_saum_liegt_am_bildrand(masks) -> None:
    """Einbettmittel umschließt die Probe - es kann nicht mitten im Bild schweben."""
    for name, mask in masks.items():
        resin = mask.resin
        border = (
            resin[0, :].any() or resin[-1, :].any()
            or resin[:, 0].any() or resin[:, -1].any()
        )
        assert border, f"{name}: gefundenes Einbettmittel berührt den Bildrand nicht"


def test_gegenprobe_des_standardverfahrens(no_resin_images: list[Path]) -> None:
    """Der schärfere Test: hier ist kein Harz im Bild."""
    found = {}
    for path in no_resin_images:
        image = read_image(path)
        mask = segment(image.gray, color=image.color)
        if mask.resin_frac >= FALSE_POSITIVE_THRESHOLD:
            found[path.name] = mask.resin_frac

    limit = EXPECTED_FALSE_POSITIVES[SpecimenConfig().method]
    assert len(found) <= limit, (
        f"Einbettmittel erfunden in {len(found)} Bildern ohne welches (erlaubt: {limit}): "
        f"{found}"
    )


@pytest.mark.parametrize("method", list(COMPARISON_METHODS))
def test_gegenprobe_je_verfahren(no_resin_images: list[Path], method: str) -> None:
    """Der Stand aller Verfahren, damit die Schwäche des Standards messbar bleibt.

    Ohne diesen Test wäre die Verfahrenswahl eine einmalige Momentaufnahme. So wird
    sichtbar, wenn eine Änderung am gemeinsamen Abschluss ein Verfahren verbessert oder
    verschlechtert - auch eines, das gerade nicht Standard ist.
    """
    cfg = SpecimenConfig(method=method)
    segmenter = get_segmenter(method)

    count = 0
    for path in no_resin_images:
        image = read_image(path)
        if image.color is not None and getattr(segmenter, "uses_color", False):
            mask = segmenter.segment_color(image.gray, image.color, cfg)
        else:
            mask = segmenter.segment(image.gray, cfg)
        count += int(mask.resin_frac >= FALSE_POSITIVE_THRESHOLD)

    assert count <= EXPECTED_FALSE_POSITIVES[method], (
        f"{method} erfindet jetzt in {count} von {len(no_resin_images)} Bildern "
        f"Einbettmittel (bisher: {EXPECTED_FALSE_POSITIVES[method]})"
    )
