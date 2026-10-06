"""Trennung Probe / Einbettmittel.

**Standardverfahren ist** ``grabcut`` (am Vergleichssatz ausgewählt). Die übrigen
Verfahren bleiben erhalten und sind über ``specimen.method`` in der
``einstellungen.yaml`` wählbar:

    grabcut          Graph-Schnitt mit selbst gelernten Farbmodellen (Standard)
    lasso            Schwellen auf Helligkeit und Textur, Randoffenheit, Materialprüfung
    random_walker    Diffusion von Saatpunkten aus
    watershed        Wasserscheide zwischen zwei Saaten auf dem Gradientenbild
    chan_vese        Levelset-Kontur auf Regionenstatistik
    boundary_path    Trennlinie als billigster Weg quer durchs Bild
    largest_region   größte helle Region
    full_frame       ganzes Bild ist Probe

Alle rechnen auf denselben Signalen (``signals.py``) und durchlaufen denselben Abschluss
(``postprocess.py``). Ergebnis ist eine ``SpecimenMask``::

    mask = segment(gray, cfg.specimen, color=color)
    mask.specimen          # Probe - Nenner der Porosität, hier wird nach Poren gesucht
    mask.resin             # Einbettmittel

``color`` ist optional: nur GrabCut nutzt die Farbkanäle.

Ein neues Verfahren = neue Datei mit einer Klasse ``segment(gray, cfg)`` plus ein
Eintrag in ``_SEGMENTERS`` unten.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ..einstellungen import Abschnitt
from .base import SpecimenMask, SpecimenSegmenter
from .boundary_path import BoundaryPathSegmenter
from .chan_vese import ChanVeseSegmenter
from .full_frame import FullFrameSegmenter
from .grabcut import GrabCutSegmenter
from .largest_region import LargestRegionSegmenter
from .lasso import LassoSegmenter
from .random_walker import RandomWalkerSegmenter
from .watershed import WatershedSegmenter

log = logging.getLogger(__name__)

#: Name -> Konstruktor. Ein neuer Segmentierer ist eine neue Datei plus ein Eintrag hier;
#: die Konfiguration wählt ihn über ``specimen.method`` aus, ohne dass Code sich ändert.
_SEGMENTERS: dict[str, Callable[[], SpecimenSegmenter]] = {
    # Schwellenbasiert
    "lasso": LassoSegmenter,
    "largest_region": LargestRegionSegmenter,
    # Ausbreitend, von Saatpunkten aus
    "random_walker": RandomWalkerSegmenter,
    "watershed": WatershedSegmenter,
    "grabcut": GrabCutSegmenter,
    # Konturbasiert
    "chan_vese": ChanVeseSegmenter,
    "boundary_path": BoundaryPathSegmenter,
    # Rückfallebene
    "full_frame": FullFrameSegmenter,
}


def get_segmenter(name: str) -> SpecimenSegmenter:
    """Der benannte Segmentierer. Ein unbekannter Name ist ein Fehler, keine Notlösung."""
    factory = _SEGMENTERS.get(name)
    if factory is None:
        raise KeyError(
            f"Unbekannter Segmentierer {name!r} - bekannt sind: "
            f"{', '.join(sorted(_SEGMENTERS))}"
        )
    return factory()


def segment(
    gray: np.ndarray,
    cfg: Abschnitt,
    color: np.ndarray | None = None,
) -> SpecimenMask:
    """Probe vom Einbettmittel trennen - der übliche Einstieg.

    ``color`` ist optional und wird nur an Verfahren weitergereicht, die etwas damit
    anfangen können. Das ist derzeit allein ``grabcut``, das seine Modelle über die
    Farbkanäle schätzt; alle anderen rechnen auf Helligkeit und Textur. Wer ein Farbbild
    zur Hand hat, sollte es übergeben - es kostet nichts und kann nur helfen.
    """
    segmenter = get_segmenter(cfg.method)
    if color is not None and getattr(segmenter, "uses_color", False):
        return segmenter.segment_color(gray, color, cfg)
    return segmenter.segment(gray, cfg)
