"""Trennung Probe / Einbettmittel.

**Standardverfahren ist** :mod:`~poredet.specimen.grabcut`. Es ist am Vergleichssatz
ausgewählt worden und wird von allen weiteren Modulen verwendet; wer nichts anderes
angibt, bekommt es. Die übrigen fünf Verfahren bleiben registriert und über
``SpecimenConfig.method`` erreichbar - sie sind nicht abgeschrieben, sondern nicht
Standard. ``poredet specimen-compare`` stellt sie nebeneinander.

Das Ergebnis dieser Schicht ist eine :class:`~poredet.specimen.base.SpecimenMask` - und
das ist die einzige Schnittstelle, die spätere Module brauchen::

    mask = segment(image.gray, color=image.color)
    bezugsflaeche_px = mask.specimen_area_px   # Nenner der Porositaet
    nur_hier_suchen  = mask.specimen           # Porendetektion arbeitet in dieser Maske

``color`` ist optional, sollte aber übergeben werden, wo es vorliegt: GrabCut schätzt
seine Modelle über die Farbkanäle. Die anderen Verfahren ignorieren es.

Der Maßstab spielt hier bewusst keine Rolle. Viele Schliffbilder kommen ohne
Maßstabsbalken an, und ein Modul, das dann nicht arbeiten könnte, wäre unbrauchbar. Wo
beides vorliegt, werden die Masken einfach verrechnet::

    from poredet.scale import overlay_mask
    auswertbar = mask.specimen & ~overlay_mask(mask.shape, scale, pad=4)
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ..config.schema import SpecimenConfig
from .base import SpecimenMask, SpecimenSegmenter
from .boundary_path import BoundaryPathSegmenter
from .chan_vese import ChanVeseSegmenter
from .full_frame import FullFrameSegmenter
from .grabcut import GrabCutSegmenter
from .largest_region import LargestRegionSegmenter
from .lasso import LassoSegmenter
from .manual_roi import from_polygon
from .random_walker import RandomWalkerSegmenter
from .signals import local_std
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

#: Die Verfahren, die im Vergleich gegeneinander antreten, in sinnvoller Reihenfolge.
COMPARISON_METHODS: tuple[str, ...] = (
    "lasso", "random_walker", "watershed", "grabcut", "chan_vese", "boundary_path",
)


def register_segmenter(name: str, factory: Callable[[], SpecimenSegmenter]) -> None:
    """Einen weiteren Segmentierer bekannt machen."""
    _SEGMENTERS[name] = factory


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
    cfg: SpecimenConfig | None = None,
    color: np.ndarray | None = None,
) -> SpecimenMask:
    """Probe vom Einbettmittel trennen - der übliche Einstieg.

    ``color`` ist optional und wird nur an Verfahren weitergereicht, die etwas damit
    anfangen können. Das ist derzeit allein ``grabcut``, das seine Modelle über die
    Farbkanäle schätzt; alle anderen rechnen auf Helligkeit und Textur. Wer ein Farbbild
    zur Hand hat, sollte es übergeben - es kostet nichts und kann nur helfen.
    """
    cfg = cfg or SpecimenConfig()
    segmenter = get_segmenter(cfg.method)
    if color is not None and getattr(segmenter, "uses_color", False):
        return segmenter.segment_color(gray, color, cfg)
    return segmenter.segment(gray, cfg)


__all__ = [
    "COMPARISON_METHODS",
    "BoundaryPathSegmenter",
    "ChanVeseSegmenter",
    "FullFrameSegmenter",
    "GrabCutSegmenter",
    "LargestRegionSegmenter",
    "LassoSegmenter",
    "RandomWalkerSegmenter",
    "SpecimenConfig",
    "SpecimenMask",
    "SpecimenSegmenter",
    "WatershedSegmenter",
    "from_polygon",
    "get_segmenter",
    "local_std",
    "register_segmenter",
    "segment",
]
