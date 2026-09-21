"""Porendetektion - der austauschbare Kern des Programms.

**Standardverfahren ist** :mod:`~poredet.detection.local_contrast`: Hintergrundabzug mit
Hysterese. Austausch über ``PoreConfig.method``; ein neuer Detektor ist eine Datei plus
ein Eintrag in :data:`_DETECTORS`, ohne dass sich sonst etwas ändert.

Die Arbeitsteilung ist eng gezogen und das ist der Punkt:

* **Der Detektor** bekommt Graubild, Probenmaske und Kontrastbild und liefert eine
  Kandidatenmaske. Sonst nichts.
* **Die Nachbearbeitung** (:mod:`~poredet.detection.postprocess`) räumt auf, füllt Löcher
  und trennt zusammengewachsene Nester - verfahrensunabhängig und für alle gleich.
* **Die Vermessung** (:mod:`poredet.measurement`) macht daraus Zahlen.
* **Die Filter** (:mod:`poredet.analysis.filters`) entscheiden, was fachlich als Pore
  gilt - mit Begründung je verworfenem Objekt.

Ein ausgetauschter Detektor kann damit weder die Vermessung noch die Filterlogik
durcheinanderbringen; er beantwortet nur die eine Frage, für die er da ist.
"""

from __future__ import annotations

from collections.abc import Callable

from .adaptive import AdaptiveDetector
from .base import DetectionInput, DetectionResult, PoreDetector
from .local_contrast import (
    LocalContrastDetector,
    contrast_image,
    estimate_background,
)
from .threshold import ThresholdDetector

#: Name -> Konstruktor. Die Konfiguration wählt über ``PoreConfig.method`` aus.
_DETECTORS: dict[str, Callable[[], PoreDetector]] = {
    "local_contrast": LocalContrastDetector,
    "threshold": ThresholdDetector,
    "adaptive": AdaptiveDetector,
}

#: Die Verfahren, die ein Vergleichslauf gegeneinander antreten lässt.
COMPARISON_METHODS: tuple[str, ...] = ("local_contrast", "threshold", "adaptive")


def register_detector(name: str, factory: Callable[[], PoreDetector]) -> None:
    """Einen weiteren Detektor bekannt machen."""
    _DETECTORS[name] = factory


def get_detector(name: str) -> PoreDetector:
    """Der benannte Detektor. Ein unbekannter Name ist ein Fehler, keine Notlösung."""
    factory = _DETECTORS.get(name)
    if factory is None:
        raise KeyError(
            f"Unbekannter Porendetektor {name!r} - bekannt sind: "
            f"{', '.join(sorted(_DETECTORS))}"
        )
    return factory()


__all__ = [
    "COMPARISON_METHODS",
    "AdaptiveDetector",
    "DetectionInput",
    "DetectionResult",
    "LocalContrastDetector",
    "PoreDetector",
    "ThresholdDetector",
    "contrast_image",
    "estimate_background",
    "get_detector",
    "register_detector",
]
