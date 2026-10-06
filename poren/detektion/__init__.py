"""Porendetektion - der austauschbare Kern des Programms.

**Standardverfahren ist** ``local_contrast``: Hintergrundabzug mit
Hysterese. Austausch über ``pore.method`` in der einstellungen.yaml; ein neuer Detektor ist eine Datei plus
ein Eintrag in :data:`_DETECTORS`, ohne dass sich sonst etwas ändert.

Die Arbeitsteilung ist eng gezogen und das ist der Punkt:

* **Der Detektor** bekommt Graubild, Probenmaske und Kontrastbild und liefert eine
  Kandidatenmaske. Sonst nichts.
* **Die Nachbearbeitung** (``postprocess.py``) räumt auf, füllt Löcher
  und trennt zusammengewachsene Nester - verfahrensunabhängig und für alle gleich.
* **Das Zusammenführen** (``merge.py``) nimmt als Letztes wieder
  zusammen, was sich berührt oder nur durch einen schmalen Spalt getrennt ist.
* **Die Vermessung** (``poren/messung.py``) macht daraus Zahlen.
* **Die Filter** (``poren/filter.py``) entscheiden, was fachlich als Pore
  gilt - mit Begründung je verworfenem Objekt.

Ein ausgetauschter Detektor kann damit weder die Vermessung noch die Filterlogik
durcheinanderbringen; er beantwortet nur die eine Frage, für die er da ist.
"""

from __future__ import annotations

from collections.abc import Callable

from .adaptive import AdaptiveDetector
from .base import DetectionInput, DetectionResult, PoreDetector
from .local_contrast import LocalContrastDetector
from .threshold import ThresholdDetector
from .untergrund import contrast_image, estimate_background

#: Name -> Konstruktor. Die Konfiguration wählt über ``pore.method`` aus.
_DETECTORS: dict[str, Callable[[], PoreDetector]] = {
    "local_contrast": LocalContrastDetector,
    "threshold": ThresholdDetector,
    "adaptive": AdaptiveDetector,
}


def get_detector(name: str) -> PoreDetector:
    """Der benannte Detektor. Ein unbekannter Name ist ein Fehler, keine Notlösung."""
    factory = _DETECTORS.get(name)
    if factory is None:
        raise KeyError(
            f"Unbekannter Porendetektor {name!r} - bekannt sind: "
            f"{', '.join(sorted(_DETECTORS))}"
        )
    return factory()


__all__ = ["DetectionInput", "DetectionResult", "contrast_image", "estimate_background",
           "get_detector"]
