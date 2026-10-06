"""PoreDetector-Protokoll: Kontext rein, Label-Bild raus.

Ein Detektor bekommt das Graubild, die **Probenmaske** und den geschätzten Untergrund -
und liefert ein Label-Bild. Mehr nicht. Alles, was danach kommt (aufräumen, trennen,
vermessen, filtern), ist verfahrensunabhängig und steht woanders.

Diese enge Schnittstelle ist Absicht. Ein neues Verfahren - ein anderer Schwellwert, ein
lokal adaptives Filter, später ein U-Net - ist damit eine Datei und ein Eintrag in ``_DETECTORS``,
und es kann weder die Vermessung noch die Filterlogik durcheinanderbringen.

**Die Maske ist bindend.** Außerhalb von ``specimen`` darf kein Detektor etwas finden.
Dort liegt Einbettmittel oder das eingebrannte Maßstabs-Overlay, und beides ist kein
Gefüge. Die Schwelle wird deshalb auch **nur über Pixel innerhalb der Maske** bestimmt -
eine globale wäre vom dunklen Harz dominiert und fände im Metall nichts mehr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..einstellungen import Abschnitt


@dataclass(frozen=True)
class DetectionInput:
    """Alles, was ein Detektor über das Bild wissen muss."""

    gray: np.ndarray             # uint8, das (unveränderte) Graubild
    specimen: np.ndarray         # bool, True = auswertbare Probenfläche
    background: np.ndarray       # float32, geschätzter Untergrund
    contrast: np.ndarray         # float32, background - gray, also "wie dunkel gegenüber der Umgebung"

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.gray.shape[0]), int(self.gray.shape[1]))

    def inside(self, values: np.ndarray) -> np.ndarray:
        """Die Werte innerhalb der Probenmaske - Grundlage jeder Schwellenbestimmung."""
        return values[self.specimen]


@dataclass(frozen=True)
class DetectionResult:
    """Was ein Detektor zurückgibt."""

    mask: np.ndarray                          # bool, Porenkandidaten
    method: str
    thresholds: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class PoreDetector(Protocol):
    """Findet Porenkandidaten innerhalb der Probenmaske."""

    name: str

    def detect(self, data: DetectionInput, cfg: Abschnitt) -> DetectionResult:
        """Liefert die Kandidatenmaske. Nachbearbeitung ist nicht seine Aufgabe."""
        ...
