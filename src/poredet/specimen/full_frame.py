"""Kein Einbettmittel im Bild - das ganze Bild ist Probe.

Die Rückfallebene, und keine schlechte: bei Aufnahmen aus dem Probeninneren gibt es
schlicht kein Harz im Sichtfeld. Dann ist jede Trennung ein Kunstfehler - sie kann nur
Probenmaterial wegschneiden.
"""

from __future__ import annotations

import numpy as np

from ..config.schema import SpecimenConfig
from .base import SpecimenMask


class FullFrameSegmenter:
    """Erklärt das ganze Bild zur Probe."""

    name = "full_frame"

    def segment(self, gray: np.ndarray, cfg: SpecimenConfig) -> SpecimenMask:
        shape = gray.shape[:2]
        return SpecimenMask(
            specimen=np.ones(shape, dtype=bool),
            resin=np.zeros(shape, dtype=bool),
            method=self.name,
            components=1,
        )
