"""PipelineContext: Transportbehälter zwischen den Stufen.

Hält Originalbild, Masken, Maßstab, Zwischenergebnisse und Warnungen. Jede Stufe liest
daraus und schreibt hinein; sie kennen einander nicht.

Der Kontext ist zugleich die Antwort auf "warum sieht das Ergebnis so aus?". Er behält
die Zwischenmasken - Overlay-Ausschluss, Probenmaske, Kontrastbild, Kandidaten vor dem
Filtern -, weil sich die Frage anders nicht beantworten lässt. Ein Kontrollbild, das nur
das Endergebnis zeigt, sagt nicht, an welcher Stufe etwas verlorenging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import Pore, RejectedPore, ScaleInfo


@dataclass
class PipelineContext:
    """Alles, was während der Auswertung eines Bildes entsteht."""

    path: Path
    gray: np.ndarray
    color: np.ndarray | None = None

    # -- Stufe 1: Maßstab ---------------------------------------------------------------
    scale: ScaleInfo | None = None
    excluded: np.ndarray | None = None        # bool, eingebranntes Overlay

    # -- Stufe 2: Probe -----------------------------------------------------------------
    specimen: np.ndarray | None = None        # bool, auswertbare Probenfläche
    resin: np.ndarray | None = None           # bool, Einbettmittel

    # -- Stufe 3: Poren -----------------------------------------------------------------
    background: np.ndarray | None = None      # float32, geschätzter Untergrund
    contrast: np.ndarray | None = None        # float32, background - gray
    candidates: np.ndarray | None = None      # bool, vor dem Aufräumen
    labels: np.ndarray | None = None          # int32, nummerierte Poren

    # -- Stufe 4: Messwerte -------------------------------------------------------------
    pores: list[Pore] = field(default_factory=list)
    rejected: list[RejectedPore] = field(default_factory=list)

    # -- Protokoll ----------------------------------------------------------------------
    warnings: list[str] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.gray.shape[0]), int(self.gray.shape[1]))

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    @property
    def specimen_area_px(self) -> int:
        return 0 if self.specimen is None else int(self.specimen.sum())

    @property
    def excluded_area_px(self) -> int:
        return 0 if self.excluded is None else int(self.excluded.sum())

    @property
    def um_per_px(self) -> float | None:
        return None if self.scale is None else self.scale.um_per_px

    def note(self, stage: str, text: str) -> None:
        """Festhalten, was eine Stufe getan hat - erscheint in Konsole und Kontrollbild."""
        self.stages[stage] = text

    def warn(self, text: str) -> None:
        self.warnings.append(text)
