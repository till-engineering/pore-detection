"""SpecimenSegmenter-Protokoll und das Ergebnis der Trennung.

Die Probenmaske ist zugleich die Bezugsfläche der Porosität - ohne sie ist der
Prozentwert vom Harzanteil dominiert und bedeutungslos. Sie ist damit nicht bloß eine
Zwischenmaske, sondern ein Messergebnis, und wird wie eines behandelt: mit Kennzahlen,
Warnungen und den Zwischenschritten, aus denen sie entstanden ist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..einstellungen import Abschnitt


@dataclass(frozen=True)
class SpecimenMask:
    """Was die Trennung ergeben hat.

    ``specimen`` und ``resin`` schließen einander aus, ergänzen sich aber nicht
    zwangsläufig zum ganzen Bild: verworfene Probenfetzen gehören zu keinem von beiden
    und fallen aus der Auswertung.

    **Poren gehören zur Probe.** Sie sind von Metall umschlossen und liegen damit
    innerhalb der Maske - das ist beabsichtigt, denn die Bezugsfläche der Porosität
    schließt die Poren ein.
    """

    specimen: np.ndarray                 # bool, True = Probe (Poren eingeschlossen)
    resin: np.ndarray                    # bool, True = Einbettmittel
    method: str
    components: int = 1                  # Anzahl getrennter Probenstücke
    warnings: tuple[str, ...] = field(default_factory=tuple)
    debug: dict[str, np.ndarray] = field(default_factory=dict)

    # -- Kennzahlen --------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.specimen.shape[0]), int(self.specimen.shape[1]))

    @property
    def total_px(self) -> int:
        return int(self.specimen.size)

    @property
    def specimen_area_px(self) -> int:
        return int(self.specimen.sum())

    @property
    def resin_area_px(self) -> int:
        return int(self.resin.sum())

    @property
    def specimen_frac(self) -> float:
        return self.specimen_area_px / self.total_px if self.total_px else 0.0

    @property
    def resin_frac(self) -> float:
        return self.resin_area_px / self.total_px if self.total_px else 0.0

    @property
    def has_resin(self) -> bool:
        return self.resin_area_px > 0

    # -- Anwendung ---------------------------------------------------------------------

    def apply(self, image: np.ndarray, fill: int = 0) -> np.ndarray:
        """Alles außerhalb der Probe auf ``fill`` setzen - für Anschauung und Export."""
        out = image.copy()
        outside = ~self.specimen
        if out.ndim == 3:
            out[outside] = fill
        else:
            out[outside] = fill
        return out

    def describe(self) -> str:
        """Einzeiler für Log und Konsole."""
        parts = [
            f"Probe {self.specimen_frac:.1%}",
            f"Harz {self.resin_frac:.1%}",
            f"[{self.method}",
        ]
        if self.components != 1:
            parts.append(f"{self.components} Stücke")
        parts[-1] = parts[-1] + "]"
        return "  ".join(parts)


@runtime_checkable
class SpecimenSegmenter(Protocol):
    """Trennt die Probe vom Einbettmittel.

    Bekommt das Graustufenbild, nicht den Pipeline-Kontext: die Trennung hängt an nichts
    anderem. Die Maßstabserkennung spielt hier bewusst keine Rolle - viele Schliffbilder
    kommen ganz ohne Maßstabsbalken an, und ein Modul, das dann nicht arbeiten könnte,
    wäre in der Praxis unbrauchbar.
    """

    name: str

    def segment(self, gray: np.ndarray, cfg: Abschnitt) -> SpecimenMask:
        """Liefert die Probenmaske. Findet sich kein Harz, ist das ganze Bild Probe."""
        ...
