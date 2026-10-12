"""Einstellungen der Zonenauswertung aus ``einstellungen_zonen.yaml`` lesen und prüfen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

STANDARD = Path(__file__).with_name("einstellungen_zonen.yaml")


@dataclass(frozen=True)
class Einstellungen:
    grenzen_pct: tuple[float, ...]
    rand_pct: float
    toleranz_pct: float
    toleranz_min_px: float
    max_winkel_grad: float
    min_punkte: int


def laden(pfad: str | Path | None = None) -> Einstellungen:
    pfad = Path(pfad) if pfad else STANDARD
    with pfad.open(encoding="utf-8") as f:
        roh = yaml.safe_load(f) or {}
    try:
        grenzen = tuple(float(g) for g in roh["grenzen_pct"])
        kante = roh["kante"]
        einst = Einstellungen(
            grenzen_pct=grenzen,
            rand_pct=float(kante["rand_pct"]),
            toleranz_pct=float(kante["toleranz_pct"]),
            toleranz_min_px=float(kante["toleranz_min_px"]),
            max_winkel_grad=float(kante["max_winkel_grad"]),
            min_punkte=int(kante["min_punkte"]),
        )
    except KeyError as exc:
        raise ValueError(f"{pfad.name}: Wert {exc} fehlt") from None
    if len(grenzen) < 2 or any(b <= a for a, b in zip(grenzen, grenzen[1:])):
        raise ValueError(f"{pfad.name}: grenzen_pct braucht mindestens zwei streng "
                         f"aufsteigende Werte, steht aber auf {list(grenzen)}")
    if not 0 <= einst.rand_pct < 50:
        raise ValueError(f"{pfad.name}: kante.rand_pct muss zwischen 0 und 50 liegen")
    return einst
