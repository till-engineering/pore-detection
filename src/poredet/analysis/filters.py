"""Poren-Filter als Plugins, in konfigurierter Reihenfolge.

Hier wird entschieden, was fachlich als Pore **gilt** - im Unterschied zur Rauschgrenze
der Detektion, die nur Einzelpixel wegwirft. Der Unterschied ist wichtig: die Rauschgrenze
ist eine technische Notwendigkeit, diese Filter sind eine fachliche Festlegung, und
fachliche Festlegungen gehören dokumentiert und begründet.

**Jede verworfene Pore behält ihren Verwerfungsgrund.** Das ist die Grundlage für die
häufigste Frage am Kontrollbild - "warum fehlt diese Pore?" - und ohne den Grund lässt
sie sich nur durch erneutes Durchrechnen beantworten. Die verworfenen Objekte werden
deshalb nicht gelöscht, sondern mitgeführt und im Kontrollbild grau eingezeichnet.

Ein neuer Filter ist eine Funktion mit ``@register`` und ein Eintrag in der
Konfiguration - am Code ändert sich nichts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..config.schema import AnalysisConfig, FilterSpec
from ..core.models import Pore, RejectedPore


@dataclass(frozen=True)
class FilterContext:
    """Was ein Filter über das Bild wissen darf."""

    image_area_px: int
    specimen_area_px: int
    um_per_px: float | None


#: Ein Filter gibt ``None`` zurück, wenn die Pore bleiben darf, sonst den Grund.
FilterFn = Callable[[Pore, FilterContext, dict[str, float]], str | None]

_FILTERS: dict[str, FilterFn] = {}


def register(name: str) -> Callable[[FilterFn], FilterFn]:
    """Einen Filter unter einem Namen bekannt machen."""

    def decorator(fn: FilterFn) -> FilterFn:
        _FILTERS[name] = fn
        return fn

    return decorator


def available() -> list[str]:
    return sorted(_FILTERS)


# --------------------------------------------------------------------------------------
# Die mitgelieferten Filter
# --------------------------------------------------------------------------------------


@register("min_area")
def _min_area(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Rauschgrenze in **Pixeln**.

    Aus demselben Grund in Pixeln wie bei ``min_diameter``: gemeint ist, ab wann ein
    Häufchen Pixel überhaupt ein Objekt ist, und das entscheidet die Abtastung - nicht
    der Maßstab. Eine Flächengrenze in µm² wäre eine Vorschrift und gehört nach
    :mod:`poredet.analysis.acceptance`.

    Der Filter liegt neben ``min_diameter`` und nicht statt seiner: der Durchmesser
    fängt das Dünne ab, die Fläche das Kleine. Ein zwei Pixel breiter, zwanzig Pixel
    langer Strich kommt über die Flächengrenze und bleibt trotzdem unmessbar.
    """
    grenze_px = params.get("min_area_px", 0.0)
    if pore.area_px < grenze_px:
        text = f"Fläche {pore.area_px:.0f} px unter {grenze_px:g} px"
        if pore.area_um2 is not None:
            text += f" ({pore.area_um2:.1f} µm²)"
        return text
    return None


@register("max_area")
def _max_area(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Obergrenze als Anteil der Probenfläche - gegen großflächige Fehltreffer."""
    anteil = params.get("max_area_frac", 1.0)
    if ctx.specimen_area_px <= 0:
        return None
    if pore.area_px / ctx.specimen_area_px > anteil:
        return f"Fläche {pore.area_px / ctx.specimen_area_px:.1%} der Probe über {anteil:.0%}"
    return None


@register("aspect_ratio")
def _aspect_ratio(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Gegen Kratzer und Schleifriefen.

    Sie sind dunkel wie Poren, aber langgestreckt. Der Wert ist bewusst großzügig: auch
    Lunker können deutlich länglich sein, und lieber eine Riefe zu viel als ein Lunker zu
    wenig - die Riefe fällt im Kontrollbild auf, die fehlende Pore nicht.
    """
    grenze = params.get("max_aspect_ratio", float("inf"))
    if pore.aspect_ratio > grenze:
        return f"Seitenverhältnis {pore.aspect_ratio:.1f} über {grenze:g}"
    return None


@register("circularity")
def _circularity(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Gegen zerklüftete Gefügebestandteile. Vorsicht: Lunker sind selten rund."""
    grenze = params.get("min_circularity", 0.0)
    if pore.circularity < grenze:
        return f"Rundheit {pore.circularity:.2f} unter {grenze:g}"
    return None


@register("solidity")
def _solidity(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Anteil an der eigenen konvexen Hülle - trennt kompakte von fransigen Objekten."""
    grenze = params.get("min_solidity", 0.0)
    if pore.solidity < grenze:
        return f"Solidität {pore.solidity:.2f} unter {grenze:g}"
    return None


@register("contrast")
def _contrast(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Trennt tiefe Hohlräume von dunklen Gefügephasen.

    Eine Pore ist ein Loch - sie wirft praktisch kein Licht zurück. Eine dunkle
    Zweitphase ist dunkles Material und hebt sich deutlich weniger vom Untergrund ab.
    """
    grenze = params.get("min_contrast", 0.0)
    if pore.contrast < grenze:
        return f"Kontrast {pore.contrast:.0f} Graustufen unter {grenze:g}"
    return None


@register("intensity")
def _intensity(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Absoluter Grauwert als Obergrenze - nur sinnvoll bei gleichmäßiger Ausleuchtung."""
    grenze = params.get("max_mean_intensity", 255.0)
    if pore.mean_intensity > grenze:
        return f"mittlerer Grauwert {pore.mean_intensity:.0f} über {grenze:g}"
    return None


# --------------------------------------------------------------------------------------
# Anwendung
# --------------------------------------------------------------------------------------


def apply(
    pores: list[Pore], cfg: AnalysisConfig, ctx: FilterContext
) -> tuple[list[Pore], list[RejectedPore]]:
    """Alle aktiven Filter der Reihe nach anwenden.

    Eine Pore wird beim **ersten** Filter verworfen, der greift - der Grund ist damit
    eindeutig einem Kriterium zugeordnet und nicht eine Sammlung von Beanstandungen.
    """
    aktive: list[FilterSpec] = [f for f in cfg.filters if f.enabled]
    unbekannt = [f.name for f in aktive if f.name not in _FILTERS]
    if unbekannt:
        raise KeyError(
            f"Unbekannte Filter: {', '.join(unbekannt)} - bekannt sind: "
            f"{', '.join(available())}"
        )

    behalten: list[Pore] = []
    verworfen: list[RejectedPore] = []
    for pore in pores:
        for spec in aktive:
            grund = _FILTERS[spec.name](pore, ctx, spec.params)
            if grund is not None:
                verworfen.append(RejectedPore(pore=pore, filter_name=spec.name, reason=grund))
                break
        else:
            behalten.append(pore)
    return behalten, verworfen
