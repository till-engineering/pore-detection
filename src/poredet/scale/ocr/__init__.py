"""OCR-Adapter hinter einem gemeinsamen Protokoll.

Engines werden über ihren Namen ausgewählt, nicht importiert - die Konfiguration
entscheidet, nicht der Code. Eine nicht installierte Engine fällt still aus der Liste,
damit ein fehlendes Paket den Lauf nicht abbricht.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from .base import OcrEngine, OcrResult, binarize, patch_variants, prepare_patch

log = logging.getLogger(__name__)

#: Name -> Konstruktor. Der Import passiert erst beim Bauen, damit ein fehlendes
#: optionales Paket nicht schon beim Import dieses Moduls auffällt.
_FACTORIES: dict[str, Callable[[], OcrEngine]] = {}


def register_engine(name: str, factory: Callable[[], OcrEngine]) -> None:
    """Eine weitere OCR bekannt machen."""
    _FACTORIES[name] = factory


def _default_factories() -> None:
    def rapid() -> OcrEngine:
        from .rapidocr_engine import RapidOcrEngine

        return RapidOcrEngine()

    def tess() -> OcrEngine:
        from .tesseract_engine import TesseractEngine

        return TesseractEngine()

    register_engine("rapidocr", rapid)
    register_engine("tesseract", tess)


_default_factories()

#: Einmal gebaute Engines werden behalten - das OCR-Modell lädt sonst je Bild neu.
_INSTANCES: dict[str, OcrEngine] = {}


def get_engine(name: str) -> OcrEngine | None:
    """Eine benannte Engine, oder ``None``, wenn sie unbekannt oder nicht nutzbar ist."""
    if name in _INSTANCES:
        return _INSTANCES[name]
    factory = _FACTORIES.get(name)
    if factory is None:
        log.warning("Unbekannte OCR-Engine %r - bekannt sind: %s",
                    name, ", ".join(sorted(_FACTORIES)))
        return None
    try:
        engine = factory()
    except Exception as exc:  # pragma: no cover - defekte Installation
        log.warning("OCR-Engine %r ließ sich nicht erzeugen: %s", name, exc)
        return None
    _INSTANCES[name] = engine
    return engine


def available_engines(names: Iterable[str]) -> list[OcrEngine]:
    """Die genannten Engines in genau dieser Reihenfolge, soweit einsatzbereit."""
    engines = []
    for name in names:
        engine = get_engine(name)
        if engine is not None and engine.available:
            engines.append(engine)
    return engines


__all__ = [
    "OcrEngine",
    "OcrResult",
    "available_engines",
    "binarize",
    "get_engine",
    "patch_variants",
    "prepare_patch",
    "register_engine",
]
