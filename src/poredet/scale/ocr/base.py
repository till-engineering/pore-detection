"""OcrEngine-Protokoll, OcrResult und die Aufbereitung des Ausschnitts.

Die Aufbereitung steht hier und nicht in den einzelnen Engines, weil sie über den Erfolg
mehr entscheidet als die Wahl der Engine. Drei Dinge sind dabei wesentlich:

* **Der Ausschnitt bleibt im Kasten.** Der Rand um die Textzeile wird am Kastenrand
  abgeschnitten. Sonst rutscht Gefüge ins Bild, und die OCR liest Strukturen als Zeichen.
* **Die Zeile wird auf eine feste Höhe skaliert.** Ein fester Vergrößerungsfaktor wäre
  schädlich - eine 8 px hohe Zeile braucht mehr Vergrößerung als eine 20 px hohe.
* **Es gibt eine weiße Ruhezone.** Ohne Rand liest die OCR aus "5 mm" gerne nur "mm";
  das ist der häufigste Ausfall überhaupt, und er fällt nicht auf, weil das Ergebnis
  plausibel aussieht.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("Die OCR-Aufbereitung benötigt opencv-python") from exc

from ...config.schema import OcrConfig
from ...core.models import BBox


@dataclass(frozen=True)
class OcrResult:
    """Ergebnis einer OCR auf einem kleinen Ausschnitt."""

    text: str
    confidence: float
    engine: str

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    @classmethod
    def empty(cls, engine: str) -> OcrResult:
        return cls(text="", confidence=0.0, engine=engine)


@runtime_checkable
class OcrEngine(Protocol):
    """Liest Text aus einem kleinen, bereits freigestellten Bildausschnitt."""

    name: str

    @property
    def available(self) -> bool:
        """``False``, wenn die Engine nicht einsatzbereit ist (Paket/Binary fehlt).

        Eine fehlende Engine ist kein Fehler, sondern ein Grund, die nächste zu nehmen.
        """
        ...

    def read(self, patch: np.ndarray) -> OcrResult:
        """Liest ``patch`` (Graustufen, uint8). Wirft nicht - liefert notfalls leer."""
        ...


def prepare_patch(
    gray: np.ndarray,
    label: BBox,
    box: BBox,
    cfg: OcrConfig,
    bar: BBox | None = None,
) -> np.ndarray:
    """Textzeile freistellen, vergrößern und mit weißer Ruhezone umgeben.

    ``bar`` wird dabei übermalt. Das ist kein Feinschliff: liegt der
    Balken mit im Ausschnitt, liest die OCR ihn als Zeichen mit - aus "47 km" wurde an
    den Testbildern "471 km", und der Maßstab war um den Faktor zehn daneben. Ein Strich
    über einer Zeile sieht nun einmal aus wie ein Zeichen.
    """
    pad = round(cfg.pad_factor * label.h)
    # Erst großzügig umranden, dann hart auf den Kasten begrenzen. Ein Pixel Abstand zum
    # Kastenrand, damit keine angeschnittene Randzeile mitkommt.
    inner = box.padded(-1)
    x0 = max(inner.x, label.x - pad)
    y0 = max(inner.y, label.y - pad)
    x1 = min(inner.x2, label.x2 + pad)
    y1 = min(inner.y2, label.y2 + pad)

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return np.full((1, 1), 255, dtype=np.uint8)
    patch = patch.copy()

    # Der Balken wird übermalt, nicht weggeschnitten. Wegschneiden würde den Ausschnitt
    # eng um die Zeile legen, und genau davon lebt die Texterkennung nicht: sie braucht
    # Luft. Übermalen nimmt den Störer heraus und lässt den Rand stehen.
    if bar is not None:
        _whiten(patch, bar, x0, y0, margin=1)

    factor = min(cfg.max_upscale, max(1.0, cfg.target_text_height_px / max(label.h, 1)))
    if factor > 1.0:
        patch = cv2.resize(patch, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    quiet = cfg.quiet_zone_px
    if quiet > 0:
        patch = cv2.copyMakeBorder(
            patch, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
        )
    return np.ascontiguousarray(patch)


def patch_variants(
    gray: np.ndarray,
    label: BBox,
    box: BBox,
    cfg: OcrConfig,
    bar: BBox | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Mehrere Fassungen derselben Textzeile, benannt, in der Reihenfolge der Befragung.

    Erst die Zielhöhen aus der Konfiguration, dann - falls gewünscht - eine binarisierte
    Fassung der ersten. Die Reihenfolge ist die Befragungsreihenfolge: der Normalfall
    steht vorn, die Spezialfälle kommen nur zum Zug, wenn noch kein Konsens besteht.
    """
    variants: list[tuple[str, np.ndarray]] = []
    for height in cfg.variant_heights:
        variant_cfg = cfg.model_copy(update={"target_text_height_px": height})
        variants.append(
            (f"h{height}", prepare_patch(gray, label, box, variant_cfg, bar))
        )
    if cfg.binarized_variant and variants:
        variants.append((f"{variants[0][0]}-bin", binarize(variants[0][1])))
    return variants


def binarize(patch: np.ndarray) -> np.ndarray:
    """Schwarz auf Weiß, per Otsu. Nimmt der vergrößerten Schrift die weichen Ränder."""
    _threshold, out = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return out


def _whiten(patch: np.ndarray, box: BBox, offset_x: int, offset_y: int, margin: int = 0) -> None:
    """Einen Bildbereich im Ausschnitt auf Kastenweiß setzen (in-place)."""
    height, width = patch.shape[:2]
    x0 = max(0, box.x - offset_x - margin)
    y0 = max(0, box.y - offset_y - margin)
    x1 = min(width, box.x2 - offset_x + margin)
    y1 = min(height, box.y2 - offset_y + margin)
    if x1 > x0 and y1 > y0:
        patch[y0:y1, x0:x1] = 255
