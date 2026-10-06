"""Beschriftung des Maßstabs lesen: Aufbereitung des Ausschnitts und die OCR-Engines.

Die Aufbereitung entscheidet über den Erfolg mehr als die Wahl der Engine. Drei Dinge
sind dabei wesentlich:

* **Der Ausschnitt bleibt im Kasten.** Der Rand um die Textzeile wird am Kastenrand
  abgeschnitten. Sonst rutscht Gefüge ins Bild, und die OCR liest Strukturen als Zeichen.
* **Die Zeile wird auf eine feste Höhe skaliert.** Ein fester Vergrößerungsfaktor wäre
  schädlich - eine 8 px hohe Zeile braucht mehr Vergrößerung als eine 20 px hohe.
* **Es gibt eine weiße Ruhezone.** Ohne Rand liest die OCR aus "5 mm" gerne nur "mm";
  das ist der häufigste Ausfall überhaupt, und er fällt nicht auf, weil das Ergebnis
  plausibel aussieht.

Engines: ``rapidocr`` (reines pip-Paket, Standard) und ``tesseract`` (zweite Meinung,
braucht eine installierte Tesseract-Binary und ``pytesseract``). Eine nicht verfügbare
Engine fällt still aus der Liste - ein fehlendes Paket bricht den Lauf nicht ab.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from ..einstellungen import Abschnitt
from ..modelle import BBox

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    """Ergebnis einer OCR auf einem kleinen Ausschnitt."""

    text: str
    confidence: float
    engine: str

    @classmethod
    def empty(cls, engine: str) -> OcrResult:
        return cls(text="", confidence=0.0, engine=engine)


# --------------------------------------------------------------------------------------
# Aufbereitung
# --------------------------------------------------------------------------------------


def prepare_patch(
    gray: np.ndarray,
    label: BBox,
    box: BBox,
    cfg: Abschnitt,
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
    cfg: Abschnitt,
    bar: BBox | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Mehrere Fassungen derselben Textzeile, benannt, in der Reihenfolge der Befragung.

    Erst die Zielhöhen aus der Konfiguration, dann - falls gewünscht - eine binarisierte
    Fassung der ersten. Die Reihenfolge ist die Befragungsreihenfolge: der Normalfall
    steht vorn, die Spezialfälle kommen nur zum Zug, wenn noch kein Konsens besteht.
    """
    variants: list[tuple[str, np.ndarray]] = []
    for height in cfg.variant_heights:
        variant_cfg = cfg.kopie(target_text_height_px=height)
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


# --------------------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------------------


class RapidOcrEngine:
    """OCR über ``rapidocr-onnxruntime``. Das Modell lädt beim ersten Aufruf."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None
        self._failed = False

    @property
    def available(self) -> bool:
        if self._engine is not None:
            return True
        if self._failed:
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        except Exception as exc:  # noqa: BLE001 - fehlendes Paket ist kein Abbruchgrund
            log.warning("RapidOCR nicht verfügbar: %s", exc)
            self._failed = True
            return False
        return True

    def read(self, patch: np.ndarray) -> OcrResult:
        if not self.available:
            return OcrResult.empty(self.name)
        rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) if patch.ndim == 2 else patch
        try:
            result, _elapsed = self._engine(rgb)
        except Exception as exc:  # noqa: BLE001
            log.warning("RapidOCR ist an diesem Ausschnitt gescheitert: %s", exc)
            return OcrResult.empty(self.name)
        if not result:
            return OcrResult.empty(self.name)
        # Ergebnis ist eine Liste aus (box, text, score) - in Lesereihenfolge verbinden.
        texts = [str(item[1]) for item in result if len(item) > 1]
        scores = [float(item[2]) for item in result if len(item) > 2]
        return OcrResult(
            text=" ".join(t for t in texts if t).strip(),
            confidence=float(np.mean(scores)) if scores else 0.0,
            engine=self.name,
        )


class TesseractEngine:
    """OCR über ``pytesseract`` und die Tesseract-Binary - die zweite Meinung.

    Eine Zeichen-Whitelist wird bewusst **nicht** gesetzt: an den Testbildern wurde aus
    "151.7241 µm" damit "nm". Tesseract nutzt das Sprachmodell über die ganze Zeile.
    """

    name = "tesseract"

    def __init__(self) -> None:
        self._binary = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        self._ok: bool | None = None

    @property
    def available(self) -> bool:
        if self._ok is None:
            self._ok = False
            if self._binary:
                try:
                    import pytesseract

                    pytesseract.pytesseract.tesseract_cmd = self._binary
                    pytesseract.get_tesseract_version()
                    self._ok = True
                except Exception as exc:  # noqa: BLE001
                    log.info("Tesseract nicht nutzbar: %s", exc)
        return self._ok

    def read(self, patch: np.ndarray) -> OcrResult:
        if not self.available:
            return OcrResult.empty(self.name)
        import pytesseract

        for config in ("--psm 7", "--psm 6"):   # eine Zeile, notfalls ein Block
            try:
                text = pytesseract.image_to_string(patch, config=config).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("Tesseract ist an diesem Ausschnitt gescheitert: %s", exc)
                return OcrResult.empty(self.name)
            if text:
                # Ohne TSV-Umweg liefert Tesseract keine brauchbare Konfidenz.
                return OcrResult(text=text, confidence=0.6, engine=self.name)
        return OcrResult.empty(self.name)


#: Einmal gebaute Engines werden behalten - das OCR-Modell lädt sonst je Bild neu.
_ENGINES = {"rapidocr": RapidOcrEngine(), "tesseract": TesseractEngine()}


def available_engines(names: Iterable[str]) -> list:
    """Die genannten Engines in genau dieser Reihenfolge, soweit einsatzbereit."""
    engines = []
    for name in names:
        engine = _ENGINES.get(name)
        if engine is None:
            log.warning("Unbekannte OCR-Engine %r - bekannt sind: %s", name, ", ".join(_ENGINES))
        elif engine.available:
            engines.append(engine)
    return engines
