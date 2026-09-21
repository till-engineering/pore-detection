"""OCR über Tesseract. Genauer bei sauberen Vorlagen, benötigt aber eine Binary.

Deshalb ist diese Engine nicht die erste Wahl, sondern die zweite Meinung: sie prüft das
Ergebnis von RapidOCR gegen. Fehlt die Binary, meldet sie sich schlicht als nicht
verfügbar - das ist kein Fehler, sondern ein Grund weiterzugehen.

Eine Zeichen-Whitelist wird bewusst **nicht** gesetzt. Sie liegt nahe (es kommen ja nur
Ziffern und Einheiten vor), verschlechtert das Ergebnis an den vorhandenen Testbildern
aber drastisch: aus "151.7241 µm" wird dann "nm". Tesseract nutzt das Sprachmodell über
die ganze Zeile, und eine Whitelist schneidet ihm genau das ab.
"""

from __future__ import annotations

import logging
import os
import shutil

import numpy as np

from .base import OcrResult

log = logging.getLogger(__name__)

#: Einzeilige Vorlage. Der Ausschnitt ist eine freigestellte Zeile, kein Seitenlayout.
_CONFIG = "--psm 7"
#: Zweiter Versuch, falls die Zeilenannahme nicht greift.
_CONFIG_FALLBACK = "--psm 6"


class TesseractEngine:
    """OCR über ``pytesseract`` und die Tesseract-Binary."""

    name = "tesseract"

    def __init__(self, binary: str | None = None) -> None:
        self._binary = binary or os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        self._checked = False
        self._ok = False

    @property
    def available(self) -> bool:
        if self._checked:
            return self._ok
        self._checked = True
        if not self._binary:
            log.info("Tesseract-Binary nicht gefunden - Engine wird übersprungen")
            return False
        try:
            import pytesseract
        except ImportError as exc:
            log.info("pytesseract nicht installiert (%s)", exc)
            return False
        pytesseract.pytesseract.tesseract_cmd = self._binary
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # pragma: no cover - hängt an der Installation
            log.info("Tesseract nicht aufrufbar: %s", exc)
            return False
        self._ok = True
        return True

    def read(self, patch: np.ndarray) -> OcrResult:
        if not self.available:
            return OcrResult.empty(self.name)
        import pytesseract

        for config in (_CONFIG, _CONFIG_FALLBACK):
            try:
                text = pytesseract.image_to_string(patch, config=config).strip()
            except Exception as exc:  # pragma: no cover - Laufzeitfehler der Binary
                log.warning("Tesseract ist an diesem Ausschnitt gescheitert: %s", exc)
                return OcrResult.empty(self.name)
            if text:
                # Tesseract liefert ohne aufwändigen TSV-Umweg keine brauchbare
                # Konfidenz; der Wert bleibt bewusst konservativ.
                return OcrResult(text=text, confidence=0.6, engine=self.name)
        return OcrResult.empty(self.name)
