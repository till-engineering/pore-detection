"""OCR über rapidocr-onnxruntime: rein pip-installierbar, keine externe Binary.

Das Modell wird erst beim ersten Aufruf geladen. Ein Lauf mit fest vorgegebenem Maßstab
kostet dadurch keine Ladezeit, und die Tests der Geometrie laufen ohne das Paket.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("rapidocr_engine benötigt opencv-python") from exc

from .base import OcrResult

log = logging.getLogger(__name__)


class RapidOcrEngine:
    """OCR über ``rapidocr-onnxruntime``."""

    name = "rapidocr"

    def __init__(self) -> None:
        self._engine: object | None = None
        self._failed = False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._engine is not None:
            return True
        if self._failed:
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            log.info("rapidocr-onnxruntime nicht installiert (%s)", exc)
            self._failed = True
            return False
        try:
            self._engine = RapidOCR()
        except Exception as exc:  # pragma: no cover - hängt an der Installation
            log.warning("RapidOCR ließ sich nicht initialisieren: %s", exc)
            self._failed = True
            return False
        return True

    def read(self, patch: np.ndarray) -> OcrResult:
        if not self._ensure_loaded():
            return OcrResult.empty(self.name)

        rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) if patch.ndim == 2 else patch
        try:
            result, _elapsed = self._engine(rgb)  # type: ignore[operator]
        except Exception as exc:  # pragma: no cover - Modellfehler zur Laufzeit
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
