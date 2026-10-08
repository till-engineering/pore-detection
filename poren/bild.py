"""Bilder laden (jpg/png/tif), Graustufen/Farbe, Bit-Tiefe, Bildliste eines Ordners.

Zwei Windows-Eigenheiten sind hier bewusst behandelt:

* ``cv2.imread`` scheitert an Pfaden mit Umlauten - es reicht die Datei roh an eine
  ANSI-API weiter. Der Testordner heißt "Maßstäbe_Bilder", der Fehler wäre also sofort
  da. Gelesen wird deshalb über ``np.fromfile`` und ``cv2.imdecode``.
* Bilder mit mehr als 8 Bit werden auf 8 Bit normiert, damit alle Schwellen im Rest des
  Programms denselben Wertebereich meinen. 16-Bit-Dateien werden nach ihrer tatsächlichen
  Bittiefe (12, 14 oder 16 Bit) umgerechnet, nicht pauschal durch 257 geteilt.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Endungen, die als Bild gelten.
DEFAULT_PATTERNS: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


@dataclass(frozen=True)
class LoadedImage:
    """Ein geladenes Bild in beiden Formen, die das Programm braucht."""

    path: Path
    gray: np.ndarray            # uint8, 2D - Grundlage jeder Messung
    color: np.ndarray | None    # BGR uint8, falls farbig - nur für Kontrollbilder

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    @property
    def name(self) -> str:
        return self.path.name

    def as_bgr(self) -> np.ndarray:
        """Farbbild zum Hineinzeichnen - notfalls aus den Graustufen aufgebaut."""
        if self.color is not None:
            return self.color.copy()
        return cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)


class ImageReadError(OSError):
    """Die Datei ließ sich nicht als Bild lesen."""


def read_image(path: str | Path) -> LoadedImage:
    """Ein Bild laden. Wirft :class:`ImageReadError`, wenn das nicht gelingt."""
    path = Path(path)
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"{path} ließ sich nicht lesen: {exc}") from exc
    if raw.size == 0:
        raise ImageReadError(f"{path} ist leer")

    decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ImageReadError(f"{path} ist kein lesbares Bildformat")

    decoded = _to_uint8(decoded)

    if decoded.ndim == 2:
        return LoadedImage(path=path, gray=decoded, color=None)

    if decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
    if decoded.shape[2] != 3:
        raise ImageReadError(f"{path}: unerwartet {decoded.shape[2]} Kanäle")

    gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    # Ein Graustufenbild, das als RGB gespeichert wurde, bleibt ein Graustufenbild.
    is_gray = bool(
        np.array_equal(decoded[:, :, 0], decoded[:, :, 1])
        and np.array_equal(decoded[:, :, 1], decoded[:, :, 2])
    )
    return LoadedImage(path=path, gray=gray, color=None if is_gray else decoded)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Auf 8 Bit bringen, ohne den Kontrast zu verändern."""
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        # Viele Mikroskopkameras speichern 12 oder 14 Bit in einer 16-Bit-Datei. Durch
        # 257 geteilt würde ein 12-Bit-Bild (bis 4095) zu 0…15 - fast schwarz, mit 16
        # Graustufen, und jede absolute Schwelle (Kastenweiß 255) ginge ins Leere.
        # Die Bittiefe wird deshalb am größten Wert erkannt: bis 255 sind es 8-Bit-Daten
        # in einer 16-Bit-Datei, bis 4095 gilt 12 Bit, bis 16383 gilt 14 Bit, darüber 16.
        hoechster = int(image.max()) if image.size else 0
        verschiebung = (0 if hoechster < 256 else 4 if hoechster < 4096
                        else 6 if hoechster < 16384 else 8)
        return (image >> verschiebung).astype(np.uint8)
    info_max = float(np.nanmax(image)) if image.size else 1.0
    if info_max <= 0:
        return np.zeros(image.shape, dtype=np.uint8)
    if image.dtype in (np.float32, np.float64) and info_max <= 1.0:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def find_images(
    root: str | Path,
    patterns: Iterable[str] = DEFAULT_PATTERNS,
    recursive: bool = False,
) -> list[Path]:
    """Bilddateien unter ``root``, sortiert. Eine einzelne Datei liefert sich selbst."""
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"{root} existiert nicht")

    found: set[Path] = set()
    for pattern in patterns:
        found.update(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted(p for p in found if p.is_file())


def iter_images(
    root: str | Path,
    patterns: Iterable[str] = DEFAULT_PATTERNS,
    recursive: bool = False,
) -> Iterator[LoadedImage]:
    """Wie :func:`find_images`, lädt aber gleich. Unlesbare Dateien werden übersprungen."""
    for path in find_images(root, patterns, recursive):
        try:
            yield read_image(path)
        except ImageReadError as exc:
            log.warning("%s wird übersprungen: %s", path.name, exc)
