"""Gemeinsame Fixtures: Testbilder, Beispielkonfiguration, temporärer Run-Store."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Die eingemessenen Maßstabsbilder. Sie tragen die ImageJ-Kalibrierung in den TIFF-Tags
#: und sind damit der einzige Datensatz, bei dem die Wahrheit für echte Bilder bekannt ist.
SCALE_IMAGE_DIR = REPO_ROOT / "test" / "Maßstäbe_Bilder"

#: Schliffe mit Einbettmittel am Rand.
RESIN_IMAGE_DIR = REPO_ROOT / "test" / "Einbett_Material"

#: Dieselben Aufnahmen wie die Maßstabsbilder, aber ohne Overlay - und vor allem **ohne
#: Einbettmittel**. Die Gegenprobe: hier darf keines gefunden werden.
NO_RESIN_IMAGE_DIR = REPO_ROOT / "test" / "Orginal_Bilder"


def _images(folder: Path, suffix: str = "*") -> list[Path]:
    import pytest as _pytest

    if not folder.is_dir():
        _pytest.skip(f"Testbilder fehlen: {folder}")
    paths = sorted(p for p in folder.glob(suffix) if p.is_file())
    if not paths:
        _pytest.skip(f"Keine Bilder in {folder}")
    return paths


@pytest.fixture(scope="session")
def resin_images() -> list[Path]:
    """Schliffe mit Einbettmittel."""
    return _images(RESIN_IMAGE_DIR)


@pytest.fixture(scope="session")
def no_resin_images() -> list[Path]:
    """Schliffe ohne Einbettmittel - die Gegenprobe."""
    return _images(NO_RESIN_IMAGE_DIR)


@pytest.fixture(scope="session")
def scale_images() -> list[Path]:
    """Alle eingemessenen Maßstabsbilder, sortiert."""
    if not SCALE_IMAGE_DIR.is_dir():
        pytest.skip(f"Testbilder fehlen: {SCALE_IMAGE_DIR}")
    paths = sorted(SCALE_IMAGE_DIR.glob("*.tif"))
    if not paths:
        pytest.skip(f"Keine TIFFs in {SCALE_IMAGE_DIR}")
    return paths


@pytest.fixture(scope="session")
def resolver():
    """Ein Resolver mit Standardkonfiguration - das OCR-Modell wird nur einmal geladen."""
    from poredet.scale.resolver import ScaleResolver

    return ScaleResolver()
