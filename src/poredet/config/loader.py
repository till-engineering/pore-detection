"""Aufloesung: config/einstellungen.yaml -> Lauf-Overrides -> AppConfig.

Die globale Einstellungsdatei ist ``config/einstellungen.yaml``; ``config/default.yaml``
ist ihre unveraenderte Vorlage mit den Standardwerten aus :mod:`.schema`. Zurueck auf
Standard geht auf zwei Wegen: ``standardwerte_verwenden: true`` in der Datei (die
eigenen Werte bleiben stehen) oder :func:`reset_settings` (die Datei wird ersetzt).

Die aufgeloeste Konfiguration wandert in den Run-Ordner und in den Bericht -
ohne sie ist ein Lauf nicht reproduzierbar."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import yaml

from .schema import AppConfig

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]
SETTINGS_PATH = PROJECT_DIR / "config" / "einstellungen.yaml"
DEFAULT_PATH = PROJECT_DIR / "config" / "default.yaml"

#: Umgebungsvariable fuer eine andere Einstellungsdatei, etwa fuer Vergleichslaeufe.
ENV_VAR = "POREDET_CONFIG"

#: Schalter in der Datei selbst; gehoert nicht zum Schema.
DEFAULTS_SWITCH = "standardwerte_verwenden"


def settings_path() -> Path:
    """Die Einstellungsdatei, die :func:`load_config` ohne Angabe liest."""
    return Path(os.environ.get(ENV_VAR) or SETTINGS_PATH)


def load_config(path: Path | str | None = None) -> AppConfig:
    """Liest die Einstellungsdatei und prueft sie gegen das Schema.

    Fehlt die Datei, gelten die Standardwerte. Ein unbekannter Schluessel oder ein
    unzulaessiger Wert ist ein Fehler - bewusst, siehe :mod:`.schema`.
    """
    path = Path(path) if path else settings_path()
    if not path.is_file():
        log.warning("Keine Einstellungsdatei unter %s - es gelten die Standardwerte", path)
        return _anchor(AppConfig())

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.pop(DEFAULTS_SWITCH, False):
        return _anchor(AppConfig())
    return _anchor(AppConfig.model_validate(data))


def _anchor(cfg: AppConfig) -> AppConfig:
    """Relative Pfade gelten ab dem Projektordner, nicht ab dem Arbeitsverzeichnis."""
    directory = cfg.corrections.directory
    if directory.is_absolute():
        return cfg
    corrections = cfg.corrections.model_copy(update={"directory": PROJECT_DIR / directory})
    return cfg.model_copy(update={"corrections": corrections})


def reset_settings() -> Path:
    """Ersetzt die Einstellungsdatei durch die Vorlage mit den Standardwerten."""
    target = settings_path()
    shutil.copyfile(DEFAULT_PATH, target)
    return target
