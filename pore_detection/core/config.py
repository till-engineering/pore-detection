"""Global configuration: read the config once, pass it on as an immutable object.

The config is the ONE file config*.yaml in the project folder (parent of the pore_detection
package): config.yaml, config_unet.yaml, ... – to switch, move another one from configs/ there.
None or more than one is an error naming the files; it is looked up again at every load.
It is assumed to be correct: no validation, no defaults.
"""

from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATTERN = "config*.yaml"


def config_path() -> Path:
    """The one config*.yaml in the project folder."""
    found = sorted(path for path in PROJECT_DIR.glob(CONFIG_PATTERN) if path.is_file())
    if len(found) != 1:
        names = ", ".join(path.name for path in found) or "none"
        raise FileNotFoundError(f"exactly one {CONFIG_PATTERN} is allowed in {PROJECT_DIR}, found: {names}")
    return found[0]


class Config:
    """Immutable, nested view of the config with attribute access: config.runtime.tile_size."""

    __slots__ = ("_data", "_path")

    def __init__(self, data: dict[str, Any], path: str = ""):
        object.__setattr__(self, "_data", {key: _freeze(value, _join(path, key)) for key, value in data.items()})
        object.__setattr__(self, "_path", path)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"config key '{_join(self._path, key)}' not found in the config") from None

    def __setattr__(self, key: str, value: Any):
        raise AttributeError("config is read-only")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        return list(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Plain (mutable) copy, e.g. to pass method parameters or to save config_used.yaml."""
        return {key: _thaw(value) for key, value in self._data.items()}

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def load_config() -> Config:
    """Read the config (config*.yaml) from the project folder."""
    with open(config_path(), encoding="utf-8") as file:
        return Config(yaml.safe_load(file))


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _freeze(value: Any, path: str) -> Any:
    if isinstance(value, dict):
        return Config(value, path)
    if isinstance(value, list):
        return tuple(_freeze(item, f"{path}[{i}]") for i, item in enumerate(value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
