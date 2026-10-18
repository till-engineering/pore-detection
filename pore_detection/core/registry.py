"""Registration and automatic discovery of methods.

Every file in the stage folders (sample, preprocessing, ...) is imported once at start;
@register("name") files the class under its stage. config.yaml selects methods by that name.
"""

import importlib
import pkgutil
from dataclasses import dataclass

from .config import Config
from .step import (Detector, Evaluator, Filter, FrameDetector, LabelStep, Labeler, MaskStep,
                   RoiDetector, ScaleDetector, Step)

STAGE_PACKAGES = ("sample", "preprocessing", "detection", "postprocessing", "evaluation")

# base class -> stage key as used in config.yaml
STAGES: dict[type[Step], str] = {
    RoiDetector: "sample.roi",
    ScaleDetector: "sample.scale",
    FrameDetector: "sample.frame",
    Filter: "preprocessing",
    Detector: "detection",
    MaskStep: "postprocessing.mask_steps",
    Labeler: "postprocessing.labeling",
    LabelStep: "postprocessing.label_steps",
    Evaluator: "evaluation",
}

_methods: dict[str, dict[str, type[Step]]] = {stage: {} for stage in STAGES.values()}
_discovered = False


@dataclass(frozen=True)
class MethodInfo:
    """Description of one registered method, returned by list_methods()."""

    name: str
    stage: str
    description: str            # first line of the class docstring
    params: dict[str, str]      # parameter name -> description
    tileable: bool
    tile_overlap_px: int
    gpu: bool                   # has a GPU path


def register(name: str):
    """Class decorator: register a method under `name` in the stage of its base class."""

    def decorator(cls: type[Step]) -> type[Step]:
        stage = stage_of(cls)
        if name in _methods[stage]:
            raise ValueError(f"{stage} method '{name}' registered twice: "
                             f"{_methods[stage][name].__module__} and {cls.__module__}")
        cls.name = name
        _methods[stage][name] = cls
        return cls

    return decorator


def stage_of(cls: type[Step]) -> str:
    for base, stage in STAGES.items():
        if issubclass(cls, base):
            return stage
    raise TypeError(f"{cls.__name__} does not derive from a stage base class ({', '.join(b.__name__ for b in STAGES)})")


def discover() -> None:
    """Import every module of the stage packages once, so that all @register calls run."""
    global _discovered
    if _discovered:
        return
    for package_name in STAGE_PACKAGES:
        package = importlib.import_module(f"pore_detection.{package_name}")
        for module in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package.__name__}.{module.name}")
    _discovered = True


def create(stage: str, entry: Config) -> Step:
    """Instantiate the method selected by a config entry ({name: ..., <params>})."""
    discover()
    params = entry.to_dict()
    name = params.pop("name")
    try:
        cls = _methods[stage][name]
    except KeyError:
        raise ValueError(f"unknown {stage} method '{name}' "
                         f"(available: {', '.join(sorted(_methods[stage])) or 'none'})") from None
    return cls(**params)


def list_methods() -> dict[str, list[MethodInfo]]:
    """All registered methods per stage."""
    discover()
    return {
        stage: [
            MethodInfo(
                name=name,
                stage=stage,
                description=(cls.__doc__ or "").strip().split("\n")[0],
                params=dict(cls.params),
                tileable=cls.tileable,
                tile_overlap_px=cls.tile_overlap_px,
                gpu=hasattr(cls, "process_gpu") or hasattr(cls, "detect_gpu"),
            )
            for name, cls in sorted(methods.items())
        ]
        for stage, methods in _methods.items()
    }
