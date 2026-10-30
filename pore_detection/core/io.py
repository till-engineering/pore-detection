"""Find and load the images of one ID folder."""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .types import ImageInfo

_BIT_DEPTHS = {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16}


def image_id_of(folder: Path) -> str:
    """The ID is the name of the input folder."""
    return Path(folder).resolve().name


def find_images(folder: Path, extensions) -> list[Path]:
    """Images directly inside `folder` (no subfolders), natural order (img2 before img10)."""
    extensions = {ext.lower() for ext in extensions}
    files = [path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in extensions]
    return sorted(files, key=_natural_key)


@dataclass(frozen=True)
class Recording:
    """One recording of an ID folder: the image the pores are detected on and the image the sample
    (ROI, scale, frame) is detected on - the same file without input.views. A missing view is None."""

    name: str                   # file stem (without views) or <recording> of <recording>_<view>
    key: str                    # key of the per-image inputs: file name (without views) or name
    pores: Path | None
    sample: Path | None


def find_recordings(folder: Path, extensions, views) -> tuple[list[Recording], list[Path]]:
    """Recordings in `folder` and the files that belong to none of them.

    views None: every image is a recording. Otherwise (views.sample, views.pores): the text after
    the last "_" of a file stem is its view, the text before it the recording; files of any other
    view are returned as ignored. Natural order of the recordings."""
    files = find_images(folder, extensions)
    if views is None:
        return [Recording(path.stem, path.name, path, path) for path in files], []
    found: dict[str, dict[str, Path]] = {}
    ignored = []
    for path in files:
        recording, _, view = path.stem.rpartition("_")
        if recording and view in (views.sample, views.pores):
            found.setdefault(recording, {})[view] = path
        else:
            ignored.append(path)
    recordings = [Recording(name, name, paths.get(views.pores), paths.get(views.sample))
                  for name, paths in found.items()]
    return sorted(recordings, key=lambda r: _natural_key(Path(r.name))), ignored


def load_recording(recording: Recording, image_id: str, views) -> tuple[ImageInfo, np.ndarray, np.ndarray]:
    """(info, pores image, sample image); without views both images are the same array."""
    for view, path in ((views and views.pores, recording.pores), (views and views.sample, recording.sample)):
        if path is None:
            raise ValueError(f"missing view {view} ({recording.name}_{view})")
    info, pores = load_image(recording.pores, image_id, recording.name)
    if recording.sample == recording.pores:
        return info, pores, pores
    _, sample = load_image(recording.sample, image_id, recording.name)
    if sample.shape[:2] != pores.shape[:2]:
        raise ValueError(f"views differ in size: {views.sample} {sample.shape[1]}x{sample.shape[0]}, "
                         f"{views.pores} {pores.shape[1]}x{pores.shape[0]} px")
    return info, pores, sample


def load_image(path: Path, image_id: str, recording: str | None = None) -> tuple[ImageInfo, np.ndarray]:
    """Load an image as RGB (H, W, 3) in its source dtype (uint8 / uint16).

    Gray sources are copied to 3 channels, an alpha channel is dropped. The image is named
    <image_id>_<recording> (recording None: the file stem).
    """
    path = Path(path)
    image = _imread(path)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    if image.dtype not in _BIT_DEPTHS:
        raise ValueError(f"unsupported pixel type {image.dtype} in {path} (only 8 and 16 bit)")

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    info = ImageInfo(
        path=path,
        image_id=image_id,
        name=f"{image_id}_{recording if recording is not None else path.stem}",
        height_px=image.shape[0],
        width_px=image.shape[1],
        bit_depth=_BIT_DEPTHS[image.dtype],
        offset_px=(0, 0),
    )
    return info, image


def _imread(path: Path) -> np.ndarray | None:
    # cv2.imread cannot open non-ASCII paths on Windows; decoding from memory costs one
    # extra copy of the file, so it is only used when needed.
    if str(path).isascii():
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]
