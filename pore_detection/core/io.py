"""Find and load the images of one ID folder."""

import re
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


def load_image(path: Path, image_id: str) -> tuple[ImageInfo, np.ndarray]:
    """Load an image as RGB (H, W, 3) in its source dtype (uint8 / uint16).

    Gray sources are copied to 3 channels, an alpha channel is dropped.
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
        name=f"{image_id}_{path.stem}",
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
