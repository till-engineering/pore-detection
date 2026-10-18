"""Embedding medium removal as in ImageJ's standard workflow.

    Image > Type > 8-bit                    unweighted RGB conversion (R + G + B) / 3
    Image > Adjust > Threshold, "Default"   IJ_IsoData, the specimen is the bright part
    Process > Binary > Fill Holes           pores inside the specimen belong to it
    + only the largest part is kept         one specimen per image (as sample.frame assumes)

No smoothing, no size filter – these three steps, then the largest connected part. Other bright
parts in the embedding medium are dropped: copies or snippets of the scale overlay in panoramas
(only the read one is removed by sample.scale; filled, their bar and text would count as pores),
dust. Very fast (a few passes over the image). Pores cut open by the specimen border are connected to the resin and are therefore
not filled: they lie outside the ROI.

Self-contained (imports only core). CPU only; sample methods are not tiled and have no GPU path.
"""

import cv2
import numpy as np

from ..core.registry import register
from ..core.step import RoiDetector


@register("imagej")
class ImageJ(RoiDetector):
    """Specimen = bright part after ImageJ's "Default" auto threshold, holes filled, largest part."""

    params = {}

    def detect_roi(self, info, original):
        rgb8 = original if original.dtype == np.uint8 else (original >> 8).astype(np.uint8)
        # ImageJ: unweighted, rounded; (r + g + b) / 3 never ends in .5, so cv2's rounding is exact
        gray = cv2.transform(rgb8, np.full((1, 3), 1 / 3))

        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.int64)
        level = _ij_default_threshold(histogram)
        specimen = gray > level
        return _largest_part(_fill_holes(specimen))


def _ij_default_threshold(histogram: np.ndarray) -> int:
    """ImageJ "Default" threshold (ImageProcessor.getAutoThreshold): IJ_IsoData variant.

    If the most frequent gray value is more than twice as frequent as the second most
    frequent one, its count is reduced to 1.5 x the second (keeps a huge background peak
    from dominating). Then iterative IsoData; gray values 0 and 255 are ignored.
    """
    data = histogram.copy()
    data[0] = data[-1] = 0           # as in ImageJ: ends are ignored for the mode search too
    mode = int(np.argmax(data))
    second = int(np.max(np.delete(data, mode)))
    if data[mode] > 2 * second and second != 0:
        data[mode] = int(second * 1.5)
    return _ij_isodata(data)


def _ij_isodata(data: np.ndarray) -> int:
    """ImageProcessor.IJIsoData: midpoint of the means below and above a moving split."""
    data = data.copy()
    max_value = len(data) - 1
    data[0] = 0                      # ignore 0 and 255 (erased / saturated areas)
    data[max_value] = 0
    nonzero = np.nonzero(data)[0]
    if nonzero.size < 2:
        return len(data) // 2
    low, high = int(nonzero[0]), int(nonzero[-1])

    values = np.arange(len(data), dtype=np.float64)
    weighted = np.cumsum(values * data)
    counts = np.cumsum(data).astype(np.float64)
    moving = low
    while True:
        below_sum, below_count = weighted[moving], counts[moving]
        above_sum, above_count = weighted[high] - below_sum, counts[high] - below_count
        result = (below_sum / below_count + above_sum / above_count) / 2.0 if above_count else float(moving)
        moving += 1
        if not ((moving + 1) <= result and moving < high - 1):
            break
    return int(round(result))


def _largest_part(mask: np.ndarray) -> np.ndarray:
    """Only the largest 8-connected part of the mask (empty mask stays empty)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8), connectivity=8)
    if count <= 2:                      # background + at most one part
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """ImageJ Fill Holes: background not 4-connected to the image border becomes foreground."""
    # a 1 px background frame connects all border background; one flood fill from it marks it
    padded = cv2.copyMakeBorder((~mask).view(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(padded, None, (0, 0), 2, flags=4)
    return padded[1:-1, 1:-1] != 2
