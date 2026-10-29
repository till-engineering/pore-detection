"""Relief: pores in shaded-relief images (several light directions merged into one picture).

In such an image a pore is not dark. It is a depression lit from one side: the wall facing away
from the light is a dark shadow, the wall facing it a bright highlight, the bottom about matrix
gray. Seen along the light: matrix | shadow | bottom | highlight | matrix. A bump (dirt, a
broken-out edge) is the other way round: bright first, then dark.

Relief contrast of a pixel for a length L: half of (mean over L px ahead along the light - mean
over L px behind). Inside a pore the shadow lies behind and the highlight ahead: large and
positive (shadow 0.6 / highlight 1.4 after background_correction -> about 0.4). Matrix and the
stripes of a stitched panorama (far larger than L) -> about 0, bumps -> negative. Same unit as
the gray image, so the thresholds read like those of hysteresis. A small pore against a long L
is averaged away: the measure is computed for every length in lengths_px / lengths_um and the
largest value counts.

Light direction (core/adaptive.py, light_angle): light_angle_deg is the base. Every image checks
it on a copy shrunk by estimate_work_scale: the cross-correlation of the dark and the bright parts
(one FFT) gives the offset shadow -> highlight. Its direction is the estimate, its asymmetry
(0 = none, 1 = all pairs in one direction) the certainty. Below min_certainty, or within
angle_tolerance_deg of the base, the base is used (results stay comparable between images);
otherwise the estimate, with a warning in the log. The angle used is only logged.

Then as hysteresis: relief contrast above seed_contrast = certain pore cores, connected pixels
above grow_contrast belong to them. The pore reaches from its shadow to its highlight, i.e. its
opening at the polished surface. Dense groups stay single pores (merging is postprocessing).
Outside the sample the matrix level is assumed, so the dark resin gives no relief at the sample
edge; cores and growth stay inside the sample. Not tileable. Exactly one of lengths_px / lengths_um.
"""

import math

import cv2
import numpy as np

from ..core import adaptive
from ..core.registry import register
from ..core.step import Detector

# light angle / 90° (mod 4) -> rotation that makes the light shine to the right, rotation back
_QUARTER_TURNS = {
    0: (None, None),
    1: (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE),       # light up
    2: (cv2.ROTATE_180, cv2.ROTATE_180),                                # light to the left
    3: (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE),       # light down
}


@register("relief")
class Relief(Detector):
    """Depressions in shaded relief: dark before, bright after along the light direction."""

    params = {
        "light_angle_deg": "base direction the light shines in (shadow -> highlight of a pore) in "
                           "degrees, 0 = to the right, 90 = up (as the image is shown)",
        "angle_tolerance_deg": "estimate within this many degrees of the base -> the base is used",
        "min_certainty": "estimate certainty 0..1 below it -> the base is used (e.g. 0.05; pure noise 0)",
        "estimate_work_scale": "the light direction is estimated on a copy shrunk by this factor",
        "lengths_px": "list of lengths in px averaged before / after a pixel (or lengths_um)",
        "lengths_um": "list of lengths in µm (or lengths_px; needs a scale), e.g. [1, 3, 8, 20]",
        "seed_contrast": "relief contrast (gray 0..1) above it = certain pore core",
        "grow_contrast": "a core grows over connected pixels above this relief contrast (< seed_contrast)",
    }

    def detect(self, ctx, image):
        if hasattr(self, "lengths_um"):
            lengths = [length / ctx.pixel_size_um for length in self.lengths_um]
        else:
            lengths = list(self.lengths_px)
        lengths = sorted({max(1, round(length)) for length in lengths})
        level = np.float32(adaptive.matrix_level(image, ctx.roi_mask))
        image = np.where(ctx.roi_mask, image, level)
        angle = adaptive.light_angle(self, ctx, image, level, lengths[-1])
        contrast = self._relief_contrast(image, angle, lengths)
        seeds = (contrast > self.seed_contrast) & ctx.roi_mask
        candidates = (contrast > self.grow_contrast) & ctx.roi_mask
        return adaptive.grow(seeds, candidates)

    @staticmethod
    def _relief_contrast(image, angle_deg, lengths):
        """Largest relief contrast over all lengths, (H, W) float32. The image is turned so the light
        shines to the right (exactly for multiples of 90°), measured along the rows, turned back."""
        quarter = angle_deg / 90
        exact = abs(quarter - round(quarter)) < 1e-6
        if exact:
            turn, back = _QUARTER_TURNS[round(quarter) % 4]
            turned = image if turn is None else cv2.rotate(image, turn)
        else:
            height, width = image.shape
            dx, dy = math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))
            size = (round(abs(dx) * width + abs(dy) * height),
                    round(abs(dy) * width + abs(dx) * height))
            rotation = np.array([[dx, dy], [-dy, dx]])           # light direction (dx, dy) -> (1, 0)
            centre = np.array([(width - 1) / 2, (height - 1) / 2])
            shift = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2]) - rotation @ centre
            matrix = np.hstack([rotation, shift[:, None]])
            turned = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # mean over the L + 1 px from x on (x itself cancels in the difference); the mean behind x is
        # the same window started L px earlier, so one box filter per length (padded once for that)
        pad = lengths[-1]
        padded = cv2.copyMakeBorder(turned, 0, 0, pad, pad, cv2.BORDER_REPLICATE)
        width_turned = turned.shape[1]
        contrast = None
        for length in lengths:
            mean = cv2.blur(padded, (length + 1, 1), anchor=(0, 0), borderType=cv2.BORDER_REPLICATE)
            difference = cv2.subtract(mean[:, pad:pad + width_turned], mean[:, pad - length:pad - length + width_turned])
            difference *= np.float32((length + 1) / (2 * length))
            contrast = difference if contrast is None else np.maximum(contrast, difference, out=contrast)
        if exact:
            return contrast if back is None else cv2.rotate(contrast, back)
        return cv2.warpAffine(contrast, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
