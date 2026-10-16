"""Noise limit: objects smaller than min_size_px are deleted.

A technical limit, not a judgement: a few pixels are no object at all. They are deleted, not
rejected – otherwise the "rejected" layer would be full of single pixels. What may count as a
pore (size, shape) is decided by the filters, which reject with a reason.
"""

import numpy as np

from ..core import geometry
from ..core.registry import register
from ..core.step import LabelStep


@register("min_size")
class MinSize(LabelStep):
    """Deletes counted pores with fewer than min_size_px pixels."""

    params = {"min_size_px": "smallest object in px (pixel count)"}

    def process(self, ctx, pores):
        small = pores.ids[geometry.areas(pores.labels, pores.ids, pores.pixels) < self.min_size_px]
        if small.size == 0:
            return pores
        flat, values = pores.pixels
        table = np.zeros(int(values.max()) + 1, dtype=bool)
        table[small] = True
        labels = pores.labels.copy()
        labels.ravel()[flat[table[values]]] = 0
        return pores.relabeled(labels)
