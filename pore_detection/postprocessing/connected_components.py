"""Labeling: every connected group of mask pixels is one pore."""

import cv2
import numpy as np
import pandas as pd

from ..core.registry import register
from ..core.step import Labeler
from ..core.types import LabeledPores


@register("connected_components")
class ConnectedComponents(Labeler):
    """Connected mask pixels form one pore."""

    params = {"connectivity": "8 = pixels touching at a corner belong together, 4 = only edges count"}

    def label(self, ctx, mask):
        n, labels = cv2.connectedComponents(mask.view(np.uint8), connectivity=self.connectivity, ltype=cv2.CV_32S)
        count = n - 1
        return LabeledPores(labels=labels, count=count,
                            properties=pd.DataFrame({"pore_id": np.arange(1, count + 1, dtype=np.int32)}))
