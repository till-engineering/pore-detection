"""Poren-Detektion in Schliffbildern. Version und öffentliche API."""

from __future__ import annotations

import os

# OpenCV meldet bei den ImageJ-TIFFs der Testbilder unbekannte private Tags (50838/50839
# sind ImageJ-ROIs). Die Warnung ist richtig und völlig folgenlos, würde aber jede
# Konsolenausgabe zumüllen. Muss vor dem ersten cv2-Import gesetzt sein.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

__version__ = "0.1.0"

__all__ = ["__version__"]
