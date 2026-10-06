"""Maßstab aus dem eingebrannten Balken.

    ergebnis = ScaleResolver(cfg.scale).resolve(gray, pfad)
    if ergebnis.scale:
        durchmesser_um = ergebnis.scale.to_um(durchmesser_px)
        nicht_werten   = overlay_mask(gray.shape, ergebnis.scale, pad=4)

Ohne erkannten Maßstab ist ``ergebnis.scale`` ``None``. Dann wird in Pixeln weiter
gemessen und die physikalischen Werte bleiben leer - es wird nirgends ein Faktor
unterstellt.

Dateien:
    kasten.py            schwarzer Balken als Loch in einem weißen Kasten (ImageJ-Stil)
    geteilter_kasten.py  Balken zerschneidet den Kasten von Wand zu Wand
    ocr.py               Beschriftung lesen (RapidOCR, optional Tesseract)
    erkennung.py         Kandidaten + OCR -> Maßstab, Overlay-Maske
"""

from .erkennung import ScaleOutcome, ScaleResolver, overlay_mask

__all__ = ["ScaleOutcome", "ScaleResolver", "overlay_mask"]
