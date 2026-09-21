"""Aus dem Label-Bild werden Poren mit Kennwerten.

Zwei Festlegungen, die später jede Zahl im Bericht tragen:

**Gemessen wird in Pixeln, umgerechnet wird erst am Ende.** Die Kennwerte entstehen ohne
jeden Bezug auf den Maßstab; die physikalischen Werte sind abgeleitete Eigenschaften und
bleiben ``None``, solange kein Maßstab bekannt ist. Damit kann nie versehentlich mit einem
geratenen Faktor gerechnet werden, und ein Bild ohne Maßstabsbalken liefert trotzdem
vollständige Formkennwerte.

**Randberührung wird unterschieden.** Eine Pore am *Bildrand* ist angeschnitten - ihre
wahre Größe ist unbekannt. Eine Pore am *Probenrand* grenzt ans Einbettmittel und ist
womöglich gar keine Pore, sondern ein Ausbruch aus der Präparation. Beides sind
verschiedene Vorbehalte, und beide gehören mitgeführt statt stillschweigend verrechnet.
"""

from __future__ import annotations

import numpy as np

from ..core.models import BBox, Pore


def measure(
    labels: np.ndarray,
    gray: np.ndarray,
    contrast: np.ndarray,
    specimen: np.ndarray,
    um_per_px: float | None = None,
) -> list[Pore]:
    """Alle Objekte im Label-Bild vermessen."""
    if not labels.any():
        return []

    from skimage.measure import regionprops

    height, width = labels.shape
    # Der Probenrand als Linie: alles, was an Nicht-Probe grenzt. Ein Pixel Dilatation
    # der Außenwelt genügt - berührt eine Pore diesen Saum, liegt sie an der Grenze.
    outside = ~specimen
    from scipy import ndimage as ndi

    specimen_edge = ndi.binary_dilation(outside, iterations=1) & specimen

    pores: list[Pore] = []
    for prop in regionprops(labels, intensity_image=gray):
        min_row, min_col, max_row, max_col = prop.bbox
        region = labels[min_row:max_row, min_col:max_col] == prop.label

        touches_image = (
            min_row == 0 or min_col == 0 or max_row >= height or max_col >= width
        )
        touches_specimen = bool(
            specimen_edge[min_row:max_row, min_col:max_col][region].any()
        )

        werte = contrast[min_row:max_row, min_col:max_col][region]

        pores.append(
            Pore(
                label=int(prop.label),
                area_px=float(prop.area),
                perimeter_px=float(prop.perimeter),
                # regionprops liefert (Zeile, Spalte) - das Datenmodell führt (x, y).
                centroid_px=(float(prop.centroid[1]), float(prop.centroid[0])),
                bbox=BBox(int(min_col), int(min_row),
                          int(max_col - min_col), int(max_row - min_row)),
                equivalent_diameter_px=float(prop.equivalent_diameter_area),
                major_axis_px=float(prop.axis_major_length),
                minor_axis_px=float(prop.axis_minor_length),
                feret_max_px=float(getattr(prop, "feret_diameter_max", 0.0) or 0.0),
                solidity=float(prop.solidity),
                eccentricity=float(prop.eccentricity),
                orientation_deg=float(np.degrees(prop.orientation)),
                mean_intensity=float(prop.intensity_mean),
                min_intensity=float(prop.intensity_min),
                contrast=float(np.median(werte)) if werte.size else 0.0,
                touches_image_edge=bool(touches_image),
                touches_specimen_edge=touches_specimen,
                um_per_px=um_per_px,
            )
        )
    return pores
