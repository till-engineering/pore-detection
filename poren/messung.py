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

**Gemessen wird in zwei Stufen.** Die meisten Kandidaten sind Rauschen und fallen schon am
ersten Filter (``min_diameter``), der nur die Fläche braucht. :func:`measure_basis` liefert
deshalb für alle Kandidaten auf einmal nur die billigen Werte (Fläche, Lage, Grauwerte,
Rand); die Formwerte - Umfang, Achsen, konvexe Hülle, Feret-Durchmesser - bleiben ``None``.
:func:`vervollstaendigen` rechnet sie nach, und zwar nur für Poren, die einen Formfilter
erreichen oder am Ende gezählt werden. Gezählte Poren haben damit immer alle Werte, und
zwar exakt dieselben wie bei :func:`measure`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from . import formeln
from .modelle import BBox, Pore


def measure(
    labels: np.ndarray,
    gray: np.ndarray,
    contrast: np.ndarray,
    specimen: np.ndarray,
    um_per_px: float | None = None,
    nur: Iterable[int] | None = None,
    overlay: np.ndarray | None = None,
) -> list[Pore]:
    """Objekte im Label-Bild mit allen Kennwerten vermessen - alle oder nur die Labels
    in ``nur``. ``overlay`` ist der ausgeschlossene Maßstabskasten (siehe :func:`_raender`)."""
    if nur is not None:
        nur = list(nur)
        if not nur:
            return []
        labels = np.where(np.isin(labels, nur), labels, 0)
    if not labels.any():
        return []

    from skimage.measure import regionprops

    height, width = labels.shape
    specimen_edge, edge_distance, overlay_saum = _raender(specimen, overlay)

    pores: list[Pore] = []
    for prop in regionprops(labels, intensity_image=gray):
        min_row, min_col, max_row, max_col = prop.bbox
        region = labels[min_row:max_row, min_col:max_col] == prop.label

        touches_image = (
            min_row == 0 or min_col == 0 or max_row >= height or max_col >= width
            or (overlay_saum is not None
                and bool(overlay_saum[min_row:max_row, min_col:max_col][region].any()))
        )
        touches_specimen = bool(
            specimen_edge[min_row:max_row, min_col:max_col][region].any()
        )

        werte = contrast[min_row:max_row, min_col:max_col][region]

        abstand = None
        if edge_distance is not None:
            innen = edge_distance[min_row:max_row, min_col:max_col][region]
            innen = innen[innen > 0]
            abstand = float(max(innen.min() - 1.0, 0.0)) if innen.size else 0.0

        pores.append(
            Pore(
                label=int(prop.label),
                area_px=float(prop.area),
                perimeter_px=float(prop.perimeter),
                # regionprops liefert (Zeile, Spalte) - das Datenmodell führt (x, y).
                centroid_px=(float(prop.centroid[1]), float(prop.centroid[0])),
                bbox=BBox(int(min_col), int(min_row),
                          int(max_col - min_col), int(max_row - min_row)),
                equivalent_diameter_px=formeln.aequivalentdurchmesser(float(prop.area)),
                major_axis_px=float(prop.axis_major_length),
                minor_axis_px=float(prop.axis_minor_length),
                feret_max_px=float(getattr(prop, "feret_diameter_max", 0.0) or 0.0),
                solidity=formeln.soliditaet(float(prop.area), float(prop.area_convex)),
                eccentricity=float(prop.eccentricity),
                orientation_deg=float(np.degrees(prop.orientation)),
                mean_intensity=float(prop.intensity_mean),
                min_intensity=float(prop.intensity_min),
                contrast=float(np.median(werte)) if werte.size else 0.0,
                touches_image_edge=bool(touches_image),
                touches_specimen_edge=touches_specimen,
                specimen_edge_distance_px=abstand,
                um_per_px=um_per_px,
            )
        )
    return pores


def measure_basis(
    labels: np.ndarray,
    gray: np.ndarray,
    contrast: np.ndarray,
    specimen: np.ndarray,
    um_per_px: float | None = None,
    overlay: np.ndarray | None = None,
) -> list[Pore]:
    """Alle Objekte mit den billigen Kennwerten - für alle Labels auf einmal gerechnet.

    Fläche, Box, Schwerpunkt, Grauwerte, Kontrast und Randlage stimmen mit
    :func:`measure` überein; die Formwerte bleiben ``None`` (siehe Modulbeschreibung).
    """
    if not labels.any():
        return []

    from scipy import ndimage as ndi

    height, width = labels.shape
    n = int(labels.max())
    fenster = ndi.find_objects(labels, max_label=n)
    vorhanden = np.array([f is not None for f in fenster], dtype=bool)
    index = np.arange(1, n + 1)[vorhanden]

    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    flaeche = np.bincount(lab, minlength=n + 1).astype(np.float64)
    zeile = np.bincount(lab, weights=ys, minlength=n + 1)
    spalte = np.bincount(lab, weights=xs, minlength=n + 1)
    grau = np.bincount(lab, weights=gray[ys, xs].astype(np.float64), minlength=n + 1)

    grau_min = ndi.minimum(gray, labels, index)
    kontrast = ndi.median(contrast, labels, index)

    specimen_edge, edge_distance, overlay_saum = _raender(specimen, overlay)
    am_rand = ndi.maximum(specimen_edge.astype(np.uint8), labels, index)
    am_overlay = (ndi.maximum(overlay_saum.astype(np.uint8), labels, index)
                  if overlay_saum is not None else np.zeros(len(index), dtype=bool))
    abstand = None
    if edge_distance is not None:
        # Wie in measure(): kleinster Abstand > 0, minus 1; ohne solchen Pixel 0.
        abstand = ndi.minimum(np.where(edge_distance > 0, edge_distance, np.inf), labels, index)

    pores: list[Pore] = []
    for i, label in enumerate(index):
        zeilen, spalten = fenster[label - 1]
        a = flaeche[label]
        d = None
        if abstand is not None:
            d = float(max(abstand[i] - 1.0, 0.0)) if np.isfinite(abstand[i]) else 0.0
        pores.append(
            Pore(
                label=int(label),
                area_px=float(a),
                perimeter_px=None,
                centroid_px=(float(spalte[label] / a), float(zeile[label] / a)),
                bbox=BBox(int(spalten.start), int(zeilen.start),
                          int(spalten.stop - spalten.start), int(zeilen.stop - zeilen.start)),
                equivalent_diameter_px=formeln.aequivalentdurchmesser(float(a)),
                major_axis_px=None,
                minor_axis_px=None,
                feret_max_px=None,
                solidity=None,
                eccentricity=None,
                orientation_deg=None,
                mean_intensity=float(grau[label] / a),
                min_intensity=float(grau_min[i]),
                contrast=float(kontrast[i]),
                touches_image_edge=bool(zeilen.start == 0 or spalten.start == 0
                                        or zeilen.stop >= height or spalten.stop >= width
                                        or am_overlay[i]),
                touches_specimen_edge=bool(am_rand[i]),
                specimen_edge_distance_px=d,
                um_per_px=um_per_px,
            )
        )
    return pores


def vervollstaendigen(
    pores: Sequence[Pore],
    labels: np.ndarray,
    gray: np.ndarray,
    contrast: np.ndarray,
    specimen: np.ndarray,
    um_per_px: float | None = None,
    overlay: np.ndarray | None = None,
) -> list[Pore]:
    """Poren ohne Formwerte voll vermessen. Reihenfolge und vollständige Poren bleiben."""
    fehlend = [p.label for p in pores if not p.vollstaendig]
    if not fehlend:
        return list(pores)
    voll = {p.label: p for p in measure(labels, gray, contrast, specimen,
                                        um_per_px=um_per_px, nur=fehlend, overlay=overlay)}
    return [voll.get(p.label, p) for p in pores]


def _raender(
    specimen: np.ndarray, overlay: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Probenrand, Abstand zum Probenrand und Saum um das Overlay.

    Der Probenrand ist alles, was an Nicht-Probe grenzt - ein Pixel Dilatation der
    Außenwelt genügt. Der ausgeschlossene **Maßstabskasten zählt dabei nicht** als
    Außenwelt: er ist kein Einbettmittel, sondern liegt über der Probe. Eine Pore, die an
    ihn stößt, ist verdeckt und damit angeschnitten wie am Bildrand - dafür der Saum.

    Der Abstand gilt je Probenpixel bis zur Nicht-Probe (ein Randpixel hat 1, daher
    später -1); ohne Nicht-Probe gibt es keinen Rand und damit keinen Abstand.
    """
    from scipy import ndimage as ndi

    drinnen = specimen if overlay is None else specimen | overlay
    outside = ~drinnen
    specimen_edge = ndi.binary_dilation(outside, iterations=1) & specimen
    edge_distance = ndi.distance_transform_edt(drinnen) if outside.any() else None
    overlay_saum = None
    if overlay is not None and overlay.any():
        overlay_saum = ndi.binary_dilation(overlay, iterations=1) & specimen
    return specimen_edge, edge_distance, overlay_saum
