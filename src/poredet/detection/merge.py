"""Nahe beieinanderliegende Porenteile zu einer Pore zusammenführen.

Der letzte Schritt der Nachbearbeitung, nach
:func:`~poredet.detection.postprocess.split_touching`.
Er beantwortet die Gegenfrage zur Trennung: was im Bild *eine* Pore ist, aber als zwei
Objekte im Label-Bild steht.

Das passiert auf zwei Wegen:

* **Berührende Labels.** Die Watershed-Trennung legt ihre Schnittlinie mitten durch
  eine Pore, wenn diese eingeschnürt ist - die beiden Hälften liegen danach Kante an
  Kante.
* **Ein schmaler Spalt.** Die Schwelle reißt am hellen Reliefsaum oder an einem
  Helligkeitsverlauf auf dem Porenboden eine ein Pixel breite Rinne durch die Maske.
  Aus einer Pore werden zwei Komponenten, die sich nicht berühren.

Zusammengeführt wird, was höchstens ``max_gap_px`` Pixel Abstand hat (0 = nur
berührende). Mit ``bridge_gaps`` wird der Spalt dazwischen der Pore zugeschlagen, damit
sie ein zusammenhängendes Objekt ist - sonst stünden für eine Pore zwei Umrisse im Bild,
und Umfang, Solidität und Feret-Durchmesser rechneten mit einer Lücke.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..config.schema import PoreConfig


def merge_close(labels: np.ndarray, cfg: PoreConfig) -> tuple[np.ndarray, int]:
    """Labels mit höchstens ``max_gap_px`` Abstand zu einem verschmelzen.

    Zurück kommt das neue Label-Bild und die Zahl der eingesparten Objekte.
    """
    merge = cfg.merge
    if not merge.enabled or not labels.any():
        return labels, 0

    mask = labels > 0
    gap = merge.max_gap_px

    # Ein Quadrat der Kantenlänge gap+1 als Strukturelement: zwei Pixel liegen danach
    # genau dann in einer 8er-Komponente, wenn ihr Schachbrettabstand höchstens gap+1
    # ist - also höchstens ``gap`` Pixel zwischen ihnen frei sind. Ein symmetrischer
    # Kern mit Radius ceil(gap/2) würde bei ungeradem gap einen Pixel mehr überbrücken.
    reach = mask
    if gap > 0:
        kernel = np.ones((gap + 1, gap + 1), dtype=np.uint8)
        reach = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    _count, groups = cv2.connectedComponents(reach.astype(np.uint8), connectivity=8)
    merged = _renumber(np.where(mask, groups, 0))

    if merge.bridge_gaps and gap > 0:
        merged = _bridge(merged, labels, gap)

    return merged, max(0, int(labels.max()) - int(merged.max()))


def _bridge(merged: np.ndarray, original: np.ndarray, gap: int) -> np.ndarray:
    """Den Spalt zwischen zusammengeführten Teilen der Pore zuschlagen.

    Nur Poren, die tatsächlich aus mehreren Teilen entstanden sind, werden angefasst -
    eine Einzelpore behält ihren Umriss unverändert. Gefüllt werden nur Pixel, die noch
    zu keiner Pore gehören.
    """
    out = merged.copy()
    kernel = np.ones((gap + 1, gap + 1), dtype=np.uint8)
    pad = gap + 1
    hoehe, breite = out.shape

    for label, fenster in enumerate(ndi.find_objects(merged), start=1):
        if fenster is None:
            continue
        zeilen, spalten = fenster
        ys = slice(max(0, zeilen.start - pad), min(hoehe, zeilen.stop + pad))
        xs = slice(max(0, spalten.start - pad), min(breite, spalten.stop + pad))

        pore = merged[ys, xs] == label
        teile = np.unique(original[ys, xs][pore])
        if len(teile) <= 1:
            continue

        geschlossen = cv2.morphologyEx(pore.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        frei = out[ys, xs] == 0
        out[ys, xs][geschlossen.astype(bool) & frei] = label

    return out


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Labels lückenlos ab 1 durchnummerieren, 0 bleibt Hintergrund."""
    werte, invers = np.unique(labels, return_inverse=True)
    if werte[0] != 0:
        return (invers.reshape(labels.shape) + 1).astype(np.int32)
    return invers.reshape(labels.shape).astype(np.int32)
