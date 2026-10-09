"""Nahe beieinanderliegende Porenteile zu einer Pore zusammenführen.

Läuft VOR ``postprocess.split_touching``: erst werden zerrissene Poren wieder
zusammengesetzt, dann zusammengewachsene Nester getrennt. Andersherum würde das
Zusammenführen jede Trennung sofort aufheben, weil sich die getrennten Teile berühren.
Er beantwortet die Gegenfrage zur Trennung: was im Bild *eine* Pore ist, aber als zwei
Objekte im Label-Bild steht.

Das passiert, wenn die Schwelle am hellen Reliefsaum oder an einem Helligkeitsverlauf
auf dem Porenboden eine ein Pixel breite Rinne durch die Maske reißt: aus einer Pore
werden zwei Komponenten, die sich nicht berühren.

Zusammengeführt wird, was höchstens ``max_gap_px`` Pixel Abstand hat (0 = nur
berührende). Mit ``bridge_gaps`` wird der Spalt dazwischen der Pore zugeschlagen, damit
sie ein zusammenhängendes Objekt ist - sonst stünden für eine Pore zwei Umrisse im Bild,
und Umfang, Solidität und Feret-Durchmesser rechneten mit einer Lücke.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..einstellungen import Abschnitt


def merge_close(labels: np.ndarray, cfg: Abschnitt,
                erlaubt: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Labels mit höchstens ``max_gap_px`` Abstand zu einem verschmelzen.

    ``erlaubt`` ist die auswertbare Probenfläche: zusammengeführt wird nur über einen
    Spalt, der in der Probe liegt. Liegt zwischen zwei Teilen Einbettmittel oder der
    Maßstabskasten, sind es zwei Hohlräume - und ein überbrückter Spalt zählte sonst
    als Porenfläche außerhalb der Probe.

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
        if erlaubt is not None:
            # Die Verbindung darf nur durch die Probe laufen. Die Poren selbst bleiben
            # in reach, auch falls eine schon außerhalb läge.
            reach &= erlaubt | mask

    _count, groups = cv2.connectedComponents(reach.astype(np.uint8), connectivity=8)
    merged = _renumber(np.where(mask, groups, 0))

    if merge.bridge_gaps and gap > 0:
        merged = _bridge(merged, labels, gap, erlaubt)

    return merged, max(0, int(labels.max()) - int(merged.max()))


def _bridge(merged: np.ndarray, original: np.ndarray, gap: int,
            erlaubt: np.ndarray | None = None) -> np.ndarray:
    """Den Spalt zwischen zusammengeführten Teilen der Pore zuschlagen.

    Nur Poren, die tatsächlich aus mehreren Teilen entstanden sind, werden angefasst -
    eine Einzelpore behält ihren Umriss unverändert. Gefüllt werden nur Pixel, die noch
    zu keiner Pore gehören und in ``erlaubt`` (der Probenfläche) liegen.
    """
    out = merged.copy()
    kernel = np.ones((gap + 1, gap + 1), dtype=np.uint8)
    pad = gap + 1
    hoehe, breite = out.shape

    zusammengesetzt = set(mehrteilig(merged, original).tolist())
    if not zusammengesetzt:
        return out

    for label, fenster in enumerate(ndi.find_objects(merged), start=1):
        if fenster is None or label not in zusammengesetzt:
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
        if erlaubt is not None:
            frei &= erlaubt[ys, xs]
        out[ys, xs][geschlossen.astype(bool) & frei] = label

    return out


def mehrteilig(gross: np.ndarray, teile: np.ndarray) -> np.ndarray:
    """Labels in ``gross``, die mehr als ein Label aus ``teile`` überdecken.

    In einem Zug über das ganze Bild - so muss nur angefasst werden, was wirklich aus
    mehreren Teilen besteht, und nicht jede einzelne Pore.
    """
    beide = (gross > 0) & (teile > 0)
    paare = np.unique(np.stack([gross[beide], teile[beide]]).astype(np.int64), axis=1)
    werte, anzahl = np.unique(paare[0], return_counts=True)
    return werte[anzahl > 1]


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Labels lückenlos ab 1 durchnummerieren, 0 bleibt Hintergrund."""
    werte, invers = np.unique(labels, return_inverse=True)
    if werte[0] != 0:
        return (invers.reshape(labels.shape) + 1).astype(np.int32)
    return invers.reshape(labels.shape).astype(np.int32)
