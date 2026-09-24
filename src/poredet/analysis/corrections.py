"""Manuelle Korrektur: Poren von Hand entfernen, wieder aufnehmen oder einzeichnen.

Kernfeature, keine Notlösung - aus demselben Grund wie der manuelle Maßstab: eine
Erkennung ohne Korrekturweg wird in der Praxis nicht akzeptiert. Irgendein Bild hat
immer einen Schleifkratzer, den kein Filter erwischt, eine flache Pore, die ein Filter
zu Unrecht verwirft, oder eine kontrastarme, die der Detektor gar nicht findet.

Drei Entscheidungen tragen das Modul:

**Korrekturen sind Daten, keine Bildbearbeitung.** Gespeichert wird "an dieser Stelle
die Pore entfernen" oder "dieser Umriss ist eine Pore", nicht ein verändertes
Label-Bild. Die Korrektur läuft damit bei jedem Lauf als eigene Stufe hinter den Filtern
und bleibt im Ergebnis sichtbar.

**Eine Pore wird über einen Bildpunkt angesprochen, nicht über ihre Nummer.** Die
Label-Nummern ändern sich, sobald ein Parameter gedreht wird - Schwelle, Trennen,
Zusammenführen. Ein Punkt trifft nach dem Neurechnen dieselbe Pore, solange es sie an
dieser Stelle noch gibt. Gibt es sie nicht mehr, wird das gemeldet und nicht geraten.

**Die Filterlogik bleibt unangetastet.** Eine entfernte Pore wandert zu den verworfenen,
mit dem Grund :data:`MANUAL_FILTER` - genau wie jede andere verworfene Pore mit ihrem
Grund. Eine wieder aufgenommene verliert ihren Verwerfungsgrund; dass sie von Hand
zurückkam, steht in :class:`CorrectionOutcome`. Eine eingezeichnete Pore durchläuft
keine Filter: wer sie zeichnet, hat bereits entschieden, dass sie eine ist.

Vermessen wird eine eingezeichnete Pore nicht hier, sondern über die Funktion, die der
Aufrufer mitgibt - dieselbe Vermessung wie für jede erkannte Pore. Das Modul bleibt so
frei von der Messtechnik.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ..core.models import Pore, RejectedPore

#: Filtername, unter dem eine von Hand entfernte Pore bei den verworfenen steht.
MANUAL_FILTER = "manuell"

#: Wie weit ein Punkt neben der Pore liegen darf und sie trotzdem trifft. Ein Klick auf
#: eine Pore von drei Pixeln landet leicht knapp daneben, und nach dem Neurechnen mit
#: anderen Parametern kann der Rand einer Pore um ein, zwei Pixel wandern.
SEARCH_RADIUS_PX = 3

#: Vermisst ein Label-Bild mit genau einem Objekt - geliefert vom Aufrufer.
MeasureFn = Callable[[np.ndarray], list[Pore]]


class Action(str, Enum):
    REMOVE = "entfernen"
    RESTORE = "aufnehmen"
    DRAW = "zeichnen"


@dataclass(frozen=True)
class PoreCorrection:
    """Eine Korrektur in Bildkoordinaten (nicht Anzeige).

    ``x, y`` ist der Ankerpunkt, über den die Pore getroffen wird. Beim Einzeichnen
    kommt der Umriss als Polygon dazu.
    """

    action: Action
    x: float
    y: float
    points: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict:
        daten = {"aktion": self.action.value, "x": round(self.x, 2), "y": round(self.y, 2)}
        if self.points:
            daten["punkte"] = [[round(px, 2), round(py, 2)] for px, py in self.points]
        return daten

    @classmethod
    def from_dict(cls, data: dict) -> PoreCorrection:
        punkte = tuple((float(px), float(py)) for px, py in data.get("punkte", []))
        return cls(Action(data["aktion"]), float(data["x"]), float(data["y"]), punkte)

    @classmethod
    def drawn(cls, points: Sequence[Sequence[float]]) -> PoreCorrection:
        """Eine eingezeichnete Pore aus ihrem Umriss."""
        punkte = tuple((float(px), float(py)) for px, py in points)
        if len(punkte) < 3:
            raise ValueError("Ein Umriss braucht mindestens drei Punkte")
        xs, ys = zip(*punkte, strict=True)
        return cls(Action.DRAW, sum(xs) / len(xs), sum(ys) / len(ys), punkte)


@dataclass
class CorrectionOutcome:
    """Ergebnis der Korrektur - samt dem, was sie getan hat und was nicht griff."""

    pores: list[Pore]
    rejected: list[RejectedPore]
    labels: np.ndarray
    removed: list[int] = field(default_factory=list)      # Labels, von Hand entfernt
    restored: list[int] = field(default_factory=list)     # Labels, von Hand aufgenommen
    #: Label der eingezeichneten Pore -> Index ihrer Korrektur in der Eingabeliste.
    drawn: dict[int, int] = field(default_factory=dict)
    missed: list[PoreCorrection] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.removed) + len(self.restored) + len(self.drawn)

    def summary(self) -> dict:
        return {
            "entfernt": list(self.removed),
            "aufgenommen": list(self.restored),
            "gezeichnet": sorted(self.drawn),
            "ohne_treffer": [c.to_dict() for c in self.missed],
        }


# --------------------------------------------------------------------------------------
# Treffen
# --------------------------------------------------------------------------------------


def label_at(labels: np.ndarray, x: float, y: float,
             radius: int = SEARCH_RADIUS_PX) -> int:
    """Das Objekt unter dem Punkt, sonst das nächste innerhalb von ``radius``; 0 = keins."""
    hoehe, breite = labels.shape
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < hoehe and 0 <= xi < breite and labels[yi, xi] > 0:
        return int(labels[yi, xi])

    y0, y1 = max(0, yi - radius), min(hoehe, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(breite, xi + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return 0
    fenster = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(fenster)
    if len(ys) == 0:
        return 0
    abstand = (ys + y0 - y) ** 2 + (xs + x0 - x) ** 2
    naechster = int(np.argmin(abstand))
    if abstand[naechster] > radius ** 2:
        return 0
    return int(fenster[ys[naechster], xs[naechster]])


def interior_point(labels: np.ndarray, label: int) -> tuple[float, float] | None:
    """Ein Punkt sicher im Inneren des Objekts - der vom Rand am weitesten entfernte.

    Der Schwerpunkt taugt als Ankerpunkt nicht: bei einer sichelförmigen oder
    verzweigten Pore liegt er außerhalb, und die Korrektur träfe den Nachbarn. Der
    innerste Punkt trifft auch nach einem Neurechnen, bei dem der Rand um ein, zwei
    Pixel wandert, noch dieselbe Pore.
    """
    from scipy import ndimage as ndi

    fenster = ndi.find_objects((labels == label).astype(np.int8))
    if not fenster or fenster[0] is None:
        return None
    zeilen, spalten = fenster[0]
    maske = np.pad(labels[zeilen, spalten] == label, 1)
    abstand = ndi.distance_transform_edt(maske)
    y, x = np.unravel_index(int(np.argmax(abstand)), abstand.shape)
    return float(x - 1 + spalten.start), float(y - 1 + zeilen.start)


def rasterize(points: Sequence[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Den Umriss als gefüllte Maske. Pixel zählen, deren Mitte im Polygon liegt."""
    maske = np.zeros(shape, dtype=np.uint8)
    polygon = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(maske, [polygon], 1)
    return maske.astype(bool)


# --------------------------------------------------------------------------------------
# Anwenden
# --------------------------------------------------------------------------------------


def apply(
    pores: Sequence[Pore],
    rejected: Sequence[RejectedPore],
    labels: np.ndarray,
    corrections: Sequence[PoreCorrection],
    measure: MeasureFn | None = None,
    allowed: np.ndarray | None = None,
) -> CorrectionOutcome:
    """Korrekturen anwenden. Die Eingaben bleiben unverändert.

    Eingezeichnete Poren kommen zuerst, dann Entfernen und Aufnehmen in der
    gespeicherten Reihenfolge. ``allowed`` begrenzt eine Zeichnung auf die auswertbare
    Fläche - eine Pore im Einbettmittel gäbe es für die Porosität nicht. Ohne
    ``measure`` lässt sich nichts einzeichnen; solche Korrekturen gelten als verfehlt.
    """
    gezaehlt: dict[int, Pore] = {p.label: p for p in pores}
    verworfen: dict[int, RejectedPore] = {r.pore.label: r for r in rejected}
    missed: list[PoreCorrection] = []
    drawn: dict[int, int] = {}
    labels = labels.copy()

    for index, korrektur in enumerate(corrections):
        if korrektur.action is not Action.DRAW:
            continue
        if measure is None:
            missed.append(korrektur)
            continue
        region = rasterize(korrektur.points, labels.shape)
        if allowed is not None:
            region &= allowed
        if not region.any():
            missed.append(korrektur)
            continue

        # Eine Zeichnung über einer erkannten Pore ersetzt sie: meist ist die Pore
        # gefunden, aber zu klein geraten, und der Umriss sagt, wie groß sie wirklich
        # ist. Die überdeckten Poren gehen ganz in der neuen auf.
        neu = int(labels.max()) + 1
        for alt in np.unique(labels[region]):
            if alt == 0:
                continue
            labels[labels == alt] = neu
            gezaehlt.pop(int(alt), None)
            verworfen.pop(int(alt), None)
            drawn.pop(int(alt), None)
        labels[region] = neu

        vermessen = measure(np.where(labels == neu, neu, 0).astype(labels.dtype))
        if not vermessen:
            missed.append(korrektur)
            labels[labels == neu] = 0
            continue
        gezaehlt[neu] = vermessen[0]
        drawn[neu] = index

    for korrektur in corrections:
        if korrektur.action is Action.DRAW:
            continue
        label = label_at(labels, korrektur.x, korrektur.y)
        if korrektur.action is Action.REMOVE and label in gezaehlt:
            pore = gezaehlt.pop(label)
            verworfen[label] = RejectedPore(pore, MANUAL_FILTER, "von Hand entfernt")
        elif korrektur.action is Action.RESTORE and label in verworfen:
            gezaehlt[label] = verworfen.pop(label).pore
        elif label == 0 or (label not in gezaehlt and label not in verworfen):
            missed.append(korrektur)
        # Sonst ist die Pore schon im gewünschten Zustand - nichts zu tun.

    vorher_gezaehlt = {p.label for p in pores}
    return CorrectionOutcome(
        pores=sorted(gezaehlt.values(), key=lambda p: p.label),
        rejected=sorted(verworfen.values(), key=lambda r: r.pore.label),
        labels=labels,
        removed=sorted(set(verworfen) & vorher_gezaehlt),
        restored=sorted(set(gezaehlt) - vorher_gezaehlt - set(drawn)),
        drawn=drawn,
        missed=missed,
    )


# --------------------------------------------------------------------------------------
# Bearbeiten - die Liste der Korrekturen fortschreiben
# --------------------------------------------------------------------------------------


def set_counted(
    pores: Sequence[Pore],
    rejected: Sequence[RejectedPore],
    labels: np.ndarray,
    corrections: Sequence[PoreCorrection],
    x: float,
    y: float,
    counted: bool,
) -> list[PoreCorrection]:
    """Die Pore am Punkt als gezählt bzw. verworfen festlegen. Gibt die neue Liste zurück.

    ``pores``, ``rejected`` und ``labels`` sind das **automatische** Ergebnis. Frühere
    Korrekturen an derselben Pore werden ersetzt, nicht gestapelt; entspricht der
    gewünschte Zustand dem automatischen, bleibt gar keine stehen - die Datei enthält
    so nur echte Abweichungen. Eingezeichnete Poren bleiben unberührt.
    """
    label = label_at(labels, x, y)
    bekannt = {p.label for p in pores} | {r.pore.label for r in rejected}
    if label not in bekannt:
        return list(corrections)

    uebrige = [c for c in corrections
               if c.action is Action.DRAW or label_at(labels, c.x, c.y) != label]
    war_gezaehlt = label in {p.label for p in pores}
    if war_gezaehlt == counted:
        return uebrige
    aktion = Action.RESTORE if counted else Action.REMOVE
    return [*uebrige, PoreCorrection(aktion, x, y)]


# --------------------------------------------------------------------------------------
# Ablage
# --------------------------------------------------------------------------------------


class CorrectionStore:
    """Korrekturen als JSON je Bild in einem Ordner.

    Die Datei ist an den **Bildinhalt** gebunden, nicht nur an den Namen: in den
    Testordnern liegen gleichnamige Bilder mit verschiedenem Inhalt, und ein Bild, das
    ersetzt wurde, soll nicht die Korrekturen seines Vorgängers erben.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, image_path: str | Path) -> Path:
        image_path = Path(image_path)
        return self.directory / f"{image_path.stem}_{_fingerprint(image_path)}.json"

    def load(self, image_path: str | Path) -> list[PoreCorrection]:
        pfad = self.path_for(image_path)
        if not pfad.is_file():
            return []
        daten = json.loads(pfad.read_text(encoding="utf-8"))
        return [PoreCorrection.from_dict(d) for d in daten.get("korrekturen", [])]

    def save(self, image_path: str | Path, corrections: Sequence[PoreCorrection]) -> Path:
        """Speichern; ohne Korrekturen wird die Datei entfernt statt leer abgelegt."""
        pfad = self.path_for(image_path)
        if not corrections:
            pfad.unlink(missing_ok=True)
            return pfad
        self.directory.mkdir(parents=True, exist_ok=True)
        daten = {
            "bild": Path(image_path).name,
            "korrekturen": [c.to_dict() for c in corrections],
        }
        pfad.write_text(json.dumps(daten, ensure_ascii=False, indent=2), encoding="utf-8")
        return pfad


def _fingerprint(image_path: Path) -> str:
    return hashlib.sha1(image_path.read_bytes()).hexdigest()[:12]
