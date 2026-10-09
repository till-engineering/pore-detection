"""Die Auswertung eines Bildes - fünf Stufen in fester Reihenfolge.

::

    1. Maßstab erkennen          massstab/      -> µm/px + Lage des Overlays
    2. Overlay ausschließen      massstab/      -> dieser Bereich ist kein Gefüge
    3. Einbettmittel entfernen   einbettmittel/ -> Probenmaske = Bezugsfläche
    4. Poren suchen              detektion/     -> nur INNERHALB der Probenmaske
       aufräumen, Spalte schließen, Nester trennen
       vermessen und filtern     messung.py, filter.py
    5. manuelle Korrektur        korrekturen.py -> entfernt/aufgenommen/eingezeichnet

**Warum diese Reihenfolge:** Der Maßstabsbalken lebt von absoluten Grauwerten (reines
Weiß, reines Schwarz) - also vor jeder anderen Verarbeitung. Bliebe das Overlay drin,
wäre der schwarze Balken die größte "Pore". Harz und Poren sind gleich dunkel, deshalb
muss das Einbettmittel vor der Porensuche weg; außerdem ist die Probenmaske der
**Nenner der Porosität**. Die Porensuche läuft zuletzt und nur in der Maske.

Welches Verfahren je Stufe läuft, steht in der ``einstellungen.yaml``
(``specimen.method``, ``pore.method``) - die Reihenfolge steht hier im Code.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import korrekturen as manual
from .bild import read_image
from .detektion import contrast_image, estimate_background, get_detector
from .detektion import merge as detection_merge
from .detektion import postprocess as detection_post
from .detektion.base import DetectionInput
from .einbettmittel import segment as segment_specimen
from .einstellungen import Abschnitt, laden
from .filter import FilterContext
from .filter import apply as apply_filters
from .massstab import ScaleResolver, overlay_mask
from .messung import measure, measure_basis, vervollstaendigen
from .modelle import ImageResult, Pore, RejectedPore, ScaleInfo

log = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Alles, was während der Auswertung eines Bildes entsteht - auch die Zwischenmasken,
    damit der Viewer zeigen kann, *an welcher Stufe* etwas passiert ist."""

    path: Path
    gray: np.ndarray
    color: np.ndarray | None = None

    scale: ScaleInfo | None = None
    excluded: np.ndarray | None = None        # bool, eingebranntes Overlay
    specimen: np.ndarray | None = None        # bool, auswertbare Probenfläche
    resin: np.ndarray | None = None           # bool, Einbettmittel
    background: np.ndarray | None = None      # float32, geschätzter Untergrund
    contrast: np.ndarray | None = None        # float32, background - gray
    candidates: np.ndarray | None = None      # bool, Detektorergebnis vor dem Aufräumen
    labels: np.ndarray | None = None          # int32, nummerierte Poren

    pores: list[Pore] = field(default_factory=list)
    rejected: list[RejectedPore] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.gray.shape[0]), int(self.gray.shape[1]))

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    @property
    def specimen_area_px(self) -> int:
        return 0 if self.specimen is None else int(self.specimen.sum())

    @property
    def excluded_area_px(self) -> int:
        return 0 if self.excluded is None else int(self.excluded.sum())

    @property
    def um_per_px(self) -> float | None:
        return None if self.scale is None else self.scale.um_per_px

    def note(self, stage: str, text: str) -> None:
        """Festhalten, was eine Stufe getan hat - erscheint im Viewer."""
        self.stages[stage] = text

    def warn(self, text: str) -> None:
        self.warnings.append(text)


class Pipeline:
    """Wertet einzelne Bilder aus. Einmal anlegen, dann für viele Bilder benutzen -
    beim Anlegen wird die OCR geladen."""

    def __init__(self, config: Abschnitt | None = None) -> None:
        self.config = config or laden()
        self.scale_resolver = ScaleResolver(self.config.scale)
        self.detector = get_detector(self.config.pore.method)

    def analyse(
        self,
        path: str | Path,
        corrections: Sequence[manual.PoreCorrection] | None = None,
    ) -> PipelineContext:
        """Alle Stufen, in dieser Reihenfolge.

        ``corrections`` ersetzt die gespeicherten Korrekturen dieses Bildes - eine leere
        Liste liefert das rein automatische Ergebnis. ``None`` heißt: aus der Ablage.
        """
        image = read_image(path)
        ctx = PipelineContext(path=image.path, gray=image.gray, color=image.color)

        self._stage_scale(ctx)
        self._stage_specimen(ctx)
        self._stage_detect(ctx)
        self._stage_measure(ctx)
        self.stage_corrections(ctx, corrections)
        return ctx

    # -- Stufe 1 und 2: Maßstab und Overlay ---------------------------------------------

    def _stage_scale(self, ctx: PipelineContext) -> None:
        outcome = self.scale_resolver.resolve(ctx.gray, ctx.path)
        ctx.scale = outcome.scale

        # Stufe 2: der Overlay-Bereich fällt ab hier aus allem heraus - auch wenn die
        # Beschriftung nicht lesbar war. Der Kasten ist gefunden, und sein schwarzer
        # Balken wäre sonst die größte "Pore" des Bildes.
        ctx.excluded = overlay_mask(ctx.shape, outcome.overlay_box,
                                    pad=self.config.scale.exclusion_pad_px)
        if ctx.excluded.any():
            ctx.note("overlay", f"{ctx.excluded_area_px} px ausgeschlossen")

        if ctx.scale is None:
            grund = outcome.rejections[0] if outcome.rejections else "kein Overlay gefunden"
            ctx.note("massstab", f"nicht erkannt ({grund})")
            if self.config.require_scale:
                raise ValueError(f"{ctx.path.name}: kein Maßstab erkannt - {grund}")
            ctx.warn("kein Maßstab erkannt - es wird in Pixeln gemessen, die physikalischen "
                     "Werte bleiben leer")
            if ctx.excluded.any():
                ctx.warn("Maßstabskasten gefunden, aber nicht lesbar - er ist trotzdem von "
                         "der Auswertung ausgeschlossen")
            return

        if ctx.scale.label_text:
            ctx.note("massstab", f"{ctx.scale.um_per_px:.6g} µm/px aus \"{ctx.scale.label_text}\"")
        else:
            ctx.note("massstab", f"{ctx.scale.um_per_px:.6g} µm/px (fest vorgegeben)")
        for warnung in ctx.scale.warnings:
            ctx.warn(f"Maßstab: {warnung}")

    # -- Stufe 3: Einbettmittel ----------------------------------------------------------

    def _stage_specimen(self, ctx: PipelineContext) -> None:
        mask = segment_specimen(ctx.gray, self.config.specimen, color=ctx.color)
        ctx.resin = mask.resin
        # Die auswertbare Fläche ist die Probe OHNE das eingebrannte Overlay.
        ctx.specimen = mask.specimen & ~ctx.excluded

        ctx.note("probe", f"{mask.specimen_frac:.1%} Probe" + (
            f", {mask.resin_frac:.1%} Einbettmittel" if mask.has_resin else ", kein Einbettmittel"))
        for warnung in mask.warnings:
            ctx.warn(f"Probenmaske: {warnung}")
        if ctx.specimen_area_px == 0:
            ctx.warn("keine auswertbare Probenfläche - es wird nichts gesucht")

    # -- Stufe 4: Poren ------------------------------------------------------------------

    def _untergrund_noetig(self) -> bool:
        """Braucht etwas den geschätzten Untergrund? Die Porensuche über den lokalen
        Kontrast und der Filter "contrast" - sonst nur der Kennwert Kontrast."""
        if self.config.pore.method == "local_contrast":
            return True
        return any(f.name == "contrast" and f.enabled for f in self.config.analysis.filters)

    def _stage_detect(self, ctx: PipelineContext) -> None:
        pore_cfg = self.config.pore
        untergrund_cfg = pore_cfg.local_contrast.background
        if not self._untergrund_noetig() and untergrund_cfg.method != "aus":
            # adaptive und threshold suchen auf dem Grauwert. Den Untergrund (closing ist
            # die teuerste Rechnung im Programm) bräuchte dann nur der Kennwert Kontrast;
            # der wird gegen die mittlere Probenhelligkeit gemessen.
            untergrund_cfg = untergrund_cfg.kopie(method="aus")
            ctx.note("untergrund", "nicht geschätzt (nicht gebraucht) - Kontrast gegen "
                     "die mittlere Probenhelligkeit")
        ctx.background = estimate_background(ctx.gray, untergrund_cfg, ctx.specimen)
        ctx.contrast = contrast_image(ctx.gray, ctx.background)

        result = self.detector.detect(
            DetectionInput(gray=ctx.gray, specimen=ctx.specimen,
                           background=ctx.background, contrast=ctx.contrast),
            pore_cfg,
        )
        ctx.candidates = result.mask
        for hinweis in result.notes:
            ctx.warn(f"Detektion: {hinweis}")

        # Aufräumen und Zusammenführen lassen Poren wachsen - nur innerhalb der Probe.
        # Pixel außerhalb zählten als Porenfläche, aber nicht zur Probenfläche.
        mask = detection_post.clean(result.mask, pore_cfg, erlaubt=ctx.specimen)
        labels = detection_post.label_image(mask, pore_cfg)
        # Erst schmale Spalte schließen (eine zerrissene Pore wird wieder eine), dann
        # zusammengewachsene Nester trennen. Umgekehrt würde das Zusammenführen jede
        # Trennung sofort wieder aufheben - die getrennten Teile berühren sich.
        labels, vereinigt = detection_merge.merge_close(labels, pore_cfg, erlaubt=ctx.specimen)
        labels, zusaetzlich = detection_post.split_touching(labels, pore_cfg)

        # Sicherung: kein Schritt darf eine Pore aus der Probe hinauswachsen lassen. Tut es
        # doch einer, wird beschnitten und gewarnt - ein solcher Fehler soll auffallen,
        # statt still in die Porosität einzugehen.
        draussen = (labels > 0) & ~ctx.specimen
        if draussen.any():
            labels = np.where(draussen, 0, labels).astype(labels.dtype)
            ctx.warn(f"Detektion: {int(draussen.sum())} Porenpixel lagen außerhalb der "
                     "Probe und wurden entfernt - bitte melden, das ist ein Programmfehler")
        ctx.labels = labels

        schwellen = ", ".join(f"{k} {v:.1f}" for k, v in result.thresholds.items())
        ctx.note("poren", f"{int(labels.max())} Kandidaten ({result.method}"
                 + (f", {schwellen}" if schwellen else "")
                 + (f", {vereinigt} zusammengeführt" if vereinigt else "")
                 + (f", {zusaetzlich} durch Trennung" if zusaetzlich else "") + ")")

    def _stage_measure(self, ctx: PipelineContext) -> None:
        # Erst grob für alle Kandidaten, die Formwerte nur für die, die sie brauchen -
        # siehe messung.py.
        pores = measure_basis(ctx.labels, ctx.gray, ctx.contrast, ctx.specimen,
                              um_per_px=ctx.um_per_px, overlay=ctx.excluded)

        def voll(poren: list[Pore]) -> list[Pore]:
            return vervollstaendigen(poren, ctx.labels, ctx.gray, ctx.contrast,
                                     ctx.specimen, um_per_px=ctx.um_per_px,
                                     overlay=ctx.excluded)

        # Angeschnittene Poren verwerfen: Filter "edge" in der einstellungen.yaml.
        ctx.pores, ctx.rejected = apply_filters(
            pores, self.config.analysis,
            FilterContext(image_area_px=ctx.width * ctx.height,
                          specimen_area_px=ctx.specimen_area_px, um_per_px=ctx.um_per_px),
            vervollstaendigen=voll,
        )
        if ctx.rejected:
            gruende: dict[str, int] = {}
            for eintrag in ctx.rejected:
                gruende[eintrag.filter_name] = gruende.get(eintrag.filter_name, 0) + 1
            ctx.note("filter", ", ".join(f"{n}x {name}" for name, n in sorted(gruende.items())))

    # -- Stufe 5: manuelle Korrektur ----------------------------------------------------

    def stage_corrections(
        self, ctx: PipelineContext, corrections: Sequence[manual.PoreCorrection] | None
    ) -> None:
        cfg = self.config.corrections
        if corrections is None:
            # Ohne Ablageordner (der Stapellauf gibt die Korrekturen selbst mit, die
            # einstellungen.yaml kennt keinen) gibt es auch nichts Gespeichertes.
            ordner = cfg.get("directory")
            if not cfg.enabled or not ordner:
                return
            corrections = manual.CorrectionStore(ordner).load(ctx.path)
        if not corrections:
            return

        def vermessen(labels: np.ndarray) -> list:
            # Eingezeichnete Poren nehmen denselben Messweg wie erkannte.
            return measure(labels, ctx.gray, ctx.contrast, ctx.specimen,
                           um_per_px=ctx.um_per_px, overlay=ctx.excluded)

        outcome = manual.apply(ctx.pores, ctx.rejected, ctx.labels, corrections,
                               measure=vermessen, allowed=ctx.specimen)
        ctx.labels = outcome.labels
        ctx.pores = outcome.pores
        ctx.rejected = outcome.rejected
        ctx.extras["korrekturen"] = outcome.summary()
        ctx.extras["korrekturen_gezeichnet"] = dict(outcome.drawn)

        ctx.note("korrektur", f"{len(outcome.removed)} von Hand entfernt, "
                 f"{len(outcome.restored)} von Hand aufgenommen, "
                 f"{len(outcome.drawn)} eingezeichnet")
        if outcome.missed:
            ctx.warn(f"{len(outcome.missed)} Korrektur(en) treffen keine Pore mehr - "
                     "vermutlich nach geänderten Parametern")

    # -- Ergebnis ------------------------------------------------------------------------

    @staticmethod
    def to_result(ctx: PipelineContext) -> ImageResult:
        return ImageResult(
            path=ctx.path, width=ctx.width, height=ctx.height, scale=ctx.scale,
            specimen_area_px=ctx.specimen_area_px, excluded_area_px=ctx.excluded_area_px,
            pores=ctx.pores, rejected=ctx.rejected,
            warnings=list(ctx.warnings), stages=dict(ctx.stages),
        )
