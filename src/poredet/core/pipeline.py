"""Die Pipeline: vier Stufen in einer Reihenfolge, die nicht beliebig ist.

::

    1. Maßstab erkennen          scale/     -> µm/px + Lage des Overlays
    2. Overlay ausschließen      scale/     -> dieser Bereich ist kein Gefüge
    3. Einbettmittel entfernen   specimen/  -> Probenmaske = Bezugsfläche
    4. Poren suchen              detection/ -> nur INNERHALB der Probenmaske
       vermessen und filtern     measurement/, analysis/

**Warum diese Reihenfolge zwingend ist**, Stufe für Stufe:

*Maßstab zuerst.* Der eingebrannte Balken lebt von absoluten Grauwerten - reines Weiß im
Kasten, reines Schwarz im Balken. Jede Beleuchtungs- oder Kontrastkorrektur verschiebt
genau diese Werte und macht die Erkennung unmöglich. Also vor allem anderen.

*Overlay ausschließen, bevor irgendetwas gemessen wird.* Der Maßstabskasten ist ein
perfektes schwarz-weißes Rechteck. Bliebe er drin, wäre der schwarze Balken die größte
"Pore" des Bildes - und das weiße Kastenfeld verschöbe jede Schwelle.

*Einbettmittel vor der Porensuche.* Harz und Poren sind gleich dunkel. Ohne diesen
Schritt findet jeder Detektor das halbe Harz als Riesenpore. Wichtiger noch: die
Probenmaske ist der **Nenner der Porosität**. Bezöge man sie auf die Bildfläche, hinge
die Kennzahl vom Anteil des Harzes ab und damit von der Wahl des Bildausschnitts - sie
wäre keine Werkstoffkennzahl mehr.

*Poren zuletzt, und nur in der Maske.* Auch die Schwelle wird nur über Pixel innerhalb
der Probe bestimmt. Eine über das ganze Bild gerechnete wäre vom dunklen Harz dominiert.

Jede Stufe ist über die Konfiguration austauschbar (``scale``, ``specimen.method``,
``pore.method``), die Reihenfolge ist es nicht - sie steckt hier im Code, weil sie keine
Einstellung ist, sondern eine Eigenschaft der Sache.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np

from ..analysis import FilterContext, apply as apply_filters
from ..config.schema import AppConfig
from ..detection import contrast_image, estimate_background, get_detector
from ..detection import postprocess as detection_post
from ..io.image_reader import ImageReadError, find_images, read_image
from ..measurement.pore_metrics import measure
from ..scale import ScaleResolver, overlay_mask
from ..specimen import segment as segment_specimen
from .context import PipelineContext
from .models import BatchResult, ImageResult

log = logging.getLogger(__name__)


class PoreDetectionPipeline:
    """Wertet einzelne Bilder oder ganze Ordner aus."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.scale_resolver = ScaleResolver(self.config.scale)
        self.detector = get_detector(self.config.pore.method)

    # -- Ordner -------------------------------------------------------------------------

    def run(self, input_dir: str | Path, recursive: bool = False) -> BatchResult:
        """Alle Bilder eines Ordners auswerten."""
        input_dir = Path(input_dir)
        batch = BatchResult(
            input_dir=input_dir,
            started_at=datetime.now(),
            config_summary={
                "scale": "overlay_bar",
                "specimen": self.config.specimen.method,
                "pore": self.config.pore.method,
                "filters": [f.name for f in self.config.analysis.filters if f.enabled],
            },
        )
        for path in find_images(input_dir, recursive=recursive):
            try:
                batch.results.append(self.run_image(path))
            except ImageReadError as exc:
                batch.failures.append((path, str(exc)))
            except Exception as exc:  # pragma: no cover - ein Bild darf den Lauf nicht töten
                log.exception("%s ist gescheitert", path.name)
                batch.failures.append((path, f"{type(exc).__name__}: {exc}"))
        batch.finished_at = datetime.now()
        return batch

    def iter_images(
        self, input_dir: str | Path, recursive: bool = False
    ) -> Iterator[tuple[PipelineContext, ImageResult]]:
        """Wie :meth:`run`, gibt aber zusätzlich den Kontext heraus.

        Für Kontrollbilder und die spätere GUI: dort werden die Zwischenmasken gebraucht,
        die ``ImageResult`` bewusst nicht mitführt.
        """
        for path in find_images(Path(input_dir), recursive=recursive):
            try:
                ctx = self.analyse(path)
            except ImageReadError as exc:
                log.warning("%s wird übersprungen: %s", path.name, exc)
                continue
            yield ctx, self._to_result(ctx, 0.0)

    # -- Einzelbild ---------------------------------------------------------------------

    def run_image(self, path: str | Path) -> ImageResult:
        """Ein Bild auswerten."""
        started = time.perf_counter()
        ctx = self.analyse(path)
        return self._to_result(ctx, time.perf_counter() - started)

    def analyse(self, path: str | Path) -> PipelineContext:
        """Die vier Stufen, in dieser Reihenfolge. Liefert den vollen Kontext."""
        image = read_image(path)
        ctx = PipelineContext(path=image.path, gray=image.gray, color=image.color)

        self._stage_scale(ctx)
        self._stage_specimen(ctx)
        self._stage_detect(ctx)
        self._stage_measure(ctx)
        return ctx

    # -- Stufe 1 und 2: Maßstab und Overlay ---------------------------------------------

    def _stage_scale(self, ctx: PipelineContext) -> None:
        outcome = self.scale_resolver.resolve(ctx.gray, ctx.path)
        ctx.scale = outcome.scale
        ctx.extras["scale_outcome"] = outcome

        if ctx.scale is None:
            grund = outcome.rejections[0] if outcome.rejections else "kein Overlay gefunden"
            ctx.note("massstab", f"nicht erkannt ({grund})")
            if self.config.require_scale:
                raise ValueError(f"{ctx.path.name}: kein Maßstab erkannt - {grund}")
            ctx.warn(
                "kein Maßstab erkannt - es wird in Pixeln gemessen, die physikalischen "
                "Werte bleiben leer"
            )
            ctx.excluded = np.zeros(ctx.shape, dtype=bool)
            return

        ctx.note("massstab", f"{ctx.scale.um_per_px:.6g} µm/px aus \"{ctx.scale.label_text}\"")
        for warnung in ctx.scale.warnings:
            ctx.warn(f"Maßstab: {warnung}")

        # Stufe 2: der Overlay-Bereich fällt ab hier aus allem heraus.
        ctx.excluded = overlay_mask(ctx.shape, ctx.scale, pad=self.config.scale.exclusion_pad_px)
        if ctx.excluded.any():
            ctx.note("overlay", f"{ctx.excluded_area_px} px ausgeschlossen")

    # -- Stufe 3: Einbettmittel ----------------------------------------------------------

    def _stage_specimen(self, ctx: PipelineContext) -> None:
        mask = segment_specimen(ctx.gray, self.config.specimen, color=ctx.color)
        ctx.resin = mask.resin
        ctx.extras["specimen_mask"] = mask

        # Die auswertbare Fläche ist die Probe OHNE das eingebrannte Overlay. Beide Masken
        # entstehen unabhängig voneinander und werden hier zusammengeführt.
        specimen = mask.specimen
        if ctx.excluded is not None:
            specimen = specimen & ~ctx.excluded
        ctx.specimen = specimen

        ctx.note(
            "probe",
            f"{mask.specimen_frac:.1%} Probe"
            + (f", {mask.resin_frac:.1%} Einbettmittel" if mask.has_resin else
               ", kein Einbettmittel"),
        )
        for warnung in mask.warnings:
            ctx.warn(f"Probenmaske: {warnung}")

        if ctx.specimen_area_px == 0:
            ctx.warn("keine auswertbare Probenfläche - es wird nichts gesucht")

    # -- Stufe 4: Poren ------------------------------------------------------------------

    def _stage_detect(self, ctx: PipelineContext) -> None:
        from ..detection.base import DetectionInput

        pore_cfg = self.config.pore
        assert ctx.specimen is not None

        ctx.background = estimate_background(ctx.gray, pore_cfg.local_contrast.background)
        ctx.contrast = contrast_image(ctx.gray, ctx.background)

        data = DetectionInput(
            gray=ctx.gray, specimen=ctx.specimen,
            background=ctx.background, contrast=ctx.contrast,
        )
        result = self.detector.detect(data, pore_cfg)
        ctx.candidates = result.mask
        ctx.extras["detection"] = result
        for hinweis in result.notes:
            ctx.warn(f"Detektion: {hinweis}")

        mask = detection_post.clean(result.mask, pore_cfg)
        labels = detection_post.label_image(mask, pore_cfg)
        labels, zusaetzlich = detection_post.split_touching(labels, pore_cfg)
        ctx.labels = labels

        schwellen = ", ".join(f"{k} {v:.1f}" for k, v in result.thresholds.items())
        ctx.note(
            "poren",
            f"{int(labels.max())} Kandidaten ({result.method}"
            + (f", {schwellen}" if schwellen else "")
            + (f", {zusaetzlich} durch Trennung" if zusaetzlich else "") + ")",
        )

    def _stage_measure(self, ctx: PipelineContext) -> None:
        assert ctx.labels is not None and ctx.specimen is not None and ctx.contrast is not None

        pores = measure(
            ctx.labels, ctx.gray, ctx.contrast, ctx.specimen, um_per_px=ctx.um_per_px
        )

        if self.config.pore.exclude_edge_pores:
            vorher = len(pores)
            pores = [p for p in pores if not p.touches_image_edge]
            if vorher != len(pores):
                ctx.note("randporen", f"{vorher - len(pores)} angeschnittene verworfen")

        behalten, verworfen = apply_filters(
            pores,
            self.config.analysis,
            FilterContext(
                image_area_px=ctx.width * ctx.height,
                specimen_area_px=ctx.specimen_area_px,
                um_per_px=ctx.um_per_px,
            ),
        )
        ctx.pores = behalten
        ctx.rejected = verworfen
        if verworfen:
            gruende: dict[str, int] = {}
            for eintrag in verworfen:
                gruende[eintrag.filter_name] = gruende.get(eintrag.filter_name, 0) + 1
            ctx.note(
                "filter",
                ", ".join(f"{anzahl}x {name}" for name, anzahl in sorted(gruende.items())),
            )

    # -- Ergebnis ------------------------------------------------------------------------

    def _to_result(self, ctx: PipelineContext, duration_s: float) -> ImageResult:
        return ImageResult(
            path=ctx.path,
            width=ctx.width,
            height=ctx.height,
            scale=ctx.scale,
            specimen_area_px=ctx.specimen_area_px,
            excluded_area_px=ctx.excluded_area_px,
            pores=ctx.pores,
            rejected=ctx.rejected,
            warnings=list(ctx.warnings),
            stages=dict(ctx.stages),
            duration_s=duration_s,
        )
