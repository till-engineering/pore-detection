"""Kandidaten -> beste ScaleInfo mit Konfidenz und Herkunft.

Ein stillschweigend falscher Maßstab macht jede Messung falsch, ohne dass es auffällt.
Herkunft und Konfidenz wandern deshalb bis in den Bericht, und der Resolver hält sich an
drei Regeln:

1. **Ein Ergebnis entsteht nur, wenn beides stimmt** - eine plausible Geometrie *und*
   eine lesbare Beschriftung. Ein Balken ohne Text ergibt keinen Maßstab.
2. **Zweifel senken die Konfidenz, sie werden nicht verschwiegen.** Jede Reparatur an
   der OCR, jeder ungewöhnliche Wert und jeder Widerspruch zwischen zwei Engines landet
   als Warnung im Ergebnis.
3. **Im Zweifel kein Maßstab.** ``None`` ist ein brauchbares Ergebnis - die späteren
   Module rechnen dann in Pixeln weiter und lassen die physikalischen Werte leer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.loader import load_config
from ..config.schema import ScaleConfig
from ..core.models import ScaleInfo, ScaleSource
from ..core.units import ParsedLength, parse_length
from .base import OverlayCandidate, ScaleDetector
from .detectors.box_overlay import BoxOverlayDetector
from .detectors.split_box import SplitBoxOverlayDetector
from .ocr import available_engines, patch_variants

log = logging.getLogger(__name__)


@dataclass
class ScaleOutcome:
    """Was bei der Maßstabserkennung herauskam - inklusive der Fehlschläge.

    Die verworfenen Kandidaten werden mitgeführt, weil sie im Kontrollbild und in der
    GUI gebraucht werden: "nichts gefunden" ist eine viel schlechtere Auskunft als
    "gefunden, aber die Beschriftung war nicht lesbar".
    """

    scale: ScaleInfo | None
    candidates: list[OverlayCandidate] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.scale is not None


class ScaleResolver:
    """Führt Detektoren und OCR zu einem Maßstab zusammen."""

    def __init__(
        self,
        cfg: ScaleConfig | None = None,
        detectors: list[ScaleDetector] | None = None,
    ) -> None:
        self.cfg = cfg or load_config().scale
        # Reihenfolge ist hier belanglos - die Kandidaten aller Detektoren werden
        # gemeinsam nach geometry_score sortiert. Die beiden schließen sich ohnehin
        # gegenseitig aus: split_box lässt jeden Balken liegen, der von Weiß
        # eingeschlossen ist, und genau die sind der Fall von box_overlay.
        self.detectors: list[ScaleDetector] = detectors or [
            BoxOverlayDetector(),
            SplitBoxOverlayDetector(),
        ]

    # -- öffentliche Schnittstelle ------------------------------------------------------

    def resolve(self, gray: np.ndarray, path: Path | None = None) -> ScaleOutcome:
        """Maßstab für ein Graustufenbild bestimmen."""
        override = self._override_for(path)
        if override is not None:
            return ScaleOutcome(scale=override)

        candidates: list[OverlayCandidate] = []
        for detector in self.detectors:
            candidates.extend(detector.detect(gray, self.cfg))
        candidates.sort(key=lambda c: -c.geometry_score)

        outcome = ScaleOutcome(scale=None, candidates=candidates)
        if not candidates:
            outcome.rejections.append("kein Balken in einem weißen Kasten gefunden")
            return outcome

        engines = available_engines(self.cfg.ocr.engines)
        if not engines:
            outcome.rejections.append(
                "keine OCR-Engine verfügbar - die Beschriftung kann nicht gelesen werden"
            )
            return outcome

        for candidate in candidates:
            scale, reason = self._read_candidate(gray, candidate, engines)
            if scale is not None:
                outcome.scale = scale
                return outcome
            outcome.rejections.append(reason)

        return outcome

    # -- ein Kandidat -------------------------------------------------------------------

    def _read_candidate(
        self, gray: np.ndarray, candidate: OverlayCandidate, engines: list
    ) -> tuple[ScaleInfo | None, str]:
        if candidate.label is None:
            return None, f"Balken bei {candidate.bar.x},{candidate.bar.y}: keine Beschriftung"

        readings, seen_texts = self._collect_readings(gray, candidate, engines)
        if not readings:
            snippet = ", ".join(sorted({t for t in seen_texts if t})) or "nichts"
            return None, (
                f"Balken bei {candidate.bar.x},{candidate.bar.y}: Beschriftung nicht "
                f"auswertbar (gelesen: {snippet!r})"
            )

        winner, agreement, dissent = _vote(readings)
        parsed = winner.parsed

        um_per_px = parsed.value_um / candidate.bar_length_px
        if not (self.cfg.min_um_per_px <= um_per_px <= self.cfg.max_um_per_px):
            return None, (
                f"Balken bei {candidate.bar.x},{candidate.bar.y}: {um_per_px:.4g} µm/px "
                f"liegt außerhalb der harten Grenzen"
            )

        warnings: list[str] = list(candidate.notes)
        warnings.extend(parsed.corrections)

        # Konfidenz aus drei Quellen: wie sehr es nach einem Overlay aussieht, wie sicher
        # die OCR war, und wie einig sich die Lesungen untereinander waren.
        confidence = (
            0.35 * candidate.geometry_score
            + 0.30 * float(winner.ocr_confidence)
            + 0.35 * agreement
        )
        if parsed.was_corrected:
            confidence *= 0.8

        if dissent and self.cfg.ocr.cross_check:
            warnings.append(
                f"abweichende Lesungen wurden überstimmt: {dissent} "
                f"(Zustimmung für {parsed}: {agreement:.0%})"
            )

        if not (self.cfg.expected_min_um_per_px <= um_per_px <= self.cfg.expected_max_um_per_px):
            warnings.append(
                f"{um_per_px:.4g} µm/px liegt außerhalb des für Schliffbilder erwarteten "
                f"Bereichs ({self.cfg.expected_min_um_per_px:g} … "
                f"{self.cfg.expected_max_um_per_px:g}) - bitte am Kontrollbild prüfen"
            )
            confidence *= 0.9

        scale = ScaleInfo.from_bar(
            value_um=parsed.value_um,
            bar_length_px=candidate.bar_length_px,
            source=ScaleSource.OVERLAY_BAR,
            confidence=max(0.0, min(1.0, confidence)),
            label_text=winner.text.strip(),
            unit_text=parsed.unit,
            bar_box=candidate.bar,
            label_box=candidate.label,
            box=candidate.box,
            engine=winner.engine,
            warnings=tuple(warnings),
        )
        log.debug("Maßstab erkannt: %s", scale.describe())
        return scale, ""

    # -- die Befragung ------------------------------------------------------------------

    def _collect_readings(
        self, gray: np.ndarray, candidate: OverlayCandidate, engines: list
    ) -> tuple[list[Reading], list[str]]:
        """Jede Engine auf jeder Fassung lesen lassen - bis der Konsens klar ist.

        Abgebrochen wird, sobald ein Wert genug Stimmgewicht ohne Gegenstimme hat. Der
        Normalfall kostet damit eine Fassung; teuer wird nur, was auch schwierig ist.
        """
        ocr_cfg = self.cfg.ocr
        assert candidate.label is not None
        variants = patch_variants(
            gray, candidate.label, candidate.box, ocr_cfg, bar=candidate.bar,
        )

        readings: list[Reading] = []
        texts: list[str] = []
        for variant_name, patch in variants:
            for engine in engines:
                result = engine.read(patch)
                texts.append(result.text)
                parsed = parse_length(result.text)
                if parsed is None:
                    continue
                readings.append(
                    Reading(
                        parsed=parsed,
                        text=result.text,
                        engine=engine.name,
                        variant=variant_name,
                        ocr_confidence=float(result.confidence),
                        weight=ocr_cfg.engine_weights.get(engine.name, 0.7),
                    )
                )
            if _consensus_reached(readings, ocr_cfg.min_consensus_weight):
                break

        return readings, texts

    # -- Vorgaben aus der Konfiguration -------------------------------------------------

    def _override_for(self, path: Path | None) -> ScaleInfo | None:
        if path is None or not self.cfg.overrides:
            return None
        name = path.name
        for pattern, um_per_px in self.cfg.overrides.items():
            if re.search(pattern, name):
                log.info("Maßstab für %s aus der Konfiguration: %g µm/px", name, um_per_px)
                return ScaleInfo(
                    um_per_px=float(um_per_px),
                    source=ScaleSource.CONFIG_OVERRIDE,
                    confidence=1.0,
                    warnings=(f"Maßstab fest vorgegeben über das Muster {pattern!r}",),
                )
        return None


def _close(a: float, b: float, rel: float = 0.001) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b))


# --------------------------------------------------------------------------------------
# Abstimmung
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """Eine einzelne Lesung: wer hat was auf welcher Fassung gelesen."""

    parsed: ParsedLength
    text: str
    engine: str
    variant: str
    ocr_confidence: float
    weight: float

    @property
    def vote(self) -> float:
        """Stimmgewicht. Eine unsichere Lesung zählt weniger, aber nie gar nichts."""
        return self.weight * (0.5 + 0.5 * max(0.0, min(1.0, self.ocr_confidence)))


def _group(readings: list[Reading]) -> list[list[Reading]]:
    """Lesungen mit übereinstimmendem Wert zusammenfassen."""
    groups: list[list[Reading]] = []
    for reading in readings:
        for group in groups:
            if _close(group[0].parsed.value_um, reading.parsed.value_um):
                group.append(reading)
                break
        else:
            groups.append([reading])
    return groups


def _consensus_reached(readings: list[Reading], min_weight: float) -> bool:
    """Ist die Sache klar genug, um die Befragung zu beenden?

    Klar heißt: ein Wert hat genug Stimmgewicht **und** es gibt keine Gegenstimme. Schon
    eine einzige abweichende Lesung genügt, um weiterzufragen - sie ist der Hinweis, dass
    die Zeile schwierig ist, und genau dann lohnt der Aufwand.
    """
    if not readings:
        return False
    groups = _group(readings)
    if len(groups) > 1:
        return False
    return sum(r.vote for r in groups[0]) >= min_weight


def _vote(readings: list[Reading]) -> tuple[Reading, float, str]:
    """Gewinner, Zustimmungsanteil und eine Kurzfassung der Gegenstimmen."""
    groups = _group(readings)
    groups.sort(key=lambda g: -sum(r.vote for r in g))

    total = sum(r.vote for r in readings) or 1.0
    winning = groups[0]
    agreement = sum(r.vote for r in winning) / total

    # Innerhalb der Gewinnergruppe die selbstsicherste Lesung als Beleg zeigen.
    best = max(winning, key=lambda r: (r.ocr_confidence, r.weight))

    dissent = "; ".join(
        f"{group[0].parsed} ({', '.join(sorted({r.engine for r in group}))})"
        for group in groups[1:]
    )
    return best, agreement, dissent
