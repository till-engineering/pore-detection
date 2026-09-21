"""Weißer Kasten mit Balken und Text (ImageJ-Stil).

Die Suche läuft bewusst **vom Balken aus**, nicht vom Kasten. Der naheliegende Weg -
erst den weißen Kasten finden - scheitert genau dort, wo er gebraucht wird: bei hellem
Gefüge verschmilzt der Kasten mit dem Untergrund und ist als zusammenhängende Fläche
nicht mehr abgrenzbar. Gemessen an den vorhandenen Testbildern findet dieser Weg 6 von
14 Overlays.

Der Balken dagegen ist ein extrem eigentümliches Objekt: ein massiver, waagerechter,
sehr breiter Block aus Schwarz, der ringsum von Weiß umgeben ist. So etwas entsteht in
einem Schliff nicht zufällig. Von ihm aus lässt sich der Kasten sicher aufspannen, denn
für den Kasten gilt eine Eigenschaft, die das Gefüge nie hat:

    **Der Kasten ist einfarbig reines Weiß** - gemessen an den Testbildern exakt 255,
    ohne eine einzige Abweichung. Gefüge besteht fast nur aus Zwischentönen, auch sehr
    helles.

Der Kasten ist damit eine Fläche aus reinem Weiß mit Löchern darin: Balken und Schrift.
Werden diese Löcher gefüllt, ist er ein massives Rechteck, und die Komponente, die den
Balken enthält, *ist* der Kasten.

Das Füllen ist hier der Kniff und nicht bloß Bequemlichkeit. Der naheliegende Weg - das
Rechteck Zeile für Zeile nach außen wachsen lassen, solange nur Weiß und Schrift
vorkommen - scheitert an der **Kantenglättung der Schrift**: jedes Zeichen ist von
Zwischentönen umsäumt, und die Textzeile bricht das Kriterium genau dort, wo der Kasten
noch weitergeht. Beim Füllen liegen dieselben Zwischentöne im Inneren eines Lochs und
stören nicht mehr.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - Abhängigkeit ist in pyproject deklariert
    raise ImportError("box_overlay benötigt opencv-python") from exc

from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as nd_label

from ...config.schema import BarConfig, ScaleConfig
from ...core.models import BBox
from ..base import OverlayCandidate


class BoxOverlayDetector:
    """Findet Maßstabs-Overlays der Bauform "schwarzer Balken in weißem Kasten"."""

    name = "box_overlay"

    def detect(self, gray: np.ndarray, cfg: ScaleConfig) -> list[OverlayCandidate]:
        bar_cfg = cfg.bar
        bars = _bar_candidates(gray, bar_cfg)
        if not bars:
            return []

        # Erst streng (Kastenweiß = exakt 255), bei Misserfolg toleranter. Die strenge
        # Schwelle ist die wichtigere: sie trennt den Kasten von sehr hellem Gefüge. Die
        # tolerante fängt Overlays auf, die durch JPEG-Kompression aufgeweicht wurden.
        for white_min in (bar_cfg.box_white_min, bar_cfg.box_white_fallback):
            candidates = self._detect_with(gray, bars, bar_cfg, white_min)
            if candidates:
                return candidates
            if white_min == bar_cfg.box_white_fallback:
                break
        return []

    def _detect_with(
        self,
        gray: np.ndarray,
        bars: list[tuple[BBox, float, tuple[str, ...]]],
        bar_cfg: BarConfig,
        white_min: int,
    ) -> list[OverlayCandidate]:
        # Einmal je Durchgang, nicht je Kandidat: das Füllen ist der teuerste Schritt.
        regions = _filled_white_regions(gray, white_min)

        candidates: list[OverlayCandidate] = []
        for bar, bar_score, bar_notes in bars:
            box, box_notes = _box_for_bar(regions, bar, gray.shape, bar_cfg)
            if box is None:
                continue
            if box.area > bar_cfg.max_box_area_frac * gray.size:
                continue
            if box.w <= bar.w:
                # Der Kasten muss den Balken umschließen, nicht mit ihm identisch sein.
                continue

            label, side = _find_label(gray, box, bar, bar_cfg)
            notes = bar_notes + box_notes
            if white_min != bar_cfg.box_white_min:
                notes += (f"Kasten erst mit abgesenkter Weißschwelle ({white_min}) gefunden",)
            if label is None:
                notes += ("keine Beschriftung im Kasten gefunden",)

            candidates.append(
                OverlayCandidate(
                    bar=bar,
                    box=box,
                    label=label,
                    label_side=side,
                    geometry_score=_score(bar, box, label, bar_score),
                    detector=self.name,
                    notes=notes,
                )
            )

        candidates.sort(key=lambda c: -c.geometry_score)
        return candidates


# --------------------------------------------------------------------------------------
# Schritt 1: der Balken
# --------------------------------------------------------------------------------------


def _bar_candidates(
    gray: np.ndarray, cfg: BarConfig
) -> list[tuple[BBox, float, tuple[str, ...]]]:
    """Massive, waagerechte, von Weiß umgebene Blöcke."""
    height, width = gray.shape
    ink = (gray <= cfg.ink_max).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, 8)

    max_bar_h = min(cfg.max_bar_height_px, max(1, int(cfg.max_bar_height_frac * height)))
    found: list[tuple[BBox, float, tuple[str, ...]]] = []

    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if w < cfg.min_bar_px or h < 1 or h > max_bar_h:
            continue
        if w / h < cfg.min_bar_aspect:
            continue
        if w >= width or h >= height:
            continue

        fill = area / float(w * h)
        if fill < cfg.min_bar_fill:
            continue

        white_frac = _environment_white(gray, x, y, w, h, cfg)
        if white_frac < cfg.min_env_white:
            continue

        bar = BBox(x, y, w, h)
        # Ein Balken ist massiv (fill), sehr breit (aspect) und liegt im Weißen (white).
        score = 0.5 * white_frac + 0.3 * fill + 0.2 * min(1.0, (w / h) / 10.0)
        notes: tuple[str, ...] = ()
        if white_frac < 0.75:
            notes += (f"Umgebung des Balkens nur zu {white_frac:.0%} weiß",)
        found.append((bar, score, notes))

    # Der breiteste Balken ist der wahrscheinlichste Maßstab; die Prüfung kostet wenig,
    # deshalb werden mehrere Kandidaten weiterverfolgt statt nur der beste geraten.
    found.sort(key=lambda item: (-item[1], -item[0].w))
    return found[: cfg.max_candidates]


def _environment_white(
    gray: np.ndarray, x: int, y: int, w: int, h: int, cfg: BarConfig
) -> float:
    """Anteil Kastenweiß direkt über und unter dem Block."""
    height = gray.shape[0]
    pad = max(2, h)
    above = gray[max(0, y - pad):y, x:x + w]
    below = gray[y + h:min(height, y + h + pad), x:x + w]
    if above.size == 0 or below.size == 0:
        return 0.0
    both = np.concatenate([above.ravel(), below.ravel()])
    return float((both >= cfg.env_white_min).mean())


# --------------------------------------------------------------------------------------
# Schritt 2: der Kasten
# --------------------------------------------------------------------------------------


def _filled_white_regions(gray: np.ndarray, white_min: int) -> np.ndarray:
    """Zusammenhängende Flächen aus Kastenweiß, mit gefüllten Löchern, als Label-Bild.

    Balken und Schrift sind Löcher in der weißen Fläche; nach dem Füllen ist der Kasten
    ein massives Rechteck. Das Gefüge bleibt davon unberührt: es erreicht das reine Weiß
    gar nicht erst, und die wenigen Stellen, die es tun, sind keine geschlossenen Ringe.
    """
    white = gray >= white_min
    filled = binary_fill_holes(white)
    labels, _count = nd_label(filled)
    return labels


def _box_for_bar(
    labels: np.ndarray, bar: BBox, shape: tuple[int, ...], cfg: BarConfig
) -> tuple[BBox | None, tuple[str, ...]]:
    """Die gefüllte weiße Fläche, in der dieser Balken liegt."""
    height, width = shape[:2]
    under_bar = labels[bar.slices()]
    ids, counts = np.unique(under_bar, return_counts=True)
    keep = ids > 0
    ids, counts = ids[keep], counts[keep]
    if ids.size == 0:
        return None, ()

    component = labels == ids[int(np.argmax(counts))]
    ys, xs = np.nonzero(component)
    box = BBox.from_corners(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    notes: tuple[str, ...] = ()
    rectangularity = float(component.sum()) / max(box.area, 1)
    if rectangularity < cfg.min_box_rectangularity:
        return None, ()
    if rectangularity < 0.95:
        notes += (
            (f"Kasten nur zu {rectangularity:.0%} rechteckig - vermutlich grenzt er "
             f"an ebenso reinweiße Bildbereiche und ist etwas zu groß geraten"),
        )

    # Auf ein plausibles Fenster um den Balken begrenzen. Greift, wenn der Kasten in
    # gleich helle Bildbereiche ausgelaufen ist; der Ausschnitt bleibt dann zu groß,
    # aber er bleibt brauchbar - das ist besser als gar kein Kasten.
    max_w = min(width, max(bar.w + 4, int(cfg.max_box_width_factor * bar.w)))
    max_h = min(height, max(bar.h + 4, int(cfg.max_box_height_factor * max(bar.h, 1))))
    if box.w > max_w or box.h > max_h:
        cx, cy = bar.center
        x0 = int(max(0, min(bar.x, cx - max_w / 2)))
        y0 = int(max(0, min(bar.y, cy - max_h / 2)))
        box = BBox.from_corners(
            x0, y0,
            int(min(width, max(bar.x2, x0 + max_w))),
            int(min(height, max(bar.y2, y0 + max_h))),
        )
        notes += ("Kasten auf ein Fenster um den Balken begrenzt",)

    if not box.contains(bar):
        return None, ()
    return box, notes


# --------------------------------------------------------------------------------------
# Schritt 3: die Beschriftung
# --------------------------------------------------------------------------------------


def _find_label(
    gray: np.ndarray, box: BBox, bar: BBox, cfg: BarConfig
) -> tuple[BBox | None, str | None]:
    """Schrift innerhalb des Kastens, ohne den Balken.

    Die Suche bleibt strikt im Kasten. Das ist der eigentliche Gewinn der Kastensuche:
    der Textausschnitt kann kein Gefüge enthalten, und die OCR sieht nur schwarze Schrift
    auf reinem Weiß.
    """
    sub = gray[box.slices()]
    ink = (sub <= cfg.ink_max).astype(np.uint8)

    # Den Balken ausblenden - er ist die Referenzlänge, nicht Teil der Beschriftung.
    bx0, by0 = bar.x - box.x, bar.y - box.y
    ink[max(0, by0):by0 + bar.h, max(0, bx0):bx0 + bar.w] = 0

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, 8)
    bar_center_y = (bar.y + bar.y2) / 2.0

    sides: dict[str, list[tuple[BBox, int]]] = {"above": [], "below": []}
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if h < cfg.min_label_height_px:
            continue
        if h >= box.h or w >= box.w:
            continue
        abs_box = BBox(box.x + x, box.y + y, w, h)
        side = "above" if (abs_box.y + abs_box.h / 2.0) < bar_center_y else "below"
        sides[side].append((abs_box, area))

    side = max(sides, key=lambda s: sum(area for _, area in sides[s]))
    parts = sides[side]
    if not parts:
        return None, None

    # Nur die Zeile, zu der das größte Zeichen gehört. Ist der Kasten in helle Bereiche
    # ausgelaufen, liegen sonst Gefügeflecken in derselben Bounding Box wie die Schrift,
    # und die OCR bekommt einen viel zu breiten Ausschnitt zu sehen.
    seed = max(parts, key=lambda item: item[1])[0]
    band_top = seed.y - 0.6 * seed.h
    band_bottom = seed.y2 + 0.6 * seed.h

    label = seed
    for part, _area in parts:
        center_y = part.center[1]
        if band_top <= center_y <= band_bottom:
            label = label.union(part)
    return label, side


# --------------------------------------------------------------------------------------
# Bewertung
# --------------------------------------------------------------------------------------


def _score(bar: BBox, box: BBox, label: BBox | None, bar_score: float) -> float:
    """Wie sehr sieht das nach einem Maßstabs-Overlay aus - allein von der Form her."""
    score = bar_score
    if label is not None:
        score += 0.25
        # Beschriftung und Balken sind mittig übereinander gesetzt.
        offset = abs(label.center[0] - bar.center[0]) / max(bar.w, 1)
        score += 0.1 * max(0.0, 1.0 - offset)
    # Der Kasten sitzt eng um seinen Inhalt.
    tightness = (bar.w * bar.h) / max(box.area, 1)
    score += 0.05 * min(1.0, tightness * 4.0)
    return float(min(1.0, score))
