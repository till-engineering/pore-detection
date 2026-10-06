"""Balken, der den weißen Kasten zerschneidet.

Der Regelfall in :mod:`kasten` ruht auf einer Eigenschaft: der Balken ist ein
**Loch** in der weißen Fläche. Werden die Löcher gefüllt, ist der Kasten ein massives
Rechteck, und die Komponente, die den Balken enthält, *ist* der Kasten. Ein Loch braucht
aber Weiß auf allen vier Seiten.

Es gibt Overlays, bei denen genau das nicht gilt: der Balken läuft von Kastenwand zu
Kastenwand. Links und rechts bleibt nichts oder eine einzelne, kantengeglättete
Pixelspalte - und die schließt das Loch nicht. Der Kasten zerfällt in zwei Hälften: oben
die Beschriftung, unten ein schmaler Streifen. ``box_overlay`` findet dann nur die obere
Hälfte, und in der steht kein Balken, sondern Text. Gemeldet wird "kein Balken in einem
weißen Kasten gefunden", obwohl beides im Bild steht.

Dieser Detektor setzt den Kasten aus den beiden Hälften wieder zusammen. Die Bauform ist
an einer Bedingung erkennbar, die sonst nicht eintritt:

    Über **und** unter dem Balken liegt je eine weiße Fläche, beide tragen seine volle
    Breite, und sie stehen bündig übereinander.

Zwei reinweiße Rechtecke, die einen massiven schwarzen Block einrahmen und dabei an
seinen Enden bündig abschließen, entstehen in einem Gefüge nicht zufällig.

**Warum nicht einfach die Weißschwelle absenken.** Das wäre der kürzere Weg: die
kantengeglätteten Randspalten gehen dann wieder als Weiß durch, der Balken ist wieder
ein Loch, und ``box_overlay`` greift. An einem gemessenen Beispiel kippt es bei 233.
Dagegen sprechen zwei Dinge. Der Wert stammt aus genau einem Bild und sagt nichts
darüber, wo das nächste kippt - er hängt an der Kantenglättung, nicht an der Bauform.
Und er weicht die Eigenschaft auf, auf der die ganze Kastensuche ruht: dass der Kasten
reines Weiß ist und Gefüge das nie erreicht. Ein Balken, der exakt so breit ist wie sein
Kasten, bliebe außerdem auch bei jeder Schwelle unerkannt - dort ist gar keine Randspalte
mehr da, die man durchgehen lassen könnte.

Der Detektor läuft **neben** ``box_overlay``, nicht statt seiner, und hält sich strikt
aus dessen Fällen heraus: liegt der Balken in einer gefüllten Weißfläche, ist er ein Loch
und gehört dort hin.
"""

from __future__ import annotations

import numpy as np

from ..einstellungen import Abschnitt
from ..modelle import BBox

# Balkensuche, Weißflächen, Beschriftung und Bewertung sind dieselben wie im Regelfall -
# unterschiedlich ist allein, wie aus dem Balken der Kasten wird. Doppelt geführt wären
# sie zwei Wahrheiten darüber, was ein Balken ist.
from .kasten import OverlayCandidate, _bar_candidates, _filled_white_regions, _find_label, _score

#: Abschlag auf die Formbewertung. Hier fehlt der geschlossene weiße Ring um den Balken,
#: das Formargument ist also schwächer als im Regelfall. Findet ein anderer Detektor
#: denselben Balken sauber eingefasst, soll dessen Kandidat zuerst gelesen werden.
SPLIT_PENALTY = 0.9


class SplitBoxOverlayDetector:
    """Findet Overlays, bei denen der Balken den weißen Kasten zerschneidet."""

    name = "split_box"

    def detect(self, gray: np.ndarray, cfg: Abschnitt) -> list[OverlayCandidate]:
        bar_cfg = cfg.bar
        bars = _bar_candidates(gray, bar_cfg)
        if not bars:
            return []

        # Dieselbe Staffelung wie im Regelfall: erst streng, dann tolerant.
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
        bar_cfg: Abschnitt,
        white_min: int,
    ) -> list[OverlayCandidate]:
        regions = _filled_white_regions(gray, white_min)

        candidates: list[OverlayCandidate] = []
        for bar, bar_score, bar_notes in bars:
            # Der Regelfall gehört box_overlay. Liegt der Balken in einer gefüllten
            # Weißfläche, ist er ein Loch - dann ist hier nichts zu tun, und zwei
            # Kandidaten für denselben Balken wären nur Arbeit für die OCR.
            if int(regions[bar.slices()].max()) > 0:
                continue

            box, box_notes = _box_from_halves(regions, bar, gray.shape, bar_cfg)
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
                    geometry_score=SPLIT_PENALTY * _score(bar, box, label, bar_score),
                    detector=self.name,
                    notes=notes,
                )
            )

        candidates.sort(key=lambda c: -c.geometry_score)
        return candidates


# --------------------------------------------------------------------------------------
# Den Kasten aus seinen beiden Hälften zusammensetzen
# --------------------------------------------------------------------------------------


def _half(
    labels: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> tuple[int, BBox] | None:
    """Die vorherrschende Weißfläche in einem Band über der Balkenbreite.

    Gesucht wird ein **Band** und nicht die eine Zeile direkt am Balken: zwischen Balken
    und Kastenweiß liegt oft eine kantengeglättete Übergangszeile, die weder das eine
    noch das andere ist. An ihr würde die Suche sonst scheitern.
    """
    if y1 <= y0 or x1 <= x0:
        return None
    band = labels[y0:y1, x0:x1]
    ids, counts = np.unique(band, return_counts=True)
    keep = ids > 0
    ids, counts = ids[keep], counts[keep]
    if ids.size == 0:
        return None

    best = int(ids[int(np.argmax(counts))])
    ys, xs = np.nonzero(labels == best)
    return best, BBox.from_corners(
        int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _box_from_halves(
    labels: np.ndarray, bar: BBox, shape: tuple[int, ...], cfg: Abschnitt
) -> tuple[BBox | None, tuple[str, ...]]:
    """Aus den Weißflächen über und unter dem Balken den Kasten aufspannen."""
    height, width = shape[:2]
    probe = max(cfg.split_probe_px, 1)

    oben = _half(labels, bar.x, bar.x2, max(0, bar.y - probe), bar.y)
    unten = _half(labels, bar.x, bar.x2, bar.y2, min(height, bar.y2 + probe))
    if oben is None or unten is None:
        return None, ()
    id_oben, kasten_oben = oben
    id_unten, kasten_unten = unten

    # Beide Hälften müssen den Balken der Breite nach tragen. Ein heller Fleck, der nur
    # ein Stück von ihm berührt, ist keine Kastenhälfte.
    for teil in (kasten_oben, kasten_unten):
        ueberlappung = min(teil.x2, bar.x2) - max(teil.x, bar.x)
        if ueberlappung < cfg.min_split_cover * bar.w:
            return None, ()

    # Und sie müssen bündig übereinanderstehen. Das ist die eigentliche Prüfung: zwei
    # unabhängige helle Flecken, die den Balken zufällig einrahmen, schließen an seinen
    # Enden nicht miteinander ab.
    toleranz = max(2.0, cfg.max_split_offset * bar.w)
    if (abs(kasten_oben.x - kasten_unten.x) > toleranz
            or abs(kasten_oben.x2 - kasten_unten.x2) > toleranz):
        return None, ()

    box = kasten_oben.union(kasten_unten).union(bar)
    if box.x < 0 or box.y < 0 or box.x2 > width or box.y2 > height:
        return None, ()

    # Ein Kasten ist rechteckig. Gezählt werden die beiden Hälften und der Balken -
    # zusammen füllen sie ihn aus, wenn es wirklich einer ist.
    teile = labels[box.slices()]
    belegt = int(np.isin(teile, list({id_oben, id_unten})).sum()) + bar.area
    rechteckigkeit = belegt / max(box.area, 1)
    if rechteckigkeit < cfg.min_box_rectangularity:
        return None, ()

    # Dieselben Fenstergrenzen wie im Regelfall, hier aber als Ausschluss und nicht als
    # Beschneidung: bei dieser Bauform ist der Kasten kaum breiter als der Balken. Ist er
    # es doch, sind die beiden Flächen kein Kasten, und ein zurechtgestutzter Ausschnitt
    # wäre eine Behauptung statt eines Fundes.
    max_w = min(width, max(bar.w + 4, int(cfg.max_box_width_factor * bar.w)))
    max_h = min(height, max(bar.h + 4, int(cfg.max_box_height_factor * max(bar.h, 1))))
    if box.w > max_w or box.h > max_h:
        return None, ()
    if not box.contains(bar):
        return None, ()

    notes: tuple[str, ...] = (
        ("Balken reicht bis an die Kastenwand - Kasten aus den Flächen über und unter "
         "ihm zusammengesetzt"),
    )
    if rechteckigkeit < 0.95:
        notes += (f"Kasten nur zu {rechteckigkeit:.0%} rechteckig",)
    return box, notes
