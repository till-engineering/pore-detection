"""Einbettmittel als randoffener Saum.

Das Problem, an dem jede reine Grauwertschwelle scheitert: **Harz und Poren sind gleich
dunkel.** Beide sind schwarze Flächen im Bild, und keine Schwelle der Welt trennt sie
voneinander, weil sie sich in der Helligkeit nicht unterscheiden.

Sie unterscheiden sich in zwei anderen Dingen, und beide werden hier ausgenutzt:

**1. Harz ist strukturlos, Gefüge nicht.** Geätztes Metall besteht aus Körnern, Lamellen
und Schleifriefen - es hat überall Textur. Harz ist eine glatte, dunkle Fläche, bestenfalls
mit runden Luftblasen darin. Über die zehn Testbilder gemessen liegt das Harz je Bild 61
bis 178 Graustufen unter der Probe und ist 2,3- bis 8,6-mal glatter. Dieses zweite Signal
ist entscheidend, denn es trennt Harz von **dunklem Gefüge** - von Perlitbändern und
Ätzkontrast, die zwar dunkel, aber nie strukturlos sind.

**2. Harz ist nach außen offen, Poren sind es nicht.** Das Einbettmittel umschließt die
Probe; das Sichtfeld liegt innerhalb des Einbettlings. Was Harz ist, läuft deshalb über
den Bildrand hinaus. Eine Pore dagegen ist von Metall umschlossen. Das ist der
geometrische Hebel, und er ist unabhängig von jeder Helligkeit.

Aus beidem zusammen:

    Harz = dunkel **und** strukturlos **und** am Bildrand offen

Keines der drei Kriterien genügt allein. Dunkel allein nimmt die Poren mit, strukturlos
allein nimmt glatte Probenbereiche mit, randoffen allein nimmt jede angeschnittene Pore
mit. Zusammen tragen sie - aber sie tragen nicht bis zum Ende, und deshalb folgen noch
zwei Schritte, die beide aus derselben Einsicht stammen: **Harz ist ein Werkstoff, kein
Helligkeitsbereich.**

*Erst* wird gemessen, wie das gefundene Harz aussieht, und der Rest daran gehalten. Das
trennt ein Gefügeband ab, das zufällig dieselbe Schwelle unterschreitet - auch dann,
wenn es mit dem echten Saum zusammenhängt und von keiner Form- oder Lageregel zu fassen
wäre.

*Dann* wird gefragt, ob sich der Fund vom Rest des Bildes überhaupt unterscheidet.
Irgendein Bereich sieht immer am ehesten nach Saum aus; ohne diese letzte Prüfung meldete
das Verfahren in 10 von 14 Bildern **ohne jedes Einbettmittel** welches. Mit ihr sind es
noch eins - und das ist eine polarisierte Aufnahme mit echten schwarzen Ecken außerhalb
des Sichtfelds, die auszuschließen richtig ist.

Der Maßstab spielt hier keine Rolle - es wird nichts vermessen, nur getrennt. Das ist
Absicht: viele Schliffbilder kommen ohne Maßstabsbalken an.

Bekannte Grenze: ein dunkles Band **innerhalb** der Probe, das oben und unten den
Bildrand berührt und sich in Helligkeit und Struktur kaum vom Harz unterscheidet, wird
mit abgetrennt. Die Probe zerfällt dann in zwei Stücke - sichtbar an ``components``, und
genau deshalb steht die Zahl im Kontrollbild.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..einstellungen import Abschnitt
from . import postprocess
from . import signals as sig
from .base import SpecimenMask
from .signals import Signals


class LassoSegmenter:
    """Trennt Probe und Einbettmittel über Helligkeit, Textur und Randoffenheit."""

    name = "lasso"

    def segment(self, gray: np.ndarray, cfg: Abschnitt) -> SpecimenMask:
        return self.from_signals(sig.compute(gray, cfg), cfg)

    def from_signals(self, signals: Signals, cfg: Abschnitt) -> SpecimenMask:
        """Getrennter Einstieg, damit der Verfahrensvergleich die Signale teilen kann."""
        lasso = cfg.lasso
        candidate = signals.is_dark & signals.is_smooth

        # Hier wird bewusst NICHT geglättet. Beide naheliegenden Wege richten Schaden an:
        # Eine morphologische Öffnung frisst die schmalen Harzbrücken zwischen den
        # Luftblasen weg und zerlegt den Saum - an einem Testbild blieben von 60 % Harz
        # noch 34 % in drei Fragmenten. Ein Mehrheitsentscheid im Fenster wiederum
        # verbindet, was nicht zusammengehört: bei der Schweißnaht verschmolz ein dunkles
        # Gefügeband mit dem echten Saum zu einem einzigen Feld, womit die Tiefenregel
        # unten nicht mehr greifen konnte.
        # Was das Aufräumen leisten sollte - kleine Störungen aussortieren -, erledigen
        # die Flächen- und Randkriterien der Komponentenauswahl ohnehin, und die
        # verschieben keine Grenze.

        resin, notes = _border_open_regions(candidate, lasso)
        if not signals.separable:
            notes += (
                ("dunkle Bildbereiche ließen sich nicht in Harz und dunkles Gefüge "
                 "trennen - die Texturschwelle wurde nicht angewandt"),
            )

        # Zweiter Durchgang: Harz ist EIN Material. Der erste Durchgang arbeitet mit
        # Schwellen, die über das ganze Bild gelten; damit rutscht hier und da etwas mit,
        # das zwar dunkel und glatt ist, aber sichtbar anders aussieht. Jetzt, wo bekannt
        # ist, *wo* das Harz liegt, lässt sich sein Aussehen messen und der Rest daran
        # halten - auch ein Gefügeband, das mit dem echten Saum zusammenhängt und deshalb
        # von keiner Form- oder Lageregel zu trennen wäre.
        if resin.any():
            resin, refine_notes = _refine_to_material(
                resin, signals.smoothed, signals.texture, lasso
            )
            notes += refine_notes
            # Sofort wieder auffüllen, noch VOR der erneuten Auswahl: was die Verfeinerung
            # herausschneidet, sind zu einem guten Teil Luftblasen mitten im Harz.
            resin = postprocess.fill_small_holes(resin, lasso.max_hole_frac)
            resin, more_notes = _border_open_regions(resin, lasso)
            notes += more_notes

        return postprocess.finish(resin, signals, cfg, self.name, notes,
                                  debug={"candidate": candidate})


# --------------------------------------------------------------------------------------
# Auswahl über die Randoffenheit
# --------------------------------------------------------------------------------------


def _border_open_regions(
    candidate: np.ndarray, cfg: Abschnitt
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Von allen dunklen, glatten Feldern nur die behalten, die ein Saum sein können.

    Zwei Bedingungen, und beide sind nötig:

    * **groß genug** - ein Harzfeld nimmt einen nennenswerten Teil des Bildes ein,
    * **lang genug am Rand** - Einbettmittel liegt am Bildrand *entlang*. Eine
      angeschnittene Pore berührt den Rand nur auf einem kurzen Stück und fällt damit
      heraus, obwohl auch sie randoffen ist.
    """
    height, width = candidate.shape
    total = float(height * width)
    perimeter = 2.0 * (height + width)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    border_labels = np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )
    contact = np.bincount(border_labels, minlength=count)
    depth_map = sig.border_distance(height, width)
    half_span = min(height, width) / 2.0

    fields: list[tuple[int, float]] = []          # (Label, relative Tiefe)
    rejected_inner = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area / total < cfg.min_area_frac:
            continue
        if contact[index] / perimeter < cfg.min_border_contact_frac:
            # Groß, dunkel, glatt - aber nicht am Rand entlang. Das ist ein
            # Gefügebereich oder eine große angeschnittene Pore, kein Saum.
            rejected_inner += 1
            continue
        region = labels == index
        fields.append((index, float(depth_map[region].max()) / half_span))

    notes: tuple[str, ...] = ()
    if rejected_inner:
        notes += (
            (f"{rejected_inner} große dunkle Fläche(n) nicht als Einbettmittel gewertet - "
             f"sie liegen nicht am Bildrand entlang"),
        )

    # Die Tiefenregel. Ein **einzelnes** Harzfeld darf bis in die Bildmitte reichen - die
    # Probe kann ein kleiner Span in der Bildecke sein. Sind es **mehrere**, geht das
    # nicht mehr: dann durchquert die Probe das Bild und teilt den Saum in Randbänder,
    # und die liegen zwangsläufig am Rand. Ein Feld, das trotzdem bis in die Mitte
    # reicht, ist ein dunkles Gefügeband, das den Bildrand nur zufällig berührt.
    if len(fields) > 1:
        shallow = [f for f in fields if f[1] <= cfg.multi_field_max_depth]
        if len(shallow) < len(fields):
            deep = len(fields) - len(shallow)
            notes += (
                (f"{deep} randberührende Fläche(n) reichen bis in die Bildmitte und "
                 f"wurden verworfen - neben anderen Harzfeldern kann das kein Saum sein, "
                 f"sondern ein dunkles Gefügeband"),
            )
            fields = shallow

    resin = np.zeros_like(candidate)
    for index, _depth in fields:
        resin |= labels == index
    return resin, notes


def _refine_to_material(
    resin: np.ndarray, gray: np.ndarray, texture: np.ndarray, cfg: Abschnitt
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Das gefundene Harz auf ein einheitliches Aussehen einschränken.

    Gemessen wird am **Kern** des Fundes - dem Teil, der vom Rand der Maske weg liegt.
    Der Kern ist sicher Harz; die Ränder sind der Übergang und würden die Schätzung
    verwässern. Als Streuungsmaß dient die mittlere absolute Abweichung vom Median: sie
    lässt sich von den Ausreißern, um die es hier gerade geht, nicht verschieben.

    Die Toleranz hat einen Boden. Sehr gleichmäßiges Harz hätte sonst eine Streuung nahe
    null, und schon die normale Bildschwankung fiele heraus.
    """
    core = sig.erode(resin, cfg.core_erode_px)
    if core.sum() < 50:
        core = resin

    gray_center, gray_spread = _median_and_mad(gray[core])
    texture_center, texture_spread = _median_and_mad(texture[core])

    gray_limit = gray_center + max(cfg.gray_sigmas * gray_spread, cfg.min_gray_tolerance)
    texture_limit = texture_center + max(
        cfg.texture_sigmas * texture_spread, cfg.min_texture_tolerance
    )

    refined = resin & (gray <= gray_limit) & (texture <= texture_limit)
    dropped = int(resin.sum() - refined.sum())
    if dropped <= 0:
        return resin, ()

    share = dropped / float(resin.sum())
    notes = (
        (f"{share:.0%} des zunächst gefundenen Harzes sehen anders aus als dessen Kern "
         f"(Grau > {gray_limit:.0f} oder Textur > {texture_limit:.1f}) und wurden "
         f"verworfen"),
    ) if share >= 0.02 else ()
    return refined, notes



def _median_and_mad(values: np.ndarray) -> tuple[float, float]:
    """Median und mittlere absolute Abweichung davon."""
    if values.size == 0:
        return 0.0, 0.0
    center = float(np.median(values))
    return center, float(np.median(np.abs(values - center)))
