"""Geometrische Ausschlusskriterien: was der Form nach keine Pore sein kann.

Die Detektion findet dunkle Flecken. Ob ein dunkler Fleck eine Pore ist, entscheidet sich
aber nicht nur an der Helligkeit, sondern an der **Form** - und die trennt Dinge, die
kein Grauwert trennt:

* **Schleifriefen** sind lang und dünn. Eine Pore ist ein Hohlraum; sie hat eine
  Ausdehnung in beide Richtungen.
* **Ätzartefakte und Korngrenzennetze** sind fransig und verzweigt. Eine Pore ist kompakt.
* **Einzelne Rauschpixel** sind zu klein, um überhaupt eine Form zu haben.

Drei Dinge, die in diesem Modul anders gemacht sind als üblich - und der Grund dafür:

**1. Rundheit über Fläche und Feret, nicht über den Umfang.**

Die verbreitete Zirkularität ``4πA/U²`` ist bei kleinen Objekten unbrauchbar. Der Umfang
eines digitalisierten Rands wird durch die Treppenstufen systematisch überschätzt, und
zwar relativ umso stärker, je kleiner das Objekt ist. Eine kreisrunde Pore von 6 px
Durchmesser kommt damit auf eine Zirkularität um 0,6 - ein Filter bei 0,5 würde also
genau die kleinen, echten Poren wegwerfen und die großen Riefen behalten.

Die **Rundheit** ``4A/(π·d_feret²)`` verwendet nur Fläche und eine einzige Länge. Beide
sind gegen Pixelierung robust, und für den perfekten Kreis ergibt sich ebenfalls 1,0.

**2. Formfilter greifen erst ab einer Mindestgröße.**

Unterhalb von etwa 25 Pixeln ist jede Formkennzahl von der Rasterung dominiert und sagt
nichts über das Objekt aus. Solche Objekte werden von den Formfiltern **übersprungen**,
nicht verworfen - über ihre Berechtigung entscheidet allein die Größe.

**3. Der Riefenfilter fragt nach der Breite, nicht nur nach dem Verhältnis.**

Ein reines Seitenverhältnis trifft auch Lunker, die von Natur aus langgestreckt sind. Eine
Schleifriefe hat aber eine zusätzliche Eigenschaft: sie ist **absolut dünn** - wenige
Mikrometer breit, unabhängig von ihrer Länge. Und sie ist gerade, also nahe an ihrer
konvexen Hülle, während ein Lunker verzweigt ist. Aus allen dreien zusammen wird ein
Kriterium, das Riefen trifft und Lunker stehen lässt.
"""

from __future__ import annotations

import math

from ..core.models import Pore
from .filters import FilterContext, register

#: Unterhalb dieser Pixelzahl ist jede Formkennzahl von der Rasterung dominiert.
#: Formfilter lassen solche Objekte deshalb durch, statt über sie zu urteilen.
MIN_PIXELS_FOR_SHAPE = 25


def roundness(pore: Pore) -> float | None:
    """Flächenbezogene Rundheit ``4A/(π·d_feret²)`` - 1,0 für den Kreis.

    Robust gegen Pixelierung, weil weder Umfang noch Randverlauf eingehen. ``None``, wenn
    kein Feret-Durchmesser vorliegt.
    """
    if pore.feret_max_px <= 0:
        return None
    return float(4.0 * pore.area_px / (math.pi * pore.feret_max_px ** 2))


def _too_small_for_shape(pore: Pore) -> bool:
    return pore.area_px < MIN_PIXELS_FOR_SHAPE


# --------------------------------------------------------------------------------------
# Mindestgröße - "ab wann ist eine Pore überhaupt eine Pore?"
# --------------------------------------------------------------------------------------


@register("min_diameter")
def _min_diameter(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Untergrenze der **Messbarkeit**, über den äquivalenten Kreisdurchmesser.

    Der Durchmesser ist hier das richtige Maß, nicht die Fläche: er ist anschaulich, er
    steht in den Normen, und er skaliert linear.

    Die Grenze steht in **Pixeln**, und das ist keine Verlegenheitslösung mangels
    Maßstab. Gefragt ist hier nicht, ab welcher Größe eine Pore zu *werten* ist, sondern
    ab wann überhaupt etwas zu *erkennen* ist. Ein Fleck aus drei Pixeln hat keine Form,
    keinen belastbaren Durchmesser und keine belastbare Fläche - was man ihm zuschreibt,
    stammt aus dem Raster und nicht aus dem Werkstoff. Das hängt an der Abtastung, also
    an Pixeln, und ein erkannter Maßstab ändert daran nichts.

    Eine Grenze in Mikrometern wäre etwas anderes: eine **Vorschrift** ("Poren unter x µm
    werden nicht gewertet"). Sie gehört nach :mod:`poredet.analysis.acceptance` und nicht
    hierher, und zwar aus einem handfesten Grund - wer hier aussortiert, nimmt die Pore
    aus der Zählung, und damit sinkt die Porosität. Die Porosität ist aber eine
    Werkstoffkennzahl über alles Auflösbare; ob eine einzelne Pore nach Vorschrift zu
    werten ist, ist eine Frage der Abnahme und keine der Messung. Beides in einem Filter
    heißt, die Kennzahl still an die Vorschrift anzupassen.

    Bis hierher standen beide Grenzen als Entweder-oder nebeneinander, mit Vorrang für
    die Mikrometer. Das war verkehrt herum: unter die Pixelgrenze kommt man nie, eine
    Vorschrift kann immer nur strenger machen. Lag ein Maßstab vor, fiel die Pixelgrenze
    ersatzlos weg - bei einer groben Aufnahme (24 µm/px) stand die 1-µm-Grenze dann bei
    0,04 px, und der Filter war abgeschaltet. Ein erkannter Maßstab darf einen Filter
    nicht schwächen.
    """
    grenze_px = params.get("min_diameter_px", 0.0)
    if pore.equivalent_diameter_px < grenze_px:
        text = f"Durchmesser {pore.equivalent_diameter_px:.1f} px unter {grenze_px:g} px"
        # Der physikalische Wert wird mitgenannt, wo er vorliegt - als Auskunft, nicht
        # als Kriterium. Wer die Verwerfungsliste liest, will wissen, wie groß das war.
        if pore.equivalent_diameter_um is not None:
            text += f" ({pore.equivalent_diameter_um:.2f} µm)"
        return text
    return None


# --------------------------------------------------------------------------------------
# Schleifriefen
# --------------------------------------------------------------------------------------


@register("scratch")
def _scratch(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Schleifriefen und Kratzer - lang, dünn und gerade.

    Alle drei Bedingungen müssen zutreffen, und jede einzelne hat einen Grund:

    * **langgestreckt** (``min_aspect_ratio``) - das offensichtliche Merkmal, aber allein
      zu grob: Lunker sind es auch.
    * **absolut dünn** (``max_width``) - das eigentliche Unterscheidungsmerkmal. Eine
      Riefe ist ein Kratzer des Schleifkorns und damit wenige Mikrometer breit, egal wie
      lang sie ist. Ein langgestreckter Lunker ist deutlich breiter.
    * **gerade** (``min_solidity``) - eine Riefe liegt nah an ihrer konvexen Hülle, ein
      verzweigter Lunker nicht. Diese Bedingung **schützt** die Lunker: was fransig ist,
      wird nicht als Riefe verworfen, auch wenn es lang und schmal ist.

    ``max_width_um`` hat Vorrang vor ``max_width_px``, sobald ein Maßstab vorliegt - eine
    Breite in Mikrometern ist über Vergrößerungen hinweg dieselbe Aussage.
    """
    min_aspect = params.get("min_aspect_ratio", 4.0)
    if pore.aspect_ratio < min_aspect:
        return None

    breite_um = params.get("max_width_um")
    if breite_um is not None and ctx.um_per_px is not None:
        breite = pore.minor_axis_um or 0.0
        if breite > breite_um:
            return None
        breiten_text = f"Breite {breite:.2f} µm"
    else:
        breite_px = params.get("max_width_px", 0.0)
        if pore.minor_axis_px > breite_px:
            return None
        breiten_text = f"Breite {pore.minor_axis_px:.1f} px"

    min_solidity = params.get("min_solidity", 0.6)
    if pore.solidity < min_solidity:
        # Fransig und verzweigt - das ist ein Lunker, keine Riefe. Bleibt.
        return None

    return (f"Schleifriefe: {breiten_text} bei Seitenverhältnis "
            f"{pore.aspect_ratio:.1f} und Solidität {pore.solidity:.2f}")


# --------------------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------------------


@register("roundness")
def _roundness(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Flächenbezogene Rundheit - gegen langgestreckte und fransige Gebilde.

    Die robustere Alternative zur Zirkularität. Wird unterhalb von
    :data:`MIN_PIXELS_FOR_SHAPE` **nicht** angewandt, weil die Kennzahl dort nichts mehr
    aussagt.

    Der Wert ist mit Bedacht zu wählen: Gasporen liegen über 0,7, Lunker können bis 0,2
    herunterreichen. Ein scharfer Filter trennt also nicht Pore von Nicht-Pore, sondern
    Gaspore von Lunker - und wirft damit womöglich genau das weg, was gesucht wird.
    """
    if _too_small_for_shape(pore):
        return None
    wert = roundness(pore)
    if wert is None:
        return None
    grenze = params.get("min_roundness", 0.0)
    if wert < grenze:
        return f"Rundheit {wert:.2f} unter {grenze:g}"
    return None


@register("compactness")
def _compactness(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Solidität als Anteil an der konvexen Hülle - gegen Korngrenzennetze.

    Ein Netz aus dunklen Korngrenzen, das die Detektion als ein Objekt zusammenfasst, hat
    eine sehr geringe Solidität: es umschließt viel leere Fläche. Eine Pore - auch ein
    zerklüfteter Lunker - ist massiv.
    """
    if _too_small_for_shape(pore):
        return None
    grenze = params.get("min_solidity", 0.0)
    if pore.solidity < grenze:
        return f"Solidität {pore.solidity:.2f} unter {grenze:g}"
    return None


@register("extent")
def _extent(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Füllgrad der umschließenden Box - trennt kompakte von diagonal gestreckten Formen.

    Ergänzt die Solidität um einen Fall, den diese nicht sieht: ein schräg durchs Bild
    laufender Kratzer ist nah an seiner konvexen Hülle (hohe Solidität), füllt seine
    achsparallele Box aber kaum aus.
    """
    if _too_small_for_shape(pore):
        return None
    flaeche_box = float(pore.bbox.area)
    if flaeche_box <= 0:
        return None
    fuellgrad = pore.area_px / flaeche_box
    grenze = params.get("min_extent", 0.0)
    if fuellgrad < grenze:
        return f"Füllgrad der Box {fuellgrad:.2f} unter {grenze:g}"
    return None


@register("edge")
def _edge(pore: Pore, ctx: FilterContext, params: dict[str, float]) -> str | None:
    """Angeschnittene Poren verwerfen.

    Standardmäßig aus. Ihre wahre Größe ist zwar unbekannt, aber sie ganz zu
    unterschlagen verfälscht die Porosität systematisch nach unten - und zwar umso mehr,
    je größer die Poren im Verhältnis zum Bildausschnitt sind. Sinnvoll wird der Filter,
    wenn es um die **Größenverteilung** geht, nicht um die Porosität.
    """
    if params.get("image_edge", 1.0) and pore.touches_image_edge:
        return "am Bildrand angeschnitten"
    if params.get("specimen_edge", 0.0) and pore.touches_specimen_edge:
        return "am Probenrand - vermutlich Ausbruch aus der Präparation"
    return None
