"""Pydantic-Schema der Konfiguration.

Strikt: unbekannte Schlüssel sind ein Fehler, damit ein Tippfehler nicht stillschweigend
den Default stehen lässt. Bislang gefüllt ist der Maßstabsteil (Phase P1); die übrigen
Abschnitte kommen mit den jeweiligen Modulen dazu.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------------------
# Maßstab
# --------------------------------------------------------------------------------------


class BarConfig(_Strict):
    """Geometrie des eingebrannten Overlays.

    Die Defaults sind an ImageJ-Maßstabsbalken kalibriert: ein massiver schwarzer Balken
    in einem Kasten aus reinem Weiß. Sie sind bewusst großzügig - aussortiert wird über
    die Kombination der Kriterien, nicht über einen einzelnen scharfen Wert.
    """

    ink_max: int = Field(
        120, ge=0, le=255,
        description="Bis hier gilt ein Pixel als Schrift oder Balken")
    box_white_min: int = Field(
        255, ge=0, le=255,
        description="Ab hier gilt ein Pixel als Kastenweiß. Bewusst bei 255: der Kasten "
                    "ist einfarbig reinweiß, und schon 254 reicht aus, damit er bei sehr "
                    "hellem Gefüge mit dem Untergrund verschmilzt und nicht mehr als "
                    "Rechteck erkennbar ist.",
    )
    box_white_fallback: int = Field(
        250, ge=0, le=255,
        description="Zweiter Versuch, falls mit box_white_min nichts gefunden wird. "
                    "Greift bei JPEG-komprimierten Overlays, deren Weiß nicht mehr "
                    "exakt 255 ist.",
    )
    env_white_min: int = Field(
        250, ge=0, le=255,
        description="Schwelle für die Prüfung 'liegt der Balken im Weißen'. Darf "
                    "toleranter sein als box_white_min - sie bewertet nur, sie entscheidet "
                    "nicht.",
    )

    min_bar_px: int = Field(10, ge=2, description="Kürzerer Balken wäre nicht sinnvoll messbar")
    max_bar_height_px: int = Field(40, ge=1)
    max_bar_height_frac: float = Field(0.05, gt=0, le=1.0)
    min_bar_aspect: float = Field(3.0, gt=0, description="Balken ist deutlich breiter als hoch")
    min_bar_fill: float = Field(
        0.85, gt=0, le=1.0,
        description="Füllgrad der Bounding Box. Ein Balken ist massiv, ein Kratzer nicht.",
    )
    min_env_white: float = Field(
        0.5, ge=0, le=1.0,
        description="Anteil Weiß direkt über und unter dem Balken. Der Balken liegt im Kasten.",
    )

    # Bauform "Balken zerschneidet den Kasten" - siehe scale/detectors/split_box.py.
    # Sie greift nur dort, wo der Balken NICHT von Weiß eingeschlossen ist; auf die
    # Regelbauform haben diese Werte keinen Einfluss.
    split_probe_px: int = Field(
        4, ge=1,
        description="So viele Zeilen über und unter dem Balken werden nach den beiden "
                    "Kastenhälften abgesucht. Mehr als eine, weil zwischen Balken und "
                    "Kastenweiß eine kantengeglättete Übergangszeile liegen kann, die "
                    "weder das eine noch das andere ist.",
    )
    min_split_cover: float = Field(
        0.9, gt=0, le=1.0,
        description="Anteil der Balkenbreite, den jede der beiden Kastenhälften tragen "
                    "muss. Ein heller Fleck, der den Balken nur streift, ist keine.",
    )
    max_split_offset: float = Field(
        0.15, ge=0, le=1.0,
        description="Zulässiger seitlicher Versatz der beiden Hälften gegeneinander, "
                    "als Anteil der Balkenbreite. Ein Kasten schließt an beiden Enden "
                    "bündig ab - zwei zufällige helle Flecken tun das nicht.",
    )

    min_box_rectangularity: float = Field(
        0.70, gt=0, le=1.0,
        description="Füllgrad der gefüllten Weißfläche in ihrer Bounding Box. Der Kasten "
                    "ist ein Rechteck; ein deutlich niedrigerer Wert heißt, dass die "
                    "Fläche in helle Bildbereiche ausgelaufen ist.",
    )
    max_box_width_factor: float = Field(
        8.0, gt=1.0, description="Kasten höchstens so viel breiter als der Balken")
    max_box_height_factor: float = Field(40.0, gt=1.0)
    max_box_area_frac: float = Field(0.35, gt=0, le=1.0)

    min_label_height_px: int = Field(3, ge=1)
    max_candidates: int = Field(8, ge=1, description="So viele Balkenkandidaten werden geprüft")


class OcrConfig(_Strict):
    """Lesen der Beschriftung.

    Gelesen wird nicht einmal, sondern mehrfach: jede Engine sieht die Zeile in mehreren
    Vergrößerungen. Der Grund ist messbar - keine einzelne Kombination aus Engine und
    Vergrößerung liest alle 14 Testbilder richtig, aber die Mehrheit über mehrere liegt
    durchweg richtig. Die Fehler der Engines sind unkorreliert, und genau das lässt sich
    ausnutzen.
    """

    engines: tuple[str, ...] = Field(
        ("rapidocr", "tesseract"),
        description="Welche Engines befragt werden. Die Reihenfolge entscheidet nicht - "
                    "das Ergebnis wird abgestimmt, nicht nach Rang vergeben.",
    )
    engine_weights: dict[str, float] = Field(
        default_factory=lambda: {"rapidocr": 1.0, "tesseract": 0.85},
        description="Stimmgewicht je Engine. Nicht aufgeführte Engines zählen mit 0.7.",
    )
    variant_heights: tuple[int, ...] = Field(
        (48, 40, 64),
        description="Zielhöhen der hochskalierten Textzeile, in dieser Reihenfolge "
                    "durchprobiert. Die erste ist der Normalfall; die weiteren werden "
                    "nur befragt, wenn noch kein klarer Konsens besteht.",
    )
    binarized_variant: bool = Field(
        True,
        description="Zusätzlich eine per Otsu binarisierte Fassung befragen. Hilft bei "
                    "kleiner Schrift, deren Kantenglättung beim Vergrößern verschmiert.",
    )
    min_consensus_weight: float = Field(
        1.5, gt=0,
        description="Ab diesem Stimmgewicht für einen Wert - und ohne Gegenstimme - wird "
                    "die Befragung abgebrochen. Hält den Normalfall schnell.",
    )

    cross_check: bool = Field(
        True,
        description="Abweichende Lesungen als Warnung vermerken, auch wenn sie überstimmt "
                    "wurden. Sie gehören ins Kontrollbild.",
    )
    target_text_height_px: int = Field(
        48, ge=8,
        description="Zielhöhe für einen einzelnen Leseversuch - genutzt, wenn ein "
                    "Ausschnitt außerhalb der Abstimmung aufbereitet wird.")
    max_upscale: float = Field(8.0, gt=1.0)
    pad_factor: float = Field(
        0.6, ge=0, description="Rand um die Textzeile, als Vielfaches ihrer Höhe. Wird "
                               "am Kastenrand abgeschnitten, damit kein Gefüge hineinrutscht.")
    quiet_zone_px: int = Field(
        24, ge=0, description="Weißer Rahmen um den Ausschnitt. Ohne Ruhezone liest die "
                              "OCR aus '5 mm' gerne nur 'mm'.")


class ScaleConfig(_Strict):
    """Maßstabserkennung insgesamt."""

    bar: BarConfig = BarConfig()
    ocr: OcrConfig = OcrConfig()

    min_um_per_px: float = Field(
        1e-6, gt=0, description="Harte Grenze: darunter wird verworfen")
    max_um_per_px: float = Field(1e12, gt=0, description="Harte Grenze: darüber wird verworfen")

    expected_min_um_per_px: float = Field(
        1e-3, gt=0,
        description="Erwartungsfenster für metallographische Aufnahmen. Außerhalb wird "
                    "nicht verworfen, sondern gewarnt - ein ungewöhnlicher Maßstab kann "
                    "richtig sein, aber er gehört gesehen.",
    )
    expected_max_um_per_px: float = Field(1e3, gt=0)

    exclusion_pad_px: int = Field(
        4, ge=0, description="Rand um das Overlay, der zusätzlich aus der Analyse fällt")

    overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Regex auf den Dateinamen -> µm/px. Greift vor jeder Erkennung.",
    )


# --------------------------------------------------------------------------------------
# Probe und Einbettmittel
# --------------------------------------------------------------------------------------


class LassoConfig(_Strict):
    """Einbettmittel als randoffener Saum.

    Die Defaults sind an den Bildern aus ``test/Einbett_Material`` kalibriert. Dort liegt
    das Harz je Bild 61 bis 178 Graustufen unter der Probe und ist 2,3- bis 8,6-mal
    glatter. **Über** die Bilder hinweg überlappen die Bereiche allerdings - der glatteste
    Probenbereich ist strukturärmer als das körnigste Harz. Feste Schwellen sind hier
    also wertlos; beide Schwellen werden je Bild aus dem Bild selbst bestimmt.
    """

    denoise_ksize: int = Field(
        5, ge=1, description="Median gegen Sensorrauschen, bevor die Textur gemessen wird")
    texture_ksize_px: int = Field(
        9, ge=3,
        description="Fenster der lokalen Standardabweichung, in Arbeitsauflösung. Groß "
                    "genug, um Gefügekörner zu erfassen, klein genug, um den Harzrand "
                    "nicht zu verschmieren.",
    )

    dark_method: str = Field(
        "otsu", pattern="^(otsu|multiotsu|percentile)$",
        description="Wie die Helligkeitsschwelle bestimmt wird. 'multiotsu' hilft, wenn "
                    "die Probe selbst stark unterschiedlich hell ist und der "
                    "Zweiklassen-Otsu mitten durch das Metall schneidet.",
    )
    dark_percentile: float = Field(25.0, gt=0, lt=100)
    multiotsu_classes: int = Field(3, ge=3, le=5)
    multiotsu_level: int = Field(0, ge=0)
    dark_offset: int = Field(
        0, description="Schwelle verschieben. Positiv = mehr gilt als Harz.")

    texture_method: str = Field(
        "otsu", pattern="^(otsu|percentile)$",
        description="Die Schwelle wird nur ueber die dunklen Pixel bestimmt - dort steht "
                    "genau die Frage an, um die es geht: Harz oder dunkles Gefuege?",
    )
    texture_percentile: float = Field(30.0, gt=0, lt=100)
    texture_clip_percentile: float = Field(
        99.0, gt=50.0, le=100.0,
        description="Robustes Maximum fuer die 8-Bit-Normierung vor Otsu. Ausreisser der "
                    "Texturkarte - Kanten, Kratzer - wuerden sonst den ganzen relevanten "
                    "Wertebereich in wenige Histogrammstufen stauchen.",
    )
    min_texture_separability: float = Field(
        0.45, ge=0, le=1.0,
        description="Otsus Trennschaerfe, ab der die Texturschwelle ueberhaupt angewandt "
                    "wird. Gibt es gar kein dunkles Gefuege, ist die Verteilung "
                    "einhuegelig - Otsu legt trotzdem eine Schwelle hinein und wuerde das "
                    "Harz selbst zerschneiden. Darunter wird das Kriterium abgeschaltet.",
    )

    core_erode_px: int = Field(
        6, ge=0,
        description="Wie weit vom Maskenrand weg der Kern gemessen wird, an dem das "
                    "Aussehen des Harzes geschaetzt wird. Der Rand ist der Uebergang zur "
                    "Probe und wuerde die Schaetzung verwaessern.",
    )
    gray_sigmas: float = Field(
        12.0, gt=0,
        description="Grautoleranz in mittleren Abweichungen. Bewusst weit: Harz hat oft "
                    "einen Helligkeitsverlauf ueber das Bild, und der gehoert dazu.",
    )
    texture_sigmas: float = Field(
        5.0, gt=0,
        description="Texturtoleranz in mittleren Abweichungen. Enger als die Grautoleranz: "
                    "die Struktur des Harzes schwankt kaum, waehrend Gefuege sie immer hat.",
    )
    min_gray_tolerance: float = Field(
        12.0, ge=0,
        description="Boden fuer die Grauwerttoleranz. Sehr gleichmaessiges Harz haette "
                    "sonst eine Streuung nahe null, und schon Bildrauschen fiele heraus.",
    )
    min_texture_tolerance: float = Field(2.0, ge=0, description="Boden fuer die Texturtoleranz")

    multi_field_max_depth: float = Field(
        0.8, gt=0, le=1.0,
        description="Nur bei MEHREREN Harzfeldern: wie weit eines in das Bild hineinreichen "
                    "darf (1.0 = exakt bis zur Bildmitte). Ein einzelnes Feld darf das "
                    "sehr wohl - die Probe kann ein kleiner Span in der Bildecke sein.",
    )

    close_radius_px: int = Field(
        10, ge=0,
        description="Schliesst den Saum ueber kleine helle Einschluesse hinweg - vor allem "
                    "ueber Luftblasen im Harz, deren Raender rau sind und die deshalb "
                    "pixelweise durchfallen.",
    )

    min_material_contrast: float = Field(
        40.0, ge=0,
        description="Mindestabstand in Graustufen zwischen dem gefundenen Harz und dem "
                    "Rest des Bildes. Einbettmittel ist ein anderer Werkstoff; an den "
                    "Testbildern betraegt der Abstand 61 bis 178 Graustufen. Ein "
                    "dunklerer Bereich desselben Gefueges schafft das nicht.",
    )
    min_area_frac: float = Field(
        0.02, gt=0, le=1.0,
        description="Mindestgröße eines Harzfeldes, Anteil der Bildfläche. Darunter ist "
                    "es eine angeschnittene Pore und bleibt bei der Probe.",
    )
    min_border_contact_frac: float = Field(
        0.12, ge=0, le=1.0,
        description="Wie viel des Bildumfangs ein Harzfeld berühren muss. Einbettmittel "
                    "ist ein Saum und liegt am Rand entlang; eine angeschnittene Pore "
                    "berührt den Rand nur auf einem kurzen Stück.",
    )
    max_hole_frac: float = Field(
        0.05, gt=0, le=1.0,
        description="Bis zu dieser Größe werden Löcher im Harz gefüllt - Luftblasen und "
                    "eingebrannte Overlays gehören zum Harz. Größere Löcher bleiben "
                    "offen: ist das Harz ein geschlossener Rahmen, wäre das Loch die "
                    "Probe selbst.",
    )


class SeedConfig(_Strict):
    """Saatpunkte für die ausbreitenden Verfahren.

    Bewusst zurückhaltend: die Saat muss das Harz nicht abdecken, nur darin liegen. Eine
    großzuegige Saat nimmt dem Verfahren genau die Arbeit ab, die geprueft werden soll.
    """

    border_band_frac: float = Field(
        0.10, gt=0, le=0.5,
        description="Breite des Randstreifens, in dem ueberhaupt nach Harzsaat gesucht "
                    "wird - als Anteil der kuerzeren Bildkante. Ein dunkles Gefuegeband "
                    "mitten in der Probe kann so gar nicht erst zur Saat werden.",
    )
    erode_px: int = Field(
        4, ge=0, description="Saat vom Rand her schrumpfen, damit sie sicher innen liegt")
    min_seed_frac: float = Field(
        0.0005, gt=0, le=1.0, description="Mindestgroesse eines Saatflecks")


class RandomWalkerConfig(_Strict):
    """Random Walker: Diffusion von der Saat aus."""

    beta: float = Field(
        130.0, gt=0,
        description="Wie stark Grauwertkanten den Weg sperren. Gross = Grenzen halten "
                    "besser, aber die Ausbreitung kommt schlechter um Stoerungen herum.",
    )
    max_px: int = Field(
        700, ge=200,
        description="Eigene, kleinere Arbeitsgroesse. Das Verfahren loest ein "
                    "Gleichungssystem ueber alle Pixel und waechst ueberproportional.",
    )
    use_texture: bool = Field(
        True,
        description="Auf einem Bild aus Helligkeit UND Textur rechnen statt nur auf Grau. "
                    "Der Unterschied Harz/dunkles Gefuege steckt gerade in der Textur.",
    )


class GrabCutConfig(_Strict):
    """GrabCut: Graph-Schnitt mit gelernten Farbmodellen."""

    iterations: int = Field(3, ge=1, le=10)
    max_px: int = Field(800, ge=200)


class ChanVeseConfig(_Strict):
    """Morphologischer Chan-Vese: Levelset auf Regionenstatistik."""

    iterations: int = Field(60, ge=5)
    smoothing: int = Field(
        3, ge=0, le=6, description="Glaettungsschritte je Iteration - haelt die Kontur ruhig")
    lambda1: float = Field(1.0, gt=0, description="Gewicht des Aussenbereichs")
    lambda2: float = Field(1.0, gt=0, description="Gewicht des Innenbereichs")
    max_px: int = Field(600, ge=200)


class BoundaryPathConfig(_Strict):
    """Grenzverfolgung als kuerzester Pfad quer durchs Bild."""

    max_px: int = Field(800, ge=200)
    gradient_weight: float = Field(
        1.0, ge=0, description="Wie stark die Kante den Pfad anzieht")
    smoothness: float = Field(
        0.35, ge=0,
        description="Grundkosten je Schritt. Hoeher = geraderer Pfad, der Umwege um "
                    "Stoerungen meidet.",
    )


class SpecimenConfig(_Strict):
    """Trennung Probe / Einbettmittel."""

    #: **Das Standardverfahren für alle weiteren Schritte.**
    #:
    #: Gewählt am 12 Bilder umfassenden Vergleichssatz (``specimen-compare``): grabcut
    #: deckt den Saum am vollständigsten ab und fängt auch die beiden Fälle, an denen
    #: lasso zu wenig findet. Die übrigen Verfahren bleiben registriert und über diesen
    #: Schlüssel erreichbar - sie sind nicht abgeschrieben, sondern nicht Standard.
    #:
    #: Bekannter Schwachpunkt dieser Wahl: in der Gegenprobe an Bildern **ohne**
    #: Einbettmittel meldet grabcut in 5 von 14 Fällen welches, lasso nur in einem.
    #: Wer auf diese Seite empfindlich ist, stellt hier auf ``lasso`` um.
    method: str = Field(
        "grabcut",
        description="Registrierter Segmentierer. grabcut ist Standard; daneben stehen "
                    "lasso, random_walker, watershed, chan_vese, boundary_path, "
                    "full_frame und largest_region zur Verfügung.",
    )

    work_max_px: int = Field(
        1200, ge=200,
        description="Längere Bildkante für die Berechnung. Die Harzgrenze ist eine grobe "
                    "Struktur; sie in voller Auflösung zu suchen kostet nur Zeit. Die "
                    "Maske wird am Ende wieder hochskaliert.",
    )
    min_specimen_frac: float = Field(
        0.02, ge=0, le=1.0,
        description="Probenstücke unterhalb dieser Größe werden verworfen")
    keep: str = Field(
        "all_above_min", pattern="^(largest|all_above_min)$",
        description="Ein Bild kann mehrere getrennte Probenstücke zeigen")
    erode_border_px: int = Field(
        0, ge=0,
        description="Probenrand schrumpfen. Die Kantenabrundung am Harzübergang ist "
                    "dunkel und würde sonst als Pore gezählt.",
    )

    lasso: LassoConfig = LassoConfig()
    seeds: SeedConfig = SeedConfig()
    random_walker: RandomWalkerConfig = RandomWalkerConfig()
    grabcut: GrabCutConfig = GrabCutConfig()
    chan_vese: ChanVeseConfig = ChanVeseConfig()
    boundary_path: BoundaryPathConfig = BoundaryPathConfig()


# --------------------------------------------------------------------------------------
# Porendetektion
# --------------------------------------------------------------------------------------


class BackgroundConfig(_Strict):
    """Schätzung des Untergrunds, gegen den eine Pore sich abheben muss."""

    kernel_px: int = Field(
        0, ge=0,
        description="Kernel der Hintergrundschätzung. 0 = automatisch aus der Bildgröße. "
                    "Er MUSS größer sein als die größte Pore, sonst wird die Pore selbst "
                    "als Hintergrund geschätzt und verschwindet aus dem Kontrastbild.",
    )
    kernel_divisor: int = Field(
        8, ge=2, description="automatisch = kürzere Bildkante / divisor")
    kernel_min_px: int = Field(31, ge=3)
    smooth_sigma_factor: float = Field(
        3.0, gt=0, description="Glättung der Schätzung, als Teiler des Kernels")


class LocalContrastConfig(_Strict):
    """Hintergrundabzug mit Hysterese - das Standardverfahren."""

    background: BackgroundConfig = BackgroundConfig()

    strict_method: str = Field(
        "mad", pattern="^(mad|otsu_masked|multiotsu_masked|percentile|fixed)$",
        description="Wie die strenge Schwelle auf dem Kontrastbild bestimmt wird. "
                    "Gerechnet wird immer nur über Pixel INNERHALB der Probe - eine "
                    "globale Schwelle wäre vom dunklen Einbettmittel dominiert. "
                    "'mad' ist der Standard und misst, wie weit ein Pixel aus dem "
                    "Grundrauschen des Kontrastbilds heraussticht. Otsu ist hier die "
                    "schlechtere Wahl: es teilt eine Verteilung in zwei Hälften, aber "
                    "die Kontrastwerte sind keine zwei Klassen - sie sind ein Berg um "
                    "null mit einem dünnen Ausläufer, und genau der Ausläufer sind die "
                    "Poren. Otsu schneidet mitten in den Berg.",
    )
    mad_sigmas: float = Field(
        6.0, gt=0,
        description="Wie viele robuste Streuungen über dem Median ein Pixel liegen muss, "
                    "um als Pore zu gelten. Groß = strenger.",
    )
    strict_percentile: float = Field(99.0, gt=50.0, lt=100.0)
    strict_fixed: float = Field(25.0, gt=0, description="Kontrast in Graustufen")
    multiotsu_classes: int = Field(3, ge=3, le=5)
    multiotsu_level: int = Field(
        -1, description="Welche Multi-Otsu-Schwelle. -1 = die oberste (dunkelste Klasse)")

    grow_factor: float = Field(
        0.5, gt=0, le=1.0,
        description="Lockere Schwelle als Anteil der strengen. Sie entscheidet, WIE WEIT "
                    "eine erkannte Pore reicht - nicht, ob sie eine ist. Zu klein, und "
                    "die Pore läuft in die Gefügesprenkelung aus.",
    )
    min_contrast: float = Field(
        8.0, ge=0,
        description="Harte Untergrenze in Graustufen. Was sich weniger vom Untergrund "
                    "abhebt, ist Rauschen - unabhängig davon, was die Schwelle sagt.",
    )


class ThresholdConfig(_Strict):
    """Schlichte Schwelle auf dem Grauwert, maskiert auf die Probe."""

    method: str = Field(
        "multiotsu_masked", pattern="^(otsu_masked|multiotsu_masked|fixed)$")
    fixed: int = Field(90, ge=0, le=255)
    multiotsu_classes: int = Field(3, ge=3, le=5)
    multiotsu_level: int = Field(
        0, ge=0,
        description="Bei geätztem Gefüge trennt der Zweiklassen-Otsu die dunkle "
                    "Sprenkelung vom hellen Grundgefüge, statt die Poren herauszulösen. "
                    "Echte Hohlräume sind die dunkelste Klasse - Stufe 0 löst sie heraus.",
    )
    offset: int = Field(0, description="Schwelle verschieben, positiv = mehr gilt als Pore")


class AdaptiveConfig(_Strict):
    """Sauvola - lokal adaptive Schwelle."""

    window_px: int = Field(51, ge=3, description="ungerade; Fenster der lokalen Statistik")
    k: float = Field(0.2, gt=0, description="Empfindlichkeit; größer = strenger")


class SplitConfig(_Strict):
    """Zusammengewachsene Porennester trennen."""

    enabled: bool = Field(
        True,
        description="Zwei berührende Poren als eine zu zählen verfälscht Anzahl, "
                    "Größenverteilung und die größte Pore - oft das Abnahmekriterium.",
    )
    min_distance_px: int = Field(
        3, ge=1,
        description="Mindestabstand zweier Zentren in der Distanztransformation. Klein "
                    "genug für benachbarte Poren, groß genug, um eine zerklüftete "
                    "Einzelpore nicht zu zerschneiden.",
    )
    min_area_ratio: float = Field(
        0.15, gt=0, le=0.5,
        description="Ein Teilstück unterhalb dieses Anteils am Ganzen ist ein Ausfransen "
                    "der Distanztransformation, keine eigene Pore.",
    )


class MergeConfig(_Strict):
    """Nahe beieinanderliegende Porenteile zu einer Pore zusammenführen."""

    enabled: bool = Field(
        True,
        description="Eine Pore, die als zwei Objekte im Label-Bild steht, verfälscht "
                    "Anzahl und Größenverteilung genauso wie zwei als eine gezählte.",
    )
    max_gap_px: int = Field(
        1, ge=0,
        description="Größter Spalt in Pixeln, über den zusammengeführt wird. 0 = nur "
                    "Teile, die sich berühren.",
    )
    bridge_gaps: bool = Field(
        True,
        description="Den Spalt der Pore zuschlagen, damit sie ein zusammenhängendes "
                    "Objekt mit einem Umriss ist.",
    )


class PoreConfig(_Strict):
    """Porendetektion insgesamt."""

    #: **Das Standardverfahren.** Eine Pore ist nicht durch einen Grauwert definiert,
    #: sondern dadurch, dass sie dunkler als ihre Umgebung ist - ``local_contrast`` macht
    #: genau das zur Entscheidungsgrundlage. Austausch über diesen Schlüssel; die übrigen
    #: Verfahren bleiben registriert.
    method: str = Field(
        "local_contrast",
        description="Registrierter Detektor: local_contrast (Standard) | threshold | adaptive",
    )

    local_contrast: LocalContrastConfig = LocalContrastConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    adaptive: AdaptiveConfig = AdaptiveConfig()
    split: SplitConfig = SplitConfig()
    merge: MergeConfig = MergeConfig()

    open_radius_px: int = Field(
        1, ge=0, description="Entfernt einzelne Rauschpixel, ohne kleine Poren zu kosten")
    close_radius_px: int = Field(0, ge=0)
    fill_holes: bool = Field(
        True,
        description="Der helle Reliefsaum und der Grauwertverlauf zum Porenboden hin "
                    "reißen sonst Löcher in die Pore.",
    )
    min_size_px: int = Field(
        4, ge=1,
        description="Reine Rauschgrenze, kein fachlicher Filter - der steht in analysis")

    exclude_edge_pores: bool = Field(
        False,
        description="Angeschnittene Poren verwerfen statt sie zu zählen und zu "
                    "kennzeichnen. Ihre wahre Fläche ist unbekannt, sie aber ganz zu "
                    "unterschlagen verfälscht die Porosität nach unten.",
    )


class FilterSpec(_Strict):
    """Ein Filtereintrag: Name, an/aus, Parameter."""

    name: str
    enabled: bool = True
    params: dict[str, float] = Field(default_factory=dict)


class AnalysisConfig(_Strict):
    """Bewertung der gefundenen Poren."""

    #: Reihenfolge = Auswertungsreihenfolge. Neue Filter werden in ``analysis/filters.py``
    #: registriert und hier nur noch eingetragen.
    filters: tuple[FilterSpec, ...] = Field(
        default_factory=lambda: (
            # -- Messbarkeit: ab wann ist ein Fleck überhaupt ein Objekt? ---------------
            # Bewusst in Pixeln. Das ist eine Frage der Abtastung, nicht des Maßstabs -
            # siehe analysis/geometry.py. Grenzen aus einer Prüfvorschrift ("unter x µm
            # wird nicht gewertet") gehören nach analysis/acceptance.py: hier angewandt
            # würden sie die Porosität senken und damit eine Werkstoffkennzahl still an
            # die Vorschrift anpassen.
            FilterSpec(
                name="min_diameter", enabled=True,
                params={"min_diameter_px": 4.0},
            ),
            # -- Form: was der Gestalt nach keine Pore sein kann ------------------------
            FilterSpec(
                name="scratch", enabled=True,
                params={"min_aspect_ratio": 4.0, "max_width_um": 4.0,
                        "max_width_px": 4.0, "min_solidity": 0.6},
            ),
            FilterSpec(name="compactness", enabled=True, params={"min_solidity": 0.40}),
            FilterSpec(name="roundness", enabled=True, params={"min_roundness": 0.20}),
            FilterSpec(name="extent", enabled=False, params={"min_extent": 0.30}),
            # -- Rauschgrenze und Ausreißer --------------------------------------------
            FilterSpec(name="min_area", enabled=True, params={"min_area_px": 6.0}),
            FilterSpec(name="max_area", enabled=False, params={"max_area_frac": 0.5}),
            # -- Zuschaltbar ------------------------------------------------------------
            # aspect_ratio geht im schaerferen scratch-Filter auf und ist nur noch als
            # grobes Werkzeug vorhanden. edge, contrast und die uebrigen siehe geometry.py
            # bzw. filters.py.
            FilterSpec(name="aspect_ratio", enabled=False, params={"max_aspect_ratio": 8.0}),
            FilterSpec(name="edge", enabled=False, params={"image_edge": 1.0}),
            FilterSpec(name="circularity", enabled=False, params={"min_circularity": 0.3}),
            FilterSpec(name="solidity", enabled=False, params={"min_solidity": 0.6}),
            FilterSpec(name="contrast", enabled=False, params={"min_contrast": 15.0}),
        ),
    )

    size_class_edges_um: tuple[float, ...] = Field(
        (0.0, 10.0, 25.0, 50.0, 100.0, 250.0),
        description="Untergrenzen der Größenklassen für den Bericht, in Mikrometern")


class CorrectionsConfig(_Strict):
    """Manuelle Korrekturen: von Hand entfernte oder wieder aufgenommene Poren."""

    enabled: bool = Field(
        True,
        description="Gespeicherte Korrekturen bei jedem Lauf anwenden. Aus = das rein "
                    "automatische Ergebnis, etwa zum Vergleich.",
    )
    directory: Path = Field(
        Path("data/korrekturen"),
        description="Ablage der Korrekturdateien, eine JSON je Bild. Relativ zum "
                    "Arbeitsverzeichnis.",
    )


class AppConfig(_Strict):
    """Die vollständige Laufkonfiguration.

    Die Reihenfolge der Abschnitte ist die Reihenfolge der Pipeline, und die ist nicht
    beliebig: Maßstab vor Beleuchtungskorrektur (das eingebrannte Overlay lebt von
    absoluten Grauwerten), Overlay-Ausschluss vor der Probenmaske, Probenmaske vor der
    Porendetektion. Siehe :mod:`poredet.core.pipeline`.
    """

    scale: ScaleConfig = ScaleConfig()
    specimen: SpecimenConfig = SpecimenConfig()
    pore: PoreConfig = PoreConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    corrections: CorrectionsConfig = CorrectionsConfig()

    require_scale: bool = Field(
        False,
        description="Bricht ein Bild ab, wenn kein Maßstab erkannt wurde. Standard ist "
                    "aus: ohne Maßstab wird in Pixeln weitergemessen und die "
                    "physikalischen Werte bleiben leer.",
    )
