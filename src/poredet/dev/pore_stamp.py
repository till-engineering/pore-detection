"""Poren mit bekannter Wahrheit in echte Schliffbilder einpraegen.

**Herkunft:** uebernommen aus V1 (``Pore_Detection/src/pore_detection/synthetic.py``).
Der Rendering-Teil ist unveraendert - er ist erprobt und hat seinen eigenen Benchmark.
Angepasst wurden nur die drei Stellen, an denen das Modul an V1s Pipeline hing:

* ``add_pores`` bestimmt die Probenmaske jetzt ueber :mod:`poredet.specimen` statt ueber
  V1s Segmentierer. Das ist der eigentliche Gewinn der Uebernahme: V1 musste die
  Probenmaske umgehen, weil sie unbrauchbar war (siehe ``platzierungsmaske`` im alten
  ``scripts/stamp_pores.py``) - in V2 traegt sie.
* ``default_config`` und ``run`` sind entfallen; sie riefen V1s Pipeline auf. Die
  Auswertung gegen eine Detektion kommt mit Phase P3 zurueck, ``evaluate`` bleibt dafuer
  erhalten.

Der urspruengliche Kopf des Moduls folgt.

---

Generator fuer synthetische Schliffbilder mit bekannter Porenwahrheit.

Fuer die Porendetektion gilt dasselbe Argument wie fuer die Massstabserkennung (siehe
``scale/synthetic.py``): an einem echten Bild laesst sich nur beurteilen, ob das Ergebnis
*plausibel* aussieht. Hier ist dagegen jede Pore per Konstruktion bekannt - Lage, Flaeche,
Aequivalentdurchmesser -, und damit werden Trefferquote, Flaechenfehler und Porositaets-
fehler messbar statt schaetzbar.

**Warum keine schwarzen Kreise.** Ein Bild aus perfekt kreisrunden, gleichmaessig
schwarzen Scheiben auf hellem Grund testet nichts: jede Schwelle zwischen Poren- und
Gefuegegrauwert loest es. Die Faelle, an denen eine Detektion im Alltag scheitert, sehen
anders aus, und genau die bildet der Generator nach:

* **Form.** Gasporen sind rund, aber nicht kreisrund - der Rand wird als Summe weniger
  Fourier-Harmonischer moduliert. Lunker (``shrinkage``) sind zerklueftete, verzweigte
  Gebilde aus mehreren verschmolzenen Lappen; ihre Rundheit liegt weit unter 1.
* **Grauwertverlauf.** Eine Pore ist kein Flaechenfarbton. Zur Porenwand hin wird sie
  heller (Kantenverrundung beim Polieren), der Kraterboden traegt einen gerichteten
  Helligkeitsgradienten aus der schraegen Beleuchtung und eine eigene Textur.
* **Relief-Saum.** Um die Pore herum steht ein heller Ring - beim Polieren wird der Rand
  abgetragen und wirft das Licht anders zurueck. Er verschiebt die gemessene Flaeche.
* **Weiche Kante.** Der Uebergang ist durch die Optik verschliffen, nicht binaer. Die
  Wahrheit ist deshalb ueber die Deckung ``alpha >= 0.5`` definiert.
* **Teilfuellung.** Poren, in die Einbettharz gelaufen ist, sind nur noch mitteldunkel -
  der schwerste Fall, weil der Grauwertabstand zum Gefuege schrumpft.
* **Stoerobjekte.** Schleifriefen und dunkle Einschluesse sind dunkel, aber keine Poren.
  Sie werden mitgefuehrt (``distractor_mask``), damit Falschtreffer nicht nur gezaehlt,
  sondern auch ihrer Ursache zugeordnet werden koennen.

Nicht Gegenstand dieses Generators ist der Massstab: er wird als Wahrheit mitgeliefert
(``um_per_px``) und im Benchmark fest vorgegeben. Die Erkennung des eingebrannten
Overlays hat ihren eigenen Benchmark.

Aufruf des fertigen Benchmarks::

    python scripts/benchmark_pores.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------------------
# Wahrheit
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PoreTruth:
    """Eine gezeichnete Pore samt exakter Wahrheit.

    Die Flaeche ist die Zahl der Pixel mit einer Deckung von mindestens 50 % - dieselbe
    Konvention, nach der eine Segmentierung den weichen Rand aufteilen muesste.
    """

    label: int
    kind: str  # gas | shrinkage | micro
    area_px: int
    centroid_px: tuple[float, float]  # (x, y)
    radius_px: float  # Sollradius vor der Randmodulation
    floor_gray: float  # Grauwert am Kraterboden
    resin_filled: bool
    touches_specimen_edge: bool

    @property
    def equivalent_diameter_px(self) -> float:
        return 2.0 * math.sqrt(self.area_px / math.pi)


@dataclass
class SyntheticImage:
    """Ein synthetisches Schliffbild und alles, was darueber bekannt ist."""

    gray: np.ndarray  # uint8, HxW
    specimen_mask: np.ndarray  # bool - die wahre Probenflaeche
    pore_labels: np.ndarray  # int32, Label n gehoert zu pores[n-1]
    pores: list[PoreTruth]
    distractor_mask: np.ndarray  # bool - Riefen und Einschluesse, die keine Poren sind
    spec: "SpecimenSpec"

    @property
    def pore_count(self) -> int:
        return len(self.pores)

    @property
    def specimen_area_px(self) -> int:
        return int(np.count_nonzero(self.specimen_mask))

    @property
    def pore_area_px(self) -> int:
        return int(sum(p.area_px for p in self.pores))

    @property
    def porosity_pct(self) -> float:
        area = self.specimen_area_px
        return 100.0 * self.pore_area_px / area if area else 0.0


# --------------------------------------------------------------------------------------
# Beschreibung einer Szene
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecimenSpec:
    """Vollstaendige Beschreibung eines synthetischen Schliffbildes.

    Jedes Feld ist eine Dimension des Benchmarks: ``iter_specs`` faehrt sie einzeln ab,
    ``iter_cross_section`` kombiniert sie zufaellig.
    """

    width: int = 640
    height: int = 480
    um_per_px: float = 2.0

    # -- Gefuege und Einbettung ----------------------------------------------------------
    microstructure: str = "geaetzt"  # poliert | geaetzt | dendritisch | mehrphasig
    matrix_gray: float = 180.0
    embedding: bool = True  # dunkles Einbettmittel am Bildrand
    resin_gray: float = 38.0

    # -- Poren ---------------------------------------------------------------------------
    pore_count: int = 18
    pore_kinds: tuple[str, ...] = ("gas",)
    radius_px: tuple[float, float] = (7.0, 22.0)
    roughness: float = 1.0  # Faktor auf die Randmodulation
    floor_gray: tuple[float, float] = (10.0, 32.0)  # Grauwert am Kraterboden
    wall_lift: float = 30.0  # Aufhellung zur Porenwand hin
    rim_highlight: float = 0.10  # Relief-Saum als Faktor auf das Gefuege
    edge_softness_px: float = 1.2  # Breite des weichen Uebergangs
    resin_filled_frac: float = 0.0  # Anteil teilgefuellter (mitteldunkler) Poren
    # Zwei getrennte Randfaelle, weil sie verschiedene Fragen stellen:
    # ``edge_pores`` setzt Poren mittig auf die Probenkante - sie sind angeschnitten und
    # nur zur Haelfte im Bild. ``edge_clearance_px`` setzt dagegen *alle* Poren mit genau
    # diesem Abstand zwischen Porenrand und Probenkante ins Metall; damit laesst sich
    # ausmessen, ab welchem Abstand eine Pore wieder zuverlaessig umschlossen ist.
    edge_pores: int = 0
    edge_clearance_px: float | None = None

    # Mindestabstand einer gewoehnlichen Pore vom Probenrand. Er trennt zwei Fragen, die
    # sonst vermengt waeren: "wird eine Pore im Probeninneren gefunden" und "was passiert
    # am Rand". Eine randnahe Pore ist nicht mehr vollstaendig von Metall umschlossen,
    # und die Probenmaske entsteht genau ueber diese Umschliessung - sie faellt deshalb
    # aus systematischen Gruenden heraus, nicht weil die Schwelle zu hoch waere.
    # Die Dimension "randnah" misst diesen Uebergang eigens aus.
    edge_margin_px: float = 12.0

    # -- Stoerobjekte, die keine Poren sind ---------------------------------------------
    scratches: int = 0  # Schleifriefen
    inclusions: int = 0  # dunkle Einschluesse / Zweitphase

    # -- Aufnahme ------------------------------------------------------------------------
    illumination: str = "none"  # none | linear | vignette | hotspot
    illumination_strength: float = 0.0
    blur_sigma: float = 0.8
    noise_sigma: float = 3.0
    jpeg_quality: int | None = None

    def describe(self) -> str:
        kinds = "+".join(self.pore_kinds)
        return (
            f"{self.microstructure:<12} | {kinds:<14} | {self.pore_count:>2} Poren "
            f"r={self.radius_px[0]:.0f}-{self.radius_px[1]:.0f}px | "
            f"Boden {self.floor_gray[0]:.0f}-{self.floor_gray[1]:.0f} | "
            f"Saum {self.rim_highlight:.2f} | Kante {self.edge_softness_px:.1f}px | "
            f"Harz {self.resin_filled_frac:.0%} | "
            f"{self.scratches} Riefen, {self.inclusions} Einschl. | "
            f"{self.illumination}{self.illumination_strength:.0%} | "
            f"Rausch {self.noise_sigma:.0f}"
            + (f" | JPEG {self.jpeg_quality}" if self.jpeg_quality else "")
        )


# --------------------------------------------------------------------------------------
# Rendern
# --------------------------------------------------------------------------------------


def render(spec: SpecimenSpec, seed: int = 0) -> SyntheticImage:
    """Erzeugt ein Schliffbild samt Wahrheit.

    Die Reihenfolge folgt der physikalischen Kette: Szene zeichnen, dann Beleuchtung,
    dann Optik (Unschaerfe), dann Sensorrauschen, dann Kompression. Umgekehrt waere das
    Rauschen mit verschliffen und das Bild unrealistisch sauber.
    """
    rng = np.random.default_rng(seed)
    height, width = spec.height, spec.width

    specimen = _specimen_mask(spec, rng)
    image, second_phase = _microstructure(spec, rng)
    image[~specimen] = _resin(spec, rng)[~specimen]

    distractors = second_phase & specimen
    _draw_scratches(image, distractors, specimen, spec, rng)
    _draw_inclusions(image, distractors, specimen, spec, rng)

    pore_labels, pores = _draw_pores(image, specimen, spec, rng)

    # Eine Riefe unter einer Pore ist keine Riefe mehr, sondern Porenrand.
    distractors &= pore_labels == 0

    image = _illuminate(image, spec)
    if spec.blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), spec.blur_sigma)
    if spec.noise_sigma > 0:
        image = image + rng.normal(0.0, spec.noise_sigma, size=image.shape).astype(np.float32)

    gray = np.clip(image, 0, 255).astype(np.uint8)
    if spec.jpeg_quality is not None:
        ok, buffer = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), spec.jpeg_quality])
        if ok:
            gray = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    return SyntheticImage(
        gray=gray,
        specimen_mask=specimen,
        pore_labels=pore_labels,
        pores=pores,
        distractor_mask=distractors,
        spec=spec,
    )


def add_pores(
    gray: np.ndarray,
    spec: SpecimenSpec,
    seed: int = 0,
    specimen_mask: np.ndarray | None = None,
    exclusion: np.ndarray | None = None,
) -> SyntheticImage:
    """Praegt einem **echten** Schliffbild Poren mit bekannter Wahrheit auf.

    Das ist die schaerfste Form des Benchmarks: das Gefuege ist keine Nachbildung,
    sondern echt - mit allen Eigenheiten, die ein Generator nicht trifft - und trotzdem
    ist jede Pore nach Lage und Flaeche exakt bekannt.

    Gezeichnet wird ausschliesslich innerhalb von ``specimen_mask`` und ausserhalb von
    ``exclusion`` (dem eingebrannten Massstabs-Overlay). Ohne Angabe wird die Probenmaske
    mit den Defaults der Pipeline bestimmt. Damit ist die Wahrheit ausdruecklich auf die
    Frage bezogen, die hier geprueft wird - *wird eine Pore im Metall gefunden und richtig
    vermessen* -, und nicht auf die Guete der Probenmaske; die hat ihre eigenen Tests.

    Anders als ``render`` wird weder global geglaettet noch global verrauscht: das echte
    Bild bringt Rauschen und Schaerfe schon mit. Geglaettet wird nur der Saum um die
    eingesetzten Poren, damit ihre Kante zur Schaerfe des Bildes passt und sie nicht
    aufgeklebt wirken.
    """
    height, width = gray.shape[:2]
    spec = replace(spec, width=width, height=height)
    rng = np.random.default_rng(seed)

    if specimen_mask is None:
        from ..specimen import segment as _segment

        specimen_mask = _segment(gray).specimen

    valid = specimen_mask.copy()
    if exclusion is not None:
        valid &= ~exclusion

    image = gray.astype(np.float32)
    pore_labels, pores = _draw_pores(image, valid, spec, rng)

    if spec.blur_sigma > 0 and pores:
        saum = cv2.dilate(
            (pore_labels > 0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ).astype(bool)
        weich = cv2.GaussianBlur(image, (0, 0), spec.blur_sigma)
        image = np.where(saum, weich, image)

    return SyntheticImage(
        gray=np.clip(image, 0, 255).astype(np.uint8),
        specimen_mask=valid,
        pore_labels=pore_labels,
        pores=pores,
        distractor_mask=np.zeros((height, width), dtype=bool),
        spec=spec,
    )


# --------------------------------------------------------------------------------------
# Probe und Einbettmittel
# --------------------------------------------------------------------------------------


def _specimen_mask(spec: SpecimenSpec, rng: np.random.Generator) -> np.ndarray:
    """Die wahre Probenflaeche.

    Ohne Einbettmittel fuellt die Probe das ganze Bild. Mit Einbettmittel ist sie ein
    unregelmaessiges Vieleck - eine exakte Bildmitte waere zu freundlich, weil der
    Detektor dann schon aus der Geometrie raten koennte.
    """
    height, width = spec.height, spec.width
    if not spec.embedding:
        return np.ones((height, width), dtype=bool)

    margin_x = width * rng.uniform(0.07, 0.13)
    margin_y = height * rng.uniform(0.07, 0.13)
    points = []
    for fx, fy in ((0, 0), (1, 0), (1, 1), (0, 1)):
        x = margin_x + fx * (width - 2 * margin_x) + rng.uniform(-0.03, 0.03) * width
        y = margin_y + fy * (height - 2 * margin_y) + rng.uniform(-0.03, 0.03) * height
        points.append((x, y))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask.astype(bool)


def _resin(spec: SpecimenSpec, rng: np.random.Generator) -> np.ndarray:
    """Einbettharz: dunkel und koernig - und genauso dunkel wie eine Pore.

    Das ist der Grund, warum die Probenmaske geometrisch und nicht ueber den Grauwert
    bestimmt wird.
    """
    base = rng.normal(spec.resin_gray, 11.0, size=(spec.height, spec.width))
    return base.astype(np.float32)


# --------------------------------------------------------------------------------------
# Gefuege
# --------------------------------------------------------------------------------------


def _microstructure(
    spec: SpecimenSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Das Gefuege und die Maske seiner dunklen, porenaehnlichen Bestandteile.

    Die zweite Rueckgabe ist nicht kosmetisch: nur mit ihr laesst sich ein Falschtreffer
    der Ursache zuordnen - dunkle Zweitphase oder wirklich aus dem Nichts.
    """
    kind = spec.microstructure
    height, width = spec.height, spec.width
    base = spec.matrix_gray
    nothing = np.zeros((height, width), dtype=bool)

    if kind == "poliert":
        # Der freundliche Fall: nahezu strukturfrei, nur Sensorrauschen.
        return rng.normal(base, 2.5, size=(height, width)).astype(np.float32), nothing

    if kind == "geaetzt":
        return _grains(spec, rng, contrast=7.0, boundary=28.0), nothing

    if kind == "mehrphasig":
        # Geaetztes Gefuege plus dunkle Zweitphase - die Zweitphase ist dunkel, aber
        # kein Hohlraum. Genau hier zaehlt eine reine Grauwertschwelle zu viel.
        image = _grains(spec, rng, contrast=9.0, boundary=32.0)
        phase = np.zeros((height, width), dtype=np.uint8)
        for _ in range(int(height * width / 700)):
            cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
            axes = (int(rng.integers(2, 6)), int(rng.integers(2, 5)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(image, (cx, cy), axes, angle, 0, 360, float(rng.uniform(85, 125)), -1)
            cv2.ellipse(phase, (cx, cy), axes, angle, 0, 360, 1, -1)
        return cv2.GaussianBlur(image, (0, 0), 0.5), phase.astype(bool)

    if kind == "dendritisch":
        # Nachbildung von final_test.jpg: helle, zusammenhaengende Dendritenarme in einer
        # dunkleren, dicht gesprenkelten Matrix. Die Arme entstehen als Zufallspfade mit
        # dickem Strich - einzelne Ellipsen waeren zu kurz und zu regelmaessig.
        image = rng.normal(base - 40.0, 10.0, size=(height, width)).astype(np.float32)
        for _ in range(int(height * width / 90)):
            cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
            cv2.circle(image, (cx, cy), int(rng.integers(1, 3)), float(base - 75), -1)

        for _ in range(int(height * width / 2600)):
            x, y = float(rng.integers(0, width)), float(rng.integers(0, height))
            angle = float(rng.normal(math.pi / 2, 0.35))
            thickness = int(rng.integers(3, 9))
            for _step in range(int(rng.integers(3, 9))):
                angle += float(rng.normal(0, 0.25))
                length = float(rng.uniform(6, 18))
                nx, ny = x + math.cos(angle) * length, y + math.sin(angle) * length
                cv2.line(
                    image, (int(x), int(y)), (int(nx), int(ny)),
                    float(base + rng.uniform(10, 30)), thickness,
                )
                x, y = nx, ny
        return cv2.GaussianBlur(image, (0, 0), 0.8), nothing

    raise ValueError(f"Unbekanntes Gefuege: {kind!r}")


def _grains(
    spec: SpecimenSpec, rng: np.random.Generator, contrast: float, boundary: float
) -> np.ndarray:
    """Koerner als Voronoi-Zerlegung, jedes mit eigenem Grauwert, Korngrenzen dunkel.

    Die Korngrenzen sind der eigentliche Stolperstein: sie sind dunkle, duenne, verzweigte
    Linien - also genau das, was eine zu lockere Schwelle als Pore aufsammelt.
    """
    height, width = spec.height, spec.width
    count = max(8, int(height * width / 700))

    seeds = np.zeros((height, width), dtype=bool)
    ys = rng.integers(0, height, size=count)
    xs = rng.integers(0, width, size=count)
    seeds[ys, xs] = True

    _dist, (iy, ix) = ndi.distance_transform_edt(~seeds, return_indices=True)
    grain = (iy.astype(np.int64) * width + ix.astype(np.int64))

    levels = rng.normal(spec.matrix_gray, contrast, size=(height * width,))
    image = levels[grain % levels.size].astype(np.float32)

    edges = ndi.grey_dilation(grain, size=3) != ndi.grey_erosion(grain, size=3)
    image[edges] -= boundary

    image += rng.normal(0.0, 3.5, size=image.shape).astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), 0.6)


# --------------------------------------------------------------------------------------
# Stoerobjekte - dunkel, aber keine Poren
# --------------------------------------------------------------------------------------


def _draw_scratches(
    image: np.ndarray,
    distractors: np.ndarray,
    specimen: np.ndarray,
    spec: SpecimenSpec,
    rng: np.random.Generator,
) -> None:
    """Schleifriefen: lang, duenn, gerade. Kennzeichen ist das Achsenverhaeltnis."""
    if spec.scratches <= 0:
        return
    height, width = spec.height, spec.width
    layer = np.zeros((height, width), dtype=np.float32)

    for _ in range(spec.scratches):
        angle = float(rng.uniform(0, math.pi))
        length = float(rng.uniform(0.15, 0.55)) * min(height, width)
        cx, cy = float(rng.uniform(0, width)), float(rng.uniform(0, height))
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        depth = float(rng.uniform(35, 75))
        cv2.line(
            layer,
            (int(cx - dx), int(cy - dy)),
            (int(cx + dx), int(cy + dy)),
            depth,
            thickness=int(rng.integers(1, 3)),
        )

    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    hit = (layer > 8.0) & specimen
    image -= layer * specimen
    distractors |= hit


def _draw_inclusions(
    image: np.ndarray,
    distractors: np.ndarray,
    specimen: np.ndarray,
    spec: SpecimenSpec,
    rng: np.random.Generator,
) -> None:
    """Einschluesse: kompakt und dunkel, aber deutlich heller als ein Hohlraum.

    Der schwerste Falschtreffer-Kandidat, weil er in Form und Groesse einer Pore gleicht.
    Unterscheidbar ist er allein am Grauwert - das ist die Aufgabe des ``intensity``-Filters.
    """
    if spec.inclusions <= 0:
        return
    height, width = spec.height, spec.width

    for _ in range(spec.inclusions):
        radius = float(rng.uniform(3.0, 9.0))
        cx = float(rng.uniform(radius + 2, width - radius - 2))
        cy = float(rng.uniform(radius + 2, height - radius - 2))
        if not specimen[int(cy), int(cx)]:
            continue

        alpha, (x0, y0) = _lobe_alpha(
            cx, cy, radius, _harmonics(rng, 3, 0.10, 0.20), rng.uniform(0.7, 1.4),
            rng.uniform(0, math.pi), 1.0, width, height,
        )
        if alpha is None:
            continue
        region = image[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]]
        keep = specimen[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]]
        grey = float(rng.uniform(70, 115))
        blended = region * (1.0 - alpha) + grey * alpha
        region[keep] = blended[keep]
        distractors[y0 : y0 + alpha.shape[0], x0 : x0 + alpha.shape[1]] |= (alpha >= 0.5) & keep


# --------------------------------------------------------------------------------------
# Poren
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Placement:
    kind: str
    cx: float
    cy: float
    radius: float
    at_edge: bool


def _draw_pores(
    image: np.ndarray, specimen: np.ndarray, spec: SpecimenSpec, rng: np.random.Generator
) -> tuple[np.ndarray, list[PoreTruth]]:
    """Verteilt und zeichnet die Poren.

    Plazierung und Zeichnen gehoeren zusammen, weil die Bedingung an die Plazierung nur
    an der fertig gezeichneten Maske pruefbar ist: wie weit ein Lunker reicht, steht erst
    fest, wenn seine Lappen gewuerfelt sind. Eine Abschaetzung ueber den Sollradius hat
    genau hier versagt und zwei Poren aneinandergrenzen lassen - in der Wahrheit zwei
    Objekte, im Bild eines, und der Benchmark haette der Detektion den Fehler angelastet.

    Verworfen wird ein Wurf, wenn die Pore eine andere beruehrt, in zwei Stuecke zerfaellt
    oder (ausser bei den bewusst angeschnittenen Randporen) nicht ganz in der Probe liegt.
    """
    height, width = spec.height, spec.width
    labels = np.zeros((height, width), dtype=np.int32)
    pores: list[PoreTruth] = []

    inner = _shrunk(specimen, spec.edge_margin_px)
    randabstand = ndi.distance_transform_edt(specimen)
    # Sicherheitssaum um bereits gezeichnete Poren; zwei Poren duerfen sich auch nicht
    # diagonal beruehren, sonst verschmelzen sie beim Labeln zu einer.
    belegt = np.zeros((height, width), dtype=bool)

    edge_left = spec.edge_pores
    radius_lo, radius_hi = spec.radius_px

    for index in range(spec.pore_count):
        kind = spec.pore_kinds[index % len(spec.pore_kinds)]
        at_edge = edge_left > 0
        if at_edge:
            edge_left -= 1

        for _attempt in range(60):
            placement = _candidate(spec, kind, at_edge, randabstand, rng, radius_lo, radius_hi)
            if placement is None:
                break

            alpha, normalized, (x0, y0) = _pore_shape(placement, spec, rng)
            if alpha is None:
                continue

            ph, pw = alpha.shape
            fenster = (slice(y0, y0 + ph), slice(x0, x0 + pw))
            core = (alpha >= 0.5) & specimen[fenster]
            if not core.any() or belegt[fenster][core].any():
                continue
            frei_platziert = not at_edge and spec.edge_clearance_px is None
            if frei_platziert and (core & ~inner[fenster]).any():
                continue
            if _stuecke(core) != 1:
                continue

            truth = _paint(image, labels, specimen, alpha, normalized, core, fenster,
                           placement, len(pores) + 1, spec, rng)
            pores.append(truth)
            belegt[fenster] |= ndi.binary_dilation(core, np.ones((3, 3), bool), iterations=2)
            break

    return labels, pores


def _shrunk(mask: np.ndarray, radius: float) -> np.ndarray:
    """Die Maske, um ``radius`` Pixel geschrumpft."""
    if radius < 0.5:
        return mask
    size = 2 * int(round(radius)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def _auf_abstand(
    randabstand: np.ndarray, ziel: float, rng: np.random.Generator
) -> tuple[float, float] | None:
    """Ein zufaelliger Punkt, dessen Abstand zum Probenrand ``ziel`` betraegt.

    Damit laesst sich der Uebergang am Probenrand ausmessen, statt ihn dem Zufall zu
    ueberlassen: bei gleichverteilter Lage faellt kaum eine Pore in die wenigen Prozent
    der Flaeche, auf die es hier ankommt.
    """
    ys, xs = np.nonzero(np.abs(randabstand - ziel) <= 1.0)
    if not len(ys):
        return None
    pick = int(rng.integers(len(ys)))
    return float(xs[pick]), float(ys[pick])


def _candidate(
    spec: SpecimenSpec,
    kind: str,
    at_edge: bool,
    randabstand: np.ndarray,
    rng: np.random.Generator,
    radius_lo: float,
    radius_hi: float,
) -> _Placement | None:
    """Ein Vorschlag fuer Ort und Groesse der naechsten Pore."""
    radius = float(rng.uniform(radius_lo, radius_hi))

    if at_edge:
        # Mittig auf die Probenkante: die Pore ist angeschnitten.
        punkt = _auf_abstand(randabstand, 0.0, rng)
        return None if punkt is None else _Placement(kind, punkt[0], punkt[1], radius, True)

    if spec.edge_clearance_px is not None:
        # Gemessen wird der Abstand des Porenrandes, nicht des Mittelpunkts - sonst
        # haengt der gemessene Abstand an der Porengroesse und die Kurve ist wertlos.
        punkt = _auf_abstand(randabstand, radius + spec.edge_clearance_px, rng)
        return None if punkt is None else _Placement(kind, punkt[0], punkt[1], radius, False)

    return _Placement(
        kind,
        float(rng.uniform(0, spec.width)),
        float(rng.uniform(0, spec.height)),
        radius,
        False,
    )


def _stuecke(mask: np.ndarray) -> int:
    count, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return count - 1


def _paint(
    image: np.ndarray,
    labels: np.ndarray,
    specimen: np.ndarray,
    alpha: np.ndarray,
    normalized: np.ndarray,
    core: np.ndarray,
    fenster: tuple[slice, slice],
    placement: _Placement,
    label: int,
    spec: SpecimenSpec,
    rng: np.random.Generator,
) -> PoreTruth:
    """Traegt eine bereits gepruefte Pore ins Bild ein und liefert ihre Wahrheit."""
    region = image[fenster]
    # Gezeichnet wird ausschliesslich innerhalb der Probe: eine angeschnittene Randpore
    # darf das Einbettmittel weder aufhellen noch abdunkeln.
    inside = specimen[fenster]
    ph, pw = alpha.shape

    # Relief-Saum: der Rand wird beim Polieren abgetragen und wirft mehr Licht zurueck.
    # Er sitzt ausserhalb der Pore und zieht eine zu lockere Segmentierung nach innen.
    if spec.rim_highlight > 0:
        saum = np.clip(1.0 - (normalized - 1.0) / 0.35, 0.0, 1.0) * (normalized > 1.0)
        region *= np.where(inside, 1.0 + spec.rim_highlight * saum, 1.0)

    resin_filled = rng.random() < spec.resin_filled_frac
    if resin_filled:
        # Harz in der Pore: der Grauwertabstand zum Gefuege schrumpft auf einen Bruchteil.
        floor = float(rng.uniform(spec.resin_gray + 8, spec.resin_gray + 38))
        wall = spec.wall_lift * 0.4
    else:
        floor = float(rng.uniform(*spec.floor_gray))
        wall = spec.wall_lift

    # Kraterboden: zur Wand hin heller, dazu ein gerichteter Gradient aus der schraegen
    # Beleuchtung und eine eigene Textur. Eine Flaechenfarbe waere hier das Unrealistische.
    depth = np.clip(normalized, 0.0, 1.0)
    grey = floor + wall * depth ** 3
    lit = float(rng.uniform(-0.35, 0.35)) * wall
    _yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    grey = grey + lit * (xx - pw / 2.0) / max(pw / 2.0, 1.0)
    grey = grey + rng.normal(0.0, 3.0, size=grey.shape).astype(np.float32)

    blended = region * (1.0 - alpha) + np.clip(grey, 0, 255) * alpha
    region[inside] = blended[inside]

    labels[fenster][core] = label

    ys, xs = np.nonzero(core)
    return PoreTruth(
        label=label,
        kind=placement.kind,
        area_px=int(core.sum()),
        centroid_px=(float(xs.mean() + fenster[1].start), float(ys.mean() + fenster[0].start)),
        radius_px=placement.radius,
        floor_gray=floor,
        resin_filled=resin_filled,
        touches_specimen_edge=placement.at_edge,
    )


def _pore_shape(
    placement: _Placement, spec: SpecimenSpec, rng: np.random.Generator
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int]]:
    """Deckungsgrad und normierter Radius einer Pore.

    ``normalized`` ist der auf den Rand normierte Abstand vom Zentrum: 0 in der Mitte,
    1.0 auf der Porenwand, darueber ausserhalb. Aus ihm folgen Tiefenverlauf und
    Relief-Saum, ohne dass dafuer eine Distanztransformation noetig waere.
    """
    height, width = spec.height, spec.width
    kind = placement.kind
    radius = placement.radius
    softness = max(0.35, spec.edge_softness_px)

    if kind == "shrinkage":
        # Lunker: mehrere ineinander laufende Lappen entlang eines Zufallspfades. Die
        # Unregelmaessigkeit kommt aus der Ueberlagerung, nicht aus hohen Harmonischen -
        # die ergaeben einen sternfoermigen Rand, den es in einem Schliff nicht gibt.
        lobes = int(rng.integers(4, 8))
        harmonics = lambda: _harmonics(rng, 5, 0.06 * spec.roughness, 0.16 * spec.roughness)
        offsets = [(0.0, 0.0)]
        x, y = 0.0, 0.0
        direction = rng.uniform(0, 2 * math.pi)
        for _ in range(lobes - 1):
            direction += float(rng.normal(0, 1.1))
            step = radius * rng.uniform(0.45, 0.8)
            x, y = x + math.cos(direction) * step, y + math.sin(direction) * step
            offsets.append((x, y))
        radii = [radius * rng.uniform(0.6, 1.0) for _ in offsets]
    elif kind == "micro":
        lobes, offsets = 1, [(0.0, 0.0)]
        harmonics = lambda: _harmonics(rng, 5, 0.05 * spec.roughness, 0.13 * spec.roughness)
        radii = [radius]
    elif kind == "gas":
        lobes, offsets = 1, [(0.0, 0.0)]
        harmonics = lambda: _harmonics(rng, 5, 0.015 * spec.roughness, 0.06 * spec.roughness)
        radii = [radius]
    else:
        raise ValueError(f"Unbekannte Porenart: {kind!r}")

    reach = max(abs(ox) + r for (ox, _oy), r in zip(offsets, radii)) * 1.4 + 3 * softness + 3
    reach_y = max(abs(oy) + r for (_ox, oy), r in zip(offsets, radii)) * 1.4 + 3 * softness + 3
    half_x, half_y = int(math.ceil(reach)), int(math.ceil(reach_y))

    x0, y0 = int(round(placement.cx)) - half_x, int(round(placement.cy)) - half_y
    x1, y1 = x0 + 2 * half_x + 1, y0 + 2 * half_y + 1
    cut_x0, cut_y0 = max(0, x0), max(0, y0)
    cut_x1, cut_y1 = min(width, x1), min(height, y1)
    if cut_x1 <= cut_x0 or cut_y1 <= cut_y0:
        return None, None, (0, 0)

    yy, xx = np.mgrid[cut_y0:cut_y1, cut_x0:cut_x1].astype(np.float32)
    yy -= placement.cy
    xx -= placement.cx

    normalized = np.full(yy.shape, np.inf, dtype=np.float32)
    for (ox, oy), r in zip(offsets, radii):
        elongation = rng.uniform(0.6, 1.6) if kind == "shrinkage" else rng.uniform(0.75, 1.3)
        angle = rng.uniform(0, math.pi)
        lobe = _normalized_radius(xx - ox, yy - oy, r, harmonics(), elongation, angle)
        normalized = np.minimum(normalized, lobe)

    alpha = np.clip(0.5 + (1.0 - normalized) * radius / softness, 0.0, 1.0).astype(np.float32)
    return alpha, normalized.astype(np.float32), (cut_x0, cut_y0)


def _lobe_alpha(
    cx: float,
    cy: float,
    radius: float,
    harmonics: list[tuple[int, float, float]],
    elongation: float,
    angle: float,
    softness: float,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, tuple[int, int]]:
    """Ein einzelner unregelmaessiger Fleck - fuer Einschluesse."""
    reach = int(math.ceil(radius * 1.5 + 3 * softness + 2))
    x0, y0 = max(0, int(cx) - reach), max(0, int(cy) - reach)
    x1, y1 = min(width, int(cx) + reach + 1), min(height, int(cy) + reach + 1)
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0)

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    normalized = _normalized_radius(xx - cx, yy - cy, radius, harmonics, elongation, angle)
    alpha = np.clip(0.5 + (1.0 - normalized) * radius / softness, 0.0, 1.0).astype(np.float32)
    return alpha, (x0, y0)


def _normalized_radius(
    dx: np.ndarray,
    dy: np.ndarray,
    radius: float,
    harmonics: list[tuple[int, float, float]],
    elongation: float,
    angle: float,
) -> np.ndarray:
    """Abstand vom Zentrum, normiert auf den modulierten Rand (1.0 = Porenwand).

    Der Rand ist ``R(theta) = radius * (1 + sum a_k cos(k*theta + phi_k))`` - wenige
    Harmonische genuegen, um aus dem Kreis eine glaubwuerdige Pore zu machen, ohne dass
    der Rand ausfranst.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xr = dx * cos_a + dy * sin_a
    yr = (-dx * sin_a + dy * cos_a) / max(elongation, 1e-3)

    r = np.sqrt(xr * xr + yr * yr)
    theta = np.arctan2(yr, xr)

    modulation = np.zeros_like(theta)
    for order, amplitude, phase in harmonics:
        modulation += amplitude * np.cos(order * theta + phase)

    boundary = np.maximum(radius * (1.0 + modulation), 0.25 * radius)
    return (r / boundary).astype(np.float32)


def _harmonics(
    rng: np.random.Generator, max_order: int, lo: float, hi: float
) -> list[tuple[int, float, float]]:
    """Zufaellige Randmodulation: Ordnung, Amplitude, Phase."""
    orders = rng.choice(np.arange(2, max_order + 1), size=min(3, max_order - 1), replace=False)
    return [
        (int(order), float(rng.uniform(lo, hi)), float(rng.uniform(0, 2 * math.pi)))
        for order in orders
    ]


# --------------------------------------------------------------------------------------
# Beleuchtung
# --------------------------------------------------------------------------------------


def _illuminate(image: np.ndarray, spec: SpecimenSpec) -> np.ndarray:
    """Multiplikativer Beleuchtungsfehler - so wirkt er auch in der Realitaet."""
    kind = spec.illumination
    if kind == "none" or spec.illumination_strength <= 0:
        return image

    height, width = image.shape
    strength = spec.illumination_strength

    if kind == "linear":
        ramp = np.linspace(1.0, 1.0 - strength, width, dtype=np.float32)
        field = np.repeat(ramp[None, :], height, axis=0)
    elif kind == "vignette":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        r = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        field = 1.0 - strength * (r / math.sqrt(2.0)) ** 2
    elif kind == "hotspot":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        g = np.exp(-(((xx - width * 0.3) ** 2 + (yy - height * 0.3) ** 2) / (2 * (width * 0.25) ** 2)))
        field = 1.0 - strength + strength * g
    else:
        raise ValueError(f"Unbekannter Beleuchtungsfehler: {kind!r}")

    return image * field.astype(np.float32)


# --------------------------------------------------------------------------------------
# Auswertung: Wahrheit gegen Ergebnis
# --------------------------------------------------------------------------------------


@dataclass
class Outcome:
    """Das Ergebnis eines Falles, aufgeschluesselt nach Fehlerart."""

    truth_count: int
    detected_count: int
    matched: list[tuple[int, int, float]] = field(default_factory=list)  # (Wahrheit, Fund, IoU)
    missed: list[PoreTruth] = field(default_factory=list)
    spurious_on_distractor: int = 0
    spurious_elsewhere: int = 0
    area_errors: list[float] = field(default_factory=list)  # relativ, mit Vorzeichen
    porosity_true: float = 0.0
    porosity_measured: float = 0.0

    @property
    def recall(self) -> float:
        return len(self.matched) / self.truth_count if self.truth_count else 1.0

    @property
    def precision(self) -> float:
        return len(self.matched) / self.detected_count if self.detected_count else 1.0

    @property
    def f1(self) -> float:
        total = self.recall + self.precision
        return 2.0 * self.recall * self.precision / total if total else 0.0

    @property
    def spurious(self) -> int:
        return self.spurious_on_distractor + self.spurious_elsewhere

    @property
    def median_area_error(self) -> float:
        if not self.area_errors:
            return 0.0
        return float(np.median(np.abs(self.area_errors)))

    @property
    def porosity_error(self) -> float:
        if self.porosity_true <= 0:
            return 0.0
        return abs(self.porosity_measured - self.porosity_true) / self.porosity_true

    def summary(self) -> str:
        return (
            f"Recall {self.recall:>5.0%} ({len(self.matched)}/{self.truth_count})  "
            f"Precision {self.precision:>5.0%}  "
            f"Flaechenfehler {self.median_area_error:>5.1%}  "
            f"Porositaet {self.porosity_measured:.2f}% statt {self.porosity_true:.2f}% "
            f"({self.porosity_error:.0%})"
        )


def evaluate(
    truth: SyntheticImage,
    detected_labels: np.ndarray,
    specimen_area_px: int,
    min_iou: float = 0.3,
) -> Outcome:
    """Vergleicht ein Segmentierungsergebnis mit der Wahrheit.

    Zugeordnet wird ueber den Flaechenueberlapp (IoU), nicht ueber den Schwerpunkt: bei
    einem Lunker liegt der Schwerpunkt unter Umstaenden gar nicht in der Pore. Die
    Zuordnung ist gierig ueber den besten Ueberlapp - fuer ueberschneidungsfrei plazierte
    Poren ist das identisch mit dem Optimum.
    """
    truth_count = truth.pore_count
    detected_count = int(detected_labels.max())

    outcome = Outcome(
        truth_count=truth_count,
        detected_count=detected_count,
        porosity_true=truth.porosity_pct,
        porosity_measured=(
            100.0 * float((detected_labels > 0).sum()) / specimen_area_px
            if specimen_area_px
            else 0.0
        ),
    )
    if truth_count == 0 or detected_count == 0:
        outcome.missed = list(truth.pores)
        if detected_count:
            outcome.spurious_elsewhere = detected_count
        return outcome

    overlap = _overlap_matrix(truth.pore_labels, detected_labels, truth_count, detected_count)
    truth_area = np.bincount(
        truth.pore_labels.ravel(), minlength=truth_count + 1
    )[1:].astype(np.float64)
    detected_area = np.bincount(
        detected_labels.ravel(), minlength=detected_count + 1
    )[1:].astype(np.float64)

    union = truth_area[:, None] + detected_area[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(union), where=union > 0)

    used_truth: set[int] = set()
    used_detected: set[int] = set()
    order = np.argsort(iou, axis=None)[::-1]
    for flat in order:
        t, d = divmod(int(flat), detected_count)
        value = float(iou[t, d])
        if value < min_iou:
            break
        if t in used_truth or d in used_detected:
            continue
        used_truth.add(t)
        used_detected.add(d)
        outcome.matched.append((t + 1, d + 1, value))
        outcome.area_errors.append(
            float((detected_area[d] - truth_area[t]) / truth_area[t])
        )

    outcome.missed = [p for i, p in enumerate(truth.pores) if i not in used_truth]

    # Falschtreffer nach Ursache trennen: ein Fund auf einer Riefe oder einem Einschluss
    # sagt etwas anderes aus als einer mitten im Gefuege.
    for d in range(detected_count):
        if d in used_detected:
            continue
        region = detected_labels == (d + 1)
        on_distractor = np.count_nonzero(region & truth.distractor_mask)
        if on_distractor >= 0.25 * np.count_nonzero(region):
            outcome.spurious_on_distractor += 1
        else:
            outcome.spurious_elsewhere += 1

    return outcome


def _overlap_matrix(
    truth_labels: np.ndarray, detected_labels: np.ndarray, truth_count: int, detected_count: int
) -> np.ndarray:
    """Pixelueberlapp jeder Wahrheit mit jedem Fund, ohne Schleife ueber die Paare."""
    flat = truth_labels.astype(np.int64).ravel() * (detected_count + 1) + detected_labels.ravel()
    counts = np.bincount(flat, minlength=(truth_count + 1) * (detected_count + 1))
    matrix = counts.reshape(truth_count + 1, detected_count + 1)
    return matrix[1:, 1:].astype(np.float64)


# --------------------------------------------------------------------------------------
# Durchlauf durch die Pipeline
# --------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------
# Systematischer Parameterraum
# --------------------------------------------------------------------------------------


BASIS = SpecimenSpec()


def iter_specs() -> list[tuple[str, SpecimenSpec]]:
    """Faehrt jede Dimension einzeln ab, ausgehend von einem Basisfall.

    Bewusst kein volles Kreuzprodukt: eine Dimension nach der anderen zu variieren zeigt
    unmittelbar, WELCHE Eigenschaft die Detektion bricht. Beim Kreuzprodukt waere die
    Ursache aus der Trefferquote allein nicht mehr ablesbar.
    """
    specs: list[tuple[str, SpecimenSpec]] = [("basis", BASIS)]

    for kind in ("poliert", "geaetzt", "dendritisch"):
        specs.append(("gefuege", replace(BASIS, microstructure=kind)))

    # Eigene Dimension, weil hier die Defaults messbar scheitern und der Mittelwert ueber
    # alle Gefuege den Befund sonst verdecken wuerde.
    specs.append(("mehrphasig", replace(BASIS, microstructure="mehrphasig")))

    specs.append(("porenart", replace(BASIS, pore_kinds=("gas",))))
    specs.append(("porenart", replace(BASIS, pore_kinds=("shrinkage",), pore_count=10)))
    specs.append(("porenart", replace(BASIS, pore_kinds=("micro",), radius_px=(3.0, 7.0))))
    specs.append(("porenart", replace(BASIS, pore_kinds=("gas", "shrinkage", "micro"))))

    for radius in ((6.0, 12.0), (12.0, 24.0)):
        specs.append(("porengroesse", replace(BASIS, radius_px=radius, pore_count=12)))

    # Mikroporen erzeugen eine Porositaet weit unter einem Prozent - der Fall, in dem die
    # unterste von drei Otsu-Klassen nicht mehr die Poren, sondern die Korngrenzen trifft.
    specs.append(("mikroporen", replace(BASIS, radius_px=(3.0, 6.0), pore_count=12)))

    # Poren groesser als der halbe Hintergrund-Kernel der Beleuchtungskorrektur. Die
    # Bedingung steht in der Doku; hier wird sie gemessen.
    specs.append(("grosse_poren", replace(BASIS, radius_px=(20.0, 40.0), pore_count=12)))

    for floor in ((5.0, 15.0), (20.0, 40.0), (45.0, 70.0), (70.0, 95.0)):
        specs.append(("porentiefe", replace(BASIS, floor_gray=floor)))

    for softness in (0.5, 1.2, 2.5, 4.0):
        specs.append(("kantenschaerfe", replace(BASIS, edge_softness_px=softness)))

    for rim in (0.0, 0.10, 0.20, 0.35):
        specs.append(("reliefsaum", replace(BASIS, rim_highlight=rim)))

    for fraction in (0.0, 0.3, 0.6):
        specs.append(("harzfuellung", replace(BASIS, resin_filled_frac=fraction)))

    for count in (10, 30, 60):
        specs.append(("riefen", replace(BASIS, scratches=count)))

    for count in (10, 30, 60):
        specs.append(("einschluesse", replace(BASIS, inclusions=count)))

    for kind, strength in (("linear", 0.35), ("vignette", 0.45), ("hotspot", 0.40)):
        specs.append((
            "beleuchtung",
            replace(BASIS, illumination=kind, illumination_strength=strength),
        ))

    for sigma in (0.0, 3.0, 7.0, 12.0):
        specs.append(("rauschen", replace(BASIS, noise_sigma=sigma)))

    for quality in (95, 75, 50):
        specs.append(("jpeg", replace(BASIS, jpeg_quality=quality)))

    specs.append(("einbettung", replace(BASIS, embedding=True)))
    specs.append(("einbettung", replace(BASIS, embedding=False)))

    # Bewusst ausserhalb der Zusicherung: eine am Probenrand angeschnittene oder randnahe
    # Pore ist nicht ringsum von Metall umschlossen und faellt damit per Konstruktion aus
    # der Maske. Der Benchmark weist diese Faelle getrennt aus und misst den Uebergang.
    specs.append(("randporen", replace(BASIS, pore_count=8, edge_pores=4)))
    for abstand in (0.0, 5.0, 10.0, 20.0):
        specs.append((
            "randnah", replace(BASIS, pore_count=8, edge_clearance_px=abstand)
        ))

    return specs


def iter_cross_section(limit: int = 60, seed: int = 0) -> list[SpecimenSpec]:
    """Zufaellige Kombinationen aus allen Dimensionen - der harte Durchlauf."""
    rng = np.random.default_rng(seed)
    gefuege = ("poliert", "geaetzt", "dendritisch", "mehrphasig")
    arten = (("gas",), ("shrinkage",), ("micro",), ("gas", "shrinkage"), ("gas", "micro"))
    beleuchtung = ("none", "linear", "vignette", "hotspot")

    specs = []
    for _ in range(limit):
        lo = float(rng.uniform(4.0, 14.0))
        illumination = beleuchtung[rng.integers(len(beleuchtung))]
        specs.append(
            SpecimenSpec(
                microstructure=gefuege[rng.integers(len(gefuege))],
                embedding=bool(rng.integers(2)),
                pore_count=int(rng.integers(8, 25)),
                pore_kinds=arten[rng.integers(len(arten))],
                radius_px=(lo, lo + float(rng.uniform(4.0, 20.0))),
                floor_gray=(float(rng.uniform(5, 40)), float(rng.uniform(45, 75))),
                rim_highlight=float(rng.uniform(0.0, 0.25)),
                edge_softness_px=float(rng.uniform(0.5, 3.0)),
                resin_filled_frac=float(rng.choice([0.0, 0.0, 0.2, 0.4])),
                scratches=int(rng.choice([0, 0, 15, 40])),
                inclusions=int(rng.choice([0, 0, 15, 40])),
                illumination=illumination,
                illumination_strength=0.0 if illumination == "none" else float(rng.uniform(0.2, 0.5)),
                noise_sigma=float(rng.uniform(0.0, 9.0)),
                jpeg_quality=int(rng.choice([0, 0, 90, 60])) or None,
            )
        )
    return specs
