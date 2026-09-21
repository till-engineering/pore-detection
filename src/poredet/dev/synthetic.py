"""Künstliche Schliffbilder mit bekannter Wahrheit (Porenzahl, Fläche, Maßstab).

Die einzige Möglichkeit, absolute Korrektheit zu prüfen statt nur Konstanz. Bei echten
Bildern lässt sich bestenfalls feststellen, dass sich ein Ergebnis nicht verändert hat -
ob es richtig ist, weiß niemand. Hier ist die Antwort vorher bekannt.

Bislang enthalten: ein Maßstabs-Overlay nach ImageJ-Bauart auf texturiertem Untergrund.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.models import BBox


@dataclass(frozen=True)
class SyntheticSpecimen:
    """Ein erzeugter Schliff mit Einbettmittel, samt der Wahrheit dazu."""

    gray: np.ndarray
    resin: np.ndarray            # bool, True = Einbettmittel
    pores: np.ndarray            # bool, True = Pore (liegt INNERHALB der Probe)

    @property
    def specimen(self) -> np.ndarray:
        return ~self.resin

    @property
    def resin_frac(self) -> float:
        return float(self.resin.mean())


def make_specimen_image(
    *,
    width: int = 600,
    height: int = 450,
    resin_sides: tuple[str, ...] = ("left",),
    resin_frac: float = 0.2,
    specimen_gray: int = 150,
    specimen_texture: float = 18.0,
    resin_gray: int = 32,
    resin_texture: float = 2.0,
    pores: tuple[tuple[int, int, int], ...] = (),
    bubbles: tuple[tuple[int, int, int], ...] = (),
    seed: int = 11,
) -> SyntheticSpecimen:
    """Ein Schliffbild mit Einbettmittel erzeugen.

    Die Wahrheit steckt in den Parametern: ``resin_sides`` und ``resin_frac`` legen den
    Saum fest, ``pores`` setzt dunkle Hohlräume **innerhalb** der Probe. Poren sind der
    eigentliche Prüfstein - sie sind genauso dunkel und glatt wie das Harz, und ein
    Verfahren, das sie mitentfernt, wäre unbrauchbar.

    ``bubbles`` setzt Lufteinschlüsse ins Harz: helle, runde Flecken, die pixelweise
    nicht nach Harz aussehen, aber dazugehören.
    """
    rng = np.random.default_rng(seed)

    # Probe: hell und strukturiert.
    gray = np.clip(
        rng.normal(specimen_gray, specimen_texture, size=(height, width)), 0, 255
    ).astype(np.uint8)
    for _ in range(60):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(4, 22))
        shade = int(np.clip(specimen_gray + rng.normal(0, 45), 0, 255))
        cv2.circle(gray, center, radius, shade, -1)

    # Einbettmittel: dunkel und glatt, als Saum an den gewählten Seiten.
    resin = np.zeros((height, width), dtype=bool)
    for side in resin_sides:
        if side == "left":
            resin[:, : int(resin_frac * width)] = True
        elif side == "right":
            resin[:, width - int(resin_frac * width):] = True
        elif side == "top":
            resin[: int(resin_frac * height), :] = True
        elif side == "bottom":
            resin[height - int(resin_frac * height):, :] = True
        else:
            raise ValueError(f"Unbekannte Seite: {side!r}")

    resin_values = np.clip(
        rng.normal(resin_gray, resin_texture, size=(height, width)), 0, 255
    ).astype(np.uint8)
    gray[resin] = resin_values[resin]

    # Luftblasen im Harz: hell und rund, gehören trotzdem zum Harz.
    for x, y, radius in bubbles:
        cv2.circle(gray, (x, y), radius, int(min(255, resin_gray + 70)), -1)
        cv2.circle(gray, (x, y), radius, int(max(0, resin_gray - 25)), 2)

    # Poren: dunkel und glatt wie Harz, aber von Probe umschlossen.
    pore_mask = np.zeros((height, width), dtype=bool)
    for x, y, radius in pores:
        cv2.circle(gray, (x, y), radius, resin_gray, -1)
        cv2.circle(pore_mask, (x, y), radius, 1, -1)  # type: ignore[arg-type]
    pore_mask = pore_mask.astype(bool) & ~resin

    return SyntheticSpecimen(gray=gray, resin=resin, pores=pore_mask)


@dataclass(frozen=True)
class SyntheticOverlay:
    """Ein erzeugtes Bild samt der Wahrheit dazu."""

    gray: np.ndarray
    bar: BBox
    box: BBox
    label_text: str
    value_um: float

    @property
    def um_per_px(self) -> float:
        return self.value_um / self.bar.w


def make_overlay_image(
    *,
    width: int = 640,
    height: int = 480,
    bar_length_px: int = 120,
    bar_height_px: int = 4,
    label_text: str = "200 um",
    value_um: float = 200.0,
    position: str = "bottom_right",
    background: str = "texture",
    label_below: bool = True,
    seed: int = 7,
) -> SyntheticOverlay:
    """Ein Bild mit eingebranntem Maßstabs-Overlay erzeugen.

    ``background`` steuert den Schwierigkeitsgrad: ``texture`` ist mittelgraues Rauschen,
    ``bright`` ein sehr helles Gefüge - der Fall, an dem eine Kastensuche ohne Balkenanker
    scheitert -, ``dark`` ein dunkles.
    """
    rng = np.random.default_rng(seed)
    base = {"texture": 140, "bright": 235, "dark": 60}.get(background, 140)
    gray = np.clip(rng.normal(base, 14, size=(height, width)), 0, 254).astype(np.uint8)
    # Etwas Struktur, damit das Bild nicht nur aus Rauschen besteht.
    for _ in range(40):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        radius = int(rng.integers(3, 18))
        shade = int(np.clip(base + rng.normal(0, 50), 0, 254))
        cv2.circle(gray, (cx, cy), radius, shade, -1)

    # Kasten: Beschriftung und Balken übereinander, mit Rand ringsum.
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (text_w, text_h), _baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    margin = 6
    gap = 5
    box_w = max(bar_length_px, text_w) + 2 * margin
    box_h = bar_height_px + gap + text_h + 2 * margin

    positions = {
        "bottom_right": (width - box_w - 8, height - box_h - 8),
        "top_left": (8, 8),
        "center": ((width - box_w) // 2, (height - box_h) // 2),
    }
    box_x, box_y = positions.get(position, positions["bottom_right"])
    box = BBox(box_x, box_y, box_w, box_h)

    # Der Kasten ist reines Weiß - genau das macht ihn auffindbar.
    gray[box.slices()] = 255

    bar_x = box_x + (box_w - bar_length_px) // 2
    text_x = box_x + (box_w - text_w) // 2
    if label_below:
        bar_y = box_y + margin
        text_baseline = bar_y + bar_height_px + gap + text_h
    else:
        text_baseline = box_y + margin + text_h
        bar_y = text_baseline + gap

    bar = BBox(bar_x, bar_y, bar_length_px, bar_height_px)
    gray[bar.slices()] = 0
    cv2.putText(gray, label_text, (text_x, text_baseline), font, font_scale, 0,
                font_thickness, cv2.LINE_AA)

    return SyntheticOverlay(
        gray=gray, bar=bar, box=box, label_text=label_text, value_um=value_um
    )
