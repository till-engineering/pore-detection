"""Prägt echten Schliffbildern Poren mit exakt bekannter Wahrheit auf.

Das ist die schärfste Form eines Prüfbildes: das Gefüge ist **echt**, mit allen
Eigenheiten, die kein Generator trifft - und trotzdem ist jede Pore nach Lage, Fläche und
Äquivalentdurchmesser genau bekannt. An einem rein synthetischen Bild ließe sich beides
nicht zugleich haben.

Gezeichnet wird nur dorthin, wo eine Pore hingehört und auch zu sehen wäre. Drei
Bedingungen müssen zusammenkommen:

* **in der Probe** - die Maske kommt aus :mod:`poredet.specimen` (Standard: grabcut).
  In V1 musste dieser Schritt umgangen werden, weil die dortige Probenmaske bei Bildern
  ohne Einbettmittel auf einen Bruchteil zusammenschrumpfte; in V2 trägt sie.
* **nicht im Overlay** - der eingebrannte Maßstabskasten wird über
  :func:`poredet.scale.overlay_mask` ausgespart. Eine Pore im Maßstabsbalken wäre Unsinn.
* **auf hellem Grund** - eine Pore auf dunklem Untergrund wäre unsichtbar, und eine
  Wahrheit, die im Bild nicht steht, ist keine. Geprüft wird gegen den geschätzten
  Hintergrund, nicht gegen den Pixelwert.

Aufruf::

    .venv\\Scripts\\python.exe scripts/stamp_pores.py
    .venv\\Scripts\\python.exe scripts/stamp_pores.py --pores 25 --seed 7 --dry-run

Die Bilder werden **an Ort und Stelle** ersetzt; die Originale liegen unter
``test/Safe_without_Pores``. Die Wahrheit landet daneben in ``data/runs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

WURZEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WURZEL / "src"))

from poredet.dev.pore_stamp import SpecimenSpec, add_pores  # noqa: E402
from poredet.io.image_reader import find_images, read_image  # noqa: E402
from poredet.scale import ScaleResolver, overlay_mask  # noqa: E402
from poredet.specimen import segment  # noqa: E402

QUELLE = WURZEL / "test" / "Pore_detection_test"
SICHERUNG = WURZEL / "test" / "Safe_without_Pores"
AUSGABE = WURZEL / "data" / "runs" / "aufgepraegte_poren"


# --------------------------------------------------------------------------------------
# Wo darf eine Pore hin?
# --------------------------------------------------------------------------------------


def sichtbarer_grund(gray: np.ndarray, hellster_porenboden: float = 38.0) -> np.ndarray:
    """Ist der Untergrund hell genug, dass sich eine Pore davon abhebt?

    Geschätzt wie in der Beleuchtungskorrektur - morphologische Schließung mit einem
    Kernel, der größer als jede Pore ist, plus Glättung. Feine dunkle Gefügeanteile füllt
    die Schließung auf; eine großflächige dunkle Stelle bleibt dunkel und fällt heraus.

    Die 60 Graustufen Abstand sind der Punkt: weniger, und die Pore ginge im Untergrund
    unter, noch bevor irgendeine Schwelle über sie entscheiden könnte. Eine Wahrheit, die
    man im Bild nicht sieht, ist als Prüfmaßstab wertlos.
    """
    kurz = min(gray.shape[:2])
    kernel_px = max(31, kurz // 8) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))

    hintergrund = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE
    )
    hintergrund = cv2.GaussianBlur(
        hintergrund.astype(np.float32), (0, 0), kernel_px / 3.0,
        borderType=cv2.BORDER_REPLICATE,
    )
    return hintergrund > hellster_porenboden + 60.0


def platzierung(image) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Probenmaske, Overlay-Aussparung und eine kurze Herkunftsangabe."""
    maske = segment(image.gray, color=image.color)

    outcome = ScaleResolver().resolve(image.gray, image.path)
    aussparung = None
    if outcome.scale is not None:
        aussparung = overlay_mask(image.gray.shape, outcome.scale, pad=6)

    erlaubt = maske.specimen & ndi.binary_fill_holes(sichtbarer_grund(image.gray))
    if aussparung is not None:
        erlaubt &= ~aussparung

    herkunft = (
        f"Probe {maske.specimen_frac:.0%}"
        + (f", Harz {maske.resin_frac:.0%}" if maske.has_resin else "")
        + (", Overlay ausgespart" if aussparung is not None and aussparung.any() else "")
    )
    return erlaubt, aussparung, herkunft


def porenspec(gray: np.ndarray, anzahl: int) -> SpecimenSpec:
    """Porengrößen relativ zur Bildgröße.

    Die Vorlagen reichen von 368 px Breite bis 3088 px. Eine feste Größe in Mikrometern
    wäre auf der einen Hälfte unsichtbar und auf der anderen bildfüllend, deshalb wird
    relativ skaliert; die Größe in Mikrometern steht anschließend in der Wahrheitsdatei.
    """
    kurz = min(gray.shape[:2])
    return SpecimenSpec(
        pore_count=anzahl,
        pore_kinds=("gas", "gas", "shrinkage", "gas", "micro"),
        radius_px=(max(3.0, 0.012 * kurz), max(6.0, 0.045 * kurz)),
        floor_gray=(8.0, 38.0),
        wall_lift=30.0,
        rim_highlight=0.12,
        edge_softness_px=1.4,
        resin_filled_frac=0.15,
        edge_margin_px=14.0,
        blur_sigma=0.7,
        noise_sigma=0.0,   # das echte Bild bringt sein Rauschen mit
        illumination="none",
    )


# --------------------------------------------------------------------------------------


def einfaerben(
    original_color: np.ndarray, original_gray: np.ndarray, gestempelt_gray: np.ndarray
) -> np.ndarray:
    """Die Aufprägung auf ein Farbbild übertragen, ohne es zu entfärben.

    Das Zeichnen arbeitet auf dem Graubild. Das Ergebnis einfach zurückzuschreiben würde
    aus einem Farbbild ein Graustufenbild machen - und ausgerechnet ``grabcut``, das
    Standardverfahren der Probenmaske, schätzt seine Modelle über die Farbkanäle. Das
    Prüfbild würde die Kette also genau an der Stelle verändern, an der sie geprüft
    werden soll.

    Übertragen wird deshalb nur die **Differenz**: außerhalb der Poren ist sie null und
    das Bild bleibt Pixel für Pixel unangetastet, innerhalb zieht sie alle drei Kanäle
    gleichmäßig ins Dunkle. Das trifft auch die Sache - ein Hohlraum wirft kein Licht
    zurück und ist deshalb nahezu farblos.
    """
    delta = gestempelt_gray.astype(np.int16) - original_gray.astype(np.int16)
    return np.clip(
        original_color.astype(np.int16) + delta[:, :, None], 0, 255
    ).astype(np.uint8)


def schreibe(pfad: Path, bild: np.ndarray) -> None:
    """Bild unter demselben Namen zurückschreiben, Umlaute im Pfad inbegriffen."""
    suffix = pfad.suffix.lower()
    parameter = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if suffix in {".jpg", ".jpeg"} else []
    erfolg, puffer = cv2.imencode(suffix, bild, parameter)
    if not erfolg:
        raise OSError(f"{pfad.name} ließ sich nicht kodieren")
    puffer.tofile(str(pfad))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pores", type=int, default=18, help="Poren je Bild")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input", type=Path, default=QUELLE)
    parser.add_argument("--output", type=Path, default=AUSGABE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur rechnen und berichten, nichts schreiben")
    args = parser.parse_args(argv)

    pfade = find_images(args.input)
    if not pfade:
        print(f"Keine Bilder unter {args.input}", file=sys.stderr)
        return 2

    if not args.dry_run and not SICHERUNG.is_dir():
        print(f"Abbruch: die Sicherung {SICHERUNG} fehlt. Die Bilder werden an Ort und "
              f"Stelle ersetzt - ohne Sicherung wäre das nicht rückholbar.", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "masken").mkdir(exist_ok=True)

    print(f"{len(pfade)} Bilder in {args.input}")
    print(f"{'Bild':<40} {'Poren':>5} {'Porositaet':>11} {'Durchmesser':>18}  Platzierung")

    wahrheit = []
    for index, pfad in enumerate(pfade):
        image = read_image(pfad)
        erlaubt, aussparung, herkunft = platzierung(image)

        spec = porenspec(image.gray, args.pores)
        gestempelt = add_pores(image.gray, spec, seed=args.seed + index,
                               specimen_mask=erlaubt, exclusion=aussparung)

        outcome = ScaleResolver().resolve(image.gray, pfad)
        um_per_px = outcome.scale.um_per_px if outcome.scale else None

        durchmesser = [p.equivalent_diameter_px for p in gestempelt.pores]
        if durchmesser and um_per_px:
            spanne = (f"{min(durchmesser) * um_per_px:7.1f}-"
                      f"{max(durchmesser) * um_per_px:<7.1f} um")
        elif durchmesser:
            spanne = f"{min(durchmesser):7.1f}-{max(durchmesser):<7.1f} px"
        else:
            spanne = "-"

        print(f"{pfad.name[:40]:<40} {gestempelt.pore_count:>5} "
              f"{gestempelt.porosity_pct:>10.2f} % {spanne:>18}  {herkunft}")

        wahrheit.append({
            "image": pfad.name,
            "width": image.width,
            "height": image.height,
            "seed": args.seed + index,
            "um_per_px": um_per_px,
            "specimen_area_px": gestempelt.specimen_area_px,
            "pore_count": gestempelt.pore_count,
            "pore_area_px": gestempelt.pore_area_px,
            "porosity_pct": gestempelt.porosity_pct,
            "pores": [
                {
                    "label": p.label,
                    "kind": p.kind,
                    "area_px": p.area_px,
                    "centroid_px": list(p.centroid_px),
                    "equivalent_diameter_px": p.equivalent_diameter_px,
                    "equivalent_diameter_um": (
                        p.equivalent_diameter_px * um_per_px if um_per_px else None
                    ),
                    "floor_gray": p.floor_gray,
                    "resin_filled": p.resin_filled,
                    "touches_specimen_edge": p.touches_specimen_edge,
                }
                for p in gestempelt.pores
            ],
        })

        if not args.dry_run:
            fertig = gestempelt.gray if image.color is None else einfaerben(
                image.color, image.gray, gestempelt.gray
            )
            schreibe(pfad, fertig)
            # Das Labelbild ist die eigentliche Wahrheit - ohne es liesse sich eine
            # Detektion nur zaehlen, nicht Pore fuer Pore zuordnen.
            cv2.imencode(".png", gestempelt.pore_labels.astype(np.uint16))[1].tofile(
                str(args.output / "masken" / f"{pfad.stem}_poren.png")
            )

    gesamt = sum(e["pore_count"] for e in wahrheit)
    flaeche = sum(e["pore_area_px"] for e in wahrheit)
    bezug = sum(e["specimen_area_px"] for e in wahrheit)
    print(f"\n{gesamt} Poren in {len(wahrheit)} Bildern, "
          f"Gesamtporositaet {100.0 * flaeche / bezug:.2f} %")

    if args.dry_run:
        print("--dry-run: nichts geschrieben")
        return 0

    ziel = args.output / "wahrheit.json"
    ziel.write_text(json.dumps(wahrheit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Bilder ersetzt in   {args.input}")
    print(f"Wahrheit  {ziel}")
    print(f"Labelbilder in      {args.output / 'masken'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
