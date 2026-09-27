"""Einstiegspunkt: run, preview, scale-check, report, benchmark.

Bis zur GUI ist die Kommandozeile das Arbeitswerkzeug. Sie hat aber auch danach einen
Zweck: sie zwingt den Core dazu, ohne Oberfläche vollständig zu funktionieren.

Umgesetzt: ``run`` (die vollstaendige Pipeline), ``scale``, ``specimen``
und ``specimen-compare``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from ..config.loader import load_config
from ..core.models import BatchResult
from ..io.image_reader import ImageReadError, find_images, read_image
from ..io.metadata import read_imagej_calibration
from ..scale.resolver import ScaleResolver
from ..scale.visualize import SheetEntry, contact_sheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poredet",
        description="Detektion und Vermessung von Poren in Schliffbildern",
    )
    parser.add_argument("--version", action="version", version=f"poredet {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Ausführliches Log")
    sub = parser.add_subparsers(dest="command", required=True)

    scale = sub.add_parser(
        "scale",
        help="Maßstab aus dem eingebrannten Balken lesen",
        description="Liest den Maßstab aus dem eingebrannten Balken und schreibt auf "
                    "Wunsch ein Übersichtsblatt zur Sichtprüfung.",
    )
    scale.add_argument("input", type=Path, help="Bilddatei oder Ordner")
    scale.add_argument("-r", "--recursive", action="store_true", help="Unterordner mitnehmen")
    scale.add_argument("-o", "--sheet", type=Path, default=None,
                       help="Übersichtsblatt als PNG schreiben")
    scale.add_argument("--json", type=Path, default=None, help="Ergebnisse als JSON ablegen")
    scale.add_argument("--no-reference", action="store_true",
                       help="Nicht gegen die ImageJ-Kalibrierung im TIFF prüfen")
    scale.set_defaults(func=cmd_scale)

    specimen = sub.add_parser(
        "specimen",
        help="Einbettmittel entfernen und die Probe freistellen",
        description="Trennt die Probe vom Einbettmittel und schreibt auf Wunsch ein "
                    "Übersichtsblatt zur Sichtprüfung. Ein Maßstab wird dafür nicht "
                    "gebraucht.",
    )
    specimen.add_argument("input", type=Path, help="Bilddatei oder Ordner")
    specimen.add_argument("-r", "--recursive", action="store_true",
                          help="Unterordner mitnehmen")
    specimen.add_argument("-o", "--sheet", type=Path, default=None,
                          help="Übersichtsblatt als PNG schreiben")
    specimen.add_argument("--json", type=Path, default=None,
                          help="Ergebnisse als JSON ablegen")
    specimen.add_argument("--masks", type=Path, default=None,
                          help="Probenmasken als PNG in diesen Ordner schreiben")
    specimen.add_argument("--method", default=None,
                          help="Verfahren; ohne Angabe das in der Konfiguration "
                               "hinterlegte (derzeit grabcut)")
    specimen.set_defaults(func=cmd_specimen)

    compare = sub.add_parser(
        "specimen-compare",
        help="Alle Trennverfahren nebeneinander auf denselben Bildern",
        description="Lässt mehrere Segmentierer auf denselben Signalen laufen und stellt "
                    "die Ergebnisse nebeneinander. Unterschiede stammen damit aus dem "
                    "Verfahren, nicht aus unterschiedlicher Vorverarbeitung.",
    )
    compare.add_argument("input", type=Path, help="Bilddatei oder Ordner")
    compare.add_argument("-r", "--recursive", action="store_true")
    compare.add_argument("-o", "--sheet", type=Path, default=None,
                         help="Vergleichsblatt als PNG schreiben")
    compare.add_argument("--json", type=Path, default=None)
    compare.add_argument("--methods", default=None,
                         help="Komma-Liste; ohne Angabe alle Vergleichsverfahren")
    compare.set_defaults(func=cmd_specimen_compare)

    run = sub.add_parser(
        "run",
        help="Die vollstaendige Pipeline: Massstab, Einbettmittel, Poren",
        description="Massstab erkennen, dessen Overlay ausschliessen, Einbettmittel "
                    "entfernen, dann die Poren suchen und vermessen. Die Reihenfolge ist "
                    "fest - siehe poredet.core.pipeline.",
    )
    run.add_argument("input", type=Path, help="Bilddatei oder Ordner")
    run.add_argument("-r", "--recursive", action="store_true")
    run.add_argument("-o", "--sheet", type=Path, default=None,
                     help="Kontrollblatt als PNG schreiben")
    run.add_argument("--csv", type=Path, default=None,
                     help="Ordner fuer poren.csv und bilder.csv")
    run.add_argument("--json", type=Path, default=None, help="Ergebnisbaum als JSON")
    run.add_argument("--pore-method", default=None,
                     help="local_contrast (Standard) | threshold | adaptive")
    run.add_argument("--specimen-method", default=None,
                     help="grabcut (Standard) | lasso | random_walker | watershed | ...")
    run.set_defaults(func=cmd_run)

    config = sub.add_parser(
        "config",
        help="Globale Einstellungen: Pfad und Stand anzeigen, auf Standard zuruecksetzen",
        description="Die Einstellungen stehen in config/einstellungen.yaml. Ohne Option "
                    "wird angezeigt, welche Datei gilt und ob sie vom Standard abweicht.",
    )
    config.add_argument("--reset", action="store_true",
                        help="einstellungen.yaml durch die Vorlage default.yaml ersetzen")
    config.set_defaults(func=cmd_config)

    return parser


# --------------------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------------------


def cmd_scale(args: argparse.Namespace) -> int:
    try:
        paths = find_images(args.input, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print(f"Keine Bilder unter {args.input}", file=sys.stderr)
        return 2

    resolver = ScaleResolver(load_config().scale)
    entries: list[SheetEntry] = []
    records: list[dict] = []

    width = min(46, max(len(p.name) for p in paths))
    header = f"{'Bild':{width}} {'µm/px':>14} {'Balken':>8} {'Beschriftung':>16} {'Konf':>5}"
    print(header)
    print("-" * len(header))

    found = 0
    deviations: list[float] = []

    for path in paths:
        try:
            image = read_image(path)
        except ImageReadError as exc:
            print(f"{path.name[:width]:{width}} -- {exc}")
            continue

        outcome = resolver.resolve(image.gray, path)
        reference = None
        if not args.no_reference:
            calibration = read_imagej_calibration(path)
            reference = calibration.um_per_px if calibration else None

        entries.append(SheetEntry(name=path.name, image=image.as_bgr(),
                                  outcome=outcome, reference_um_per_px=reference))

        scale = outcome.scale
        if scale is None:
            reason = outcome.rejections[0] if outcome.rejections else "kein Overlay gefunden"
            print(f"{path.name[:width]:{width}} {'-':>14} {'-':>8} {'-':>16} {'-':>5}  {reason}")
            records.append({"image": path.name, "found": False, "reason": reason})
            continue

        found += 1
        suffix = ""
        if reference:
            deviation = 100.0 * (scale.um_per_px - reference) / reference
            deviations.append(abs(deviation))
            suffix = f"  Referenz {reference:.6g} ({deviation:+.3f} %)"
        print(
            f"{path.name[:width]:{width}} {scale.um_per_px:>14.6g} "
            f"{scale.bar_length_px:>8.0f} {(scale.label_text or '')[:16]:>16} "
            f"{scale.confidence:>5.0%}{suffix}"
        )
        for warning in scale.warnings:
            print(f"{'':{width}}   ! {warning}")

        records.append({
            "image": path.name,
            "found": True,
            "um_per_px": scale.um_per_px,
            "label_text": scale.label_text,
            "value_um": scale.value_um,
            "unit": scale.unit_text,
            "bar_length_px": scale.bar_length_px,
            "confidence": scale.confidence,
            "engine": scale.engine,
            "source": str(scale.source),
            "box": None if scale.box is None else
                   [scale.box.x, scale.box.y, scale.box.w, scale.box.h],
            "warnings": list(scale.warnings),
            "reference_um_per_px": reference,
        })

    print("-" * len(header))
    print(f"{found} von {len(entries)} Bildern mit erkanntem Maßstab")
    if deviations:
        print(f"Abweichung zur ImageJ-Referenz: max {max(deviations):.4f} %, "
              f"Mittel {sum(deviations) / len(deviations):.4f} %")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON geschrieben: {args.json}")

    if args.sheet:
        out = contact_sheet(entries, args.sheet)
        print(f"Übersicht geschrieben: {out}")

    return 0 if found == len(entries) else 1


# --------------------------------------------------------------------------------------
# specimen
# --------------------------------------------------------------------------------------


def cmd_specimen(args: argparse.Namespace) -> int:
    import cv2

    from ..specimen import get_segmenter
    from ..specimen.visualize import SheetEntry as SpecimenEntry
    from ..specimen.visualize import contact_sheet as specimen_sheet

    try:
        paths = find_images(args.input, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print(f"Keine Bilder unter {args.input}", file=sys.stderr)
        return 2

    cfg = load_config().specimen
    if args.method:
        cfg = cfg.model_copy(update={"method": args.method})
    try:
        segmenter = get_segmenter(cfg.method)
    except KeyError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    entries: list[SpecimenEntry] = []
    records: list[dict] = []

    width = min(40, max(len(p.name) for p in paths))
    header = f"{'Bild':{width}} {'Größe':>12} {'Harz':>7} {'Probe':>7} {'Stücke':>7}"
    print(header)
    print("-" * len(header))

    with_resin = 0
    for path in paths:
        try:
            image = read_image(path)
        except ImageReadError as exc:
            print(f"{path.name[:width]:{width}} -- {exc}")
            continue

        # Farbe durchreichen, wo das Verfahren sie nutzt - GrabCut schaetzt seine
        # Modelle ueber die Farbkanaele.
        if image.color is not None and getattr(segmenter, "uses_color", False):
            mask = segmenter.segment_color(image.gray, image.color, cfg)
        else:
            mask = segmenter.segment(image.gray, cfg)
        entries.append(SpecimenEntry(name=path.name, image=image.as_bgr(), mask=mask))
        with_resin += int(mask.has_resin)

        size = f"{image.width}x{image.height}"
        print(
            f"{path.name[:width]:{width}} {size:>12} {mask.resin_frac:>7.1%} "
            f"{mask.specimen_frac:>7.1%} {mask.components:>7}"
        )
        for warning in mask.warnings:
            print(f"{'':{width}}   ! {warning}")

        records.append({
            "image": path.name,
            "width": image.width,
            "height": image.height,
            "method": mask.method,
            "resin_fraction": mask.resin_frac,
            "specimen_fraction": mask.specimen_frac,
            "specimen_area_px": mask.specimen_area_px,
            "resin_area_px": mask.resin_area_px,
            "components": mask.components,
            "warnings": list(mask.warnings),
        })

        if args.masks:
            args.masks.mkdir(parents=True, exist_ok=True)
            target = args.masks / f"{path.stem}_specimen.png"
            cv2.imwrite(str(target), mask.specimen.astype(np.uint8) * 255)

    print("-" * len(header))
    print(f"{with_resin} von {len(entries)} Bildern mit erkanntem Einbettmittel")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"JSON geschrieben: {args.json}")

    if args.sheet:
        print(f"Übersicht geschrieben: {specimen_sheet(entries, args.sheet)}")

    return 0


# --------------------------------------------------------------------------------------
# specimen-compare
# --------------------------------------------------------------------------------------


def cmd_specimen_compare(args: argparse.Namespace) -> int:
    import time

    from ..specimen import COMPARISON_METHODS, get_segmenter
    from ..specimen import signals as sig
    from ..specimen.visualize import ComparisonRow, comparison_sheet

    try:
        paths = find_images(args.input, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print(f"Keine Bilder unter {args.input}", file=sys.stderr)
        return 2

    methods = (
        [m.strip() for m in args.methods.split(",") if m.strip()]
        if args.methods else list(COMPARISON_METHODS)
    )
    try:
        segmenters = {name: get_segmenter(name) for name in methods}
    except KeyError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    cfg = load_config().specimen
    rows: list[ComparisonRow] = []
    records: list[dict] = []
    seconds = dict.fromkeys(methods, 0.0)
    flagged = dict.fromkeys(methods, 0)

    width = min(30, max(len(p.name) for p in paths))
    header = f"{'Bild':{width}} " + " ".join(f"{m[:13]:>13}" for m in methods)
    print(header)
    print("-" * len(header))

    for path in paths:
        try:
            image = read_image(path)
        except ImageReadError as exc:
            print(f"{path.name[:width]:{width}} -- {exc}")
            continue

        # Die Signale werden EINMAL berechnet und geteilt. Nur so misst der Vergleich die
        # Verfahren und nicht deren jeweilige Vorverarbeitung.
        signals = sig.compute(image.gray, cfg)
        results = {}
        cells = []
        for name, segmenter in segmenters.items():
            started = time.perf_counter()
            if getattr(segmenter, "uses_color", False):
                mask = segmenter.from_signals(signals, cfg, color=image.color)
            else:
                mask = segmenter.from_signals(signals, cfg)
            seconds[name] += time.perf_counter() - started
            results[name] = mask
            flagged[name] += int(bool(mask.warnings))
            cells.append(f"{mask.resin_frac:>11.1%}{'!' if mask.warnings else ' '} ")
            records.append({
                "image": path.name, "method": name,
                "resin_fraction": mask.resin_frac,
                "specimen_fraction": mask.specimen_frac,
                "components": mask.components,
                "warnings": list(mask.warnings),
            })

        rows.append(ComparisonRow(name=path.name, image=image.as_bgr(), results=results))
        print(f"{path.name[:width]:{width}} " + " ".join(cells))

    print("-" * len(header))
    print(f"{'Vorbehalte':{width}} " + " ".join(
        f"{flagged[m]:>10}/{len(rows)} " for m in methods))
    print(f"{'Sekunden gesamt':{width}} " + " ".join(
        f"{seconds[m]:>12.1f} " for m in methods))
    print("\n! = das Verfahren meldet einen Vorbehalt zum eigenen Ergebnis")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"JSON geschrieben: {args.json}")

    if args.sheet:
        print(f"Vergleichsblatt geschrieben: {comparison_sheet(rows, methods, args.sheet)}")

    return 0




# --------------------------------------------------------------------------------------
# run - die vollstaendige Pipeline
# --------------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from ..core.pipeline import PoreDetectionPipeline
    from ..detection.visualize import SheetEntry as PoreEntry
    from ..detection.visualize import contact_sheet as pore_sheet
    from ..io.exporters import csv_export, json_export

    cfg = load_config()
    if args.pore_method:
        cfg = cfg.model_copy(update={"pore": cfg.pore.model_copy(
            update={"method": args.pore_method})})
    if args.specimen_method:
        cfg = cfg.model_copy(update={"specimen": cfg.specimen.model_copy(
            update={"method": args.specimen_method})})

    try:
        pipeline = PoreDetectionPipeline(cfg)
    except KeyError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    try:
        paths = find_images(args.input, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print(f"Keine Bilder unter {args.input}", file=sys.stderr)
        return 2

    print(f"Verfahren: Massstab -> Overlay ausschliessen -> {cfg.specimen.method} "
          f"-> {cfg.pore.method}\n")

    width = min(38, max(len(p.name) for p in paths))
    header = (f"{'Bild':{width}} {'Poren':>6} {'Porositaet':>11} {'groesste':>12} "
              f"{'Dichte':>11} {'verw.':>6}")
    print(header)
    print("-" * len(header))

    batch = BatchResult(input_dir=Path(args.input), started_at=datetime.now())
    entries: list[PoreEntry] = []

    for path in paths:
        try:
            ctx = pipeline.analyse(path)
        except ImageReadError as exc:
            print(f"{path.name[:width]:{width}} -- {exc}")
            batch.failures.append((path, str(exc)))
            continue
        except Exception as exc:  # pragma: no cover
            print(f"{path.name[:width]:{width}} -- {type(exc).__name__}: {exc}")
            batch.failures.append((path, f"{type(exc).__name__}: {exc}"))
            continue

        result = pipeline._to_result(ctx, 0.0)
        batch.results.append(result)
        entries.append(PoreEntry(name=path.name, context=ctx, result=result))

        groesste = result.largest_pore
        if groesste is None:
            groesse = "-"
        elif groesste.equivalent_diameter_um is not None:
            groesse = f"{groesste.equivalent_diameter_um:.1f} um"
        else:
            groesse = f"{groesste.equivalent_diameter_px:.0f} px"

        dichte = result.pore_density_per_mm2
        porositaet = result.porosity_pct
        print(
            f"{path.name[:width]:{width}} {result.pore_count:>6} "
            f"{(f'{porositaet:.2f} %' if porositaet is not None else '-'):>11} "
            f"{groesse:>12} "
            f"{(f'{dichte:.1f}/mm2' if dichte is not None else '-'):>11} "
            f"{len(result.rejected):>6}"
        )
        if args.verbose:
            for name, text in result.stages.items():
                print(f"{'':{width}}   {name}: {text}")
        for warnung in result.warnings:
            print(f"{'':{width}}   ! {warnung}")

    batch.finished_at = datetime.now()
    print("-" * len(header))
    gesamt = batch.mean_porosity_pct
    print(f"{batch.total_pores} Poren in {len(batch.results)} Bildern"
          + (f", Gesamtporositaet {gesamt:.2f} %" if gesamt is not None else ""))
    if batch.failures:
        print(f"{len(batch.failures)} Bilder gescheitert")

    verworfen = csv_export.rejection_summary(batch)
    if verworfen:
        gesamt = sum(verworfen.values())
        print(f"\n{gesamt} Kandidaten von den Filtern verworfen:")
        for name, anzahl in verworfen.items():
            print(f"  {name:<16} {anzahl:>7}  ({anzahl / gesamt:5.1%})")

    if args.csv:
        poren, bilder, verworfen_csv = csv_export.write_all(batch, args.csv)
        print(f"CSV geschrieben: {poren}, {bilder}, {verworfen_csv}")
    if args.json:
        print(f"JSON geschrieben: {json_export.write(batch, args.json)}")
    if args.sheet:
        print(f"Kontrollblatt geschrieben: {pore_sheet(entries, args.sheet)}")

    return 0


# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# config - die globale Einstellungsdatei
# --------------------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    import yaml

    from ..config.loader import DEFAULT_PATH, reset_settings, settings_path

    if args.reset:
        print(f"Auf Standard zurueckgesetzt: {reset_settings()}")
        return 0

    path = settings_path()
    print(f"Einstellungsdatei: {path}")
    try:
        cfg = load_config(path)
    except (ValueError, yaml.YAMLError) as exc:  # ValidationError ist ein ValueError
        print(f"Fehler in der Einstellungsdatei: {exc}", file=sys.stderr)
        return 2
    standard = cfg == load_config(DEFAULT_PATH)
    print("Stand: Standardwerte" if standard else "Stand: weicht vom Standard ab")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
