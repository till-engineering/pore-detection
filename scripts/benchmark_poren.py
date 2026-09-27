"""Benchmark der Porenerkennung gegen Referenzmasken - mehrere Verfahrensvarianten.

Gemessen wird **nur die Porenerkennung**: Kontrastbild, Detektor, Aufräumen (Öffnen,
Löcher füllen, Rauschgrenze), Vermessung und Formfilter. Alles andere läuft nicht:

* kein Maßstab und kein Einbettmittel - die Bilder haben beides nicht, das ganze Bild
  gilt als Probe,
* kein Trennen und kein Zusammenführen von Poren - das ändert nur die Aufteilung in
  Objekte, nicht, welche Pixel Pore sind,
* keine Abstände, Stege, Verteilungen oder Korrekturen.

Erwartet unter ``test/Benchmark``::

    images/<name>.tif
    masks/<name>_processed.tif      (0 = Gefüge, >0 = Pore)

Die Varianten (``--varianten``):

    ohne_untergrenze  das alte Verfahren, strenge Schwelle ohne Untergrenze
    standard          das heutige Verfahren, mit Untergrenze der strengen Schwelle
    halbwert          Rand je Pore nach dem Halbwertskriterium (--anteil)
    glatt             grobe Suche auf geglättetem Kontrastbild (--sigma)
    kombiniert        Halbwert und geglättet zusammen

Die Untergrenze gilt für alle Varianten außer ``ohne_untergrenze``; ohne
``--untergrenze`` ist es der Wert aus der Konfiguration.

Zwei Sichten, weil sie verschiedene Fragen beantworten:

* **Pixel** - stimmt die Fläche? Grundlage der Porosität.
* **Poren** - wird die Pore gefunden, und was wird zusätzlich gemeldet? Eine
  Referenzpore gilt als gefunden, sobald eine gezählte Pore sie berührt. Eine gemeldete
  Pore ohne jede Überlappung ist ein Fehlfund. Referenzobjekte unter 10 px sind fast
  durchweg Einzelpixel der Masken; sie werden getrennt ausgewiesen.

Aufruf::

    .venv\\Scripts\\python.exe scripts/benchmark_poren.py
    .venv\\Scripts\\python.exe scripts/benchmark_poren.py --varianten standard,halbwert
    .venv\\Scripts\\python.exe scripts/benchmark_poren.py --untergrenze 50 --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WURZEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WURZEL / "src"))

from poredet.analysis import FilterContext  # noqa: E402
from poredet.analysis import apply as filter_anwenden  # noqa: E402
from poredet.config.schema import AnalysisConfig, LocalContrastConfig, PoreConfig  # noqa: E402
from poredet.core.models import Pore, RejectedPore  # noqa: E402
from poredet.detection import contrast_image, estimate_background, get_detector  # noqa: E402
from poredet.detection import postprocess  # noqa: E402
from poredet.detection.base import DetectionInput  # noqa: E402
from poredet.io.image_reader import find_images, read_image  # noqa: E402
from poredet.measurement.pore_metrics import measure  # noqa: E402

#: Proben, deren Masken Gefügetextur statt Poren markieren (am Bild geprüft).
DEFEKTE_MASKEN = "27,47,48,59,80"

#: Größenklassen der Referenzporen nach Fläche in Pixeln.
KLASSEN = [(0, 10, "< 10 px"), (10, 50, "10-50 px"), (50, 200, "50-200 px"),
           (200, 10**9, "> 200 px")]

BESCHREIBUNG = {
    "ohne_untergrenze": "altes Verfahren, strenge Schwelle ohne Untergrenze",
    "standard": "heutiges Verfahren (mit Untergrenze)",
    "halbwert": "Rand je Pore nach Halbwert",
    "glatt": "grobe Suche geglättet",
    "kombiniert": "Halbwert und geglättet",
}


def varianten(untergrenze: float | None, sigma: float, anteil: float) -> dict[str, dict]:
    """Name -> Abweichungen in ``LocalContrastConfig`` gegenüber dem Standard.

    ``untergrenze`` gilt für alle Varianten außer ``ohne_untergrenze``; ``None`` heißt,
    es bleibt beim Wert aus der Konfiguration.
    """
    grenze = {} if untergrenze is None else {"strict_floor": untergrenze}
    halb = {"extent_method": "half_max", "half_max_fraction": anteil}
    glatt = {"presmooth_sigma_px": sigma}
    return {
        "ohne_untergrenze": {"strict_floor": 0.0},
        "standard": grenze,
        "halbwert": {**grenze, **halb},
        "glatt": {**grenze, **glatt},
        "kombiniert": {**grenze, **halb, **glatt},
    }


# --------------------------------------------------------------------------------------
# Porenerkennung - nur die Porenschritte
# --------------------------------------------------------------------------------------


@dataclass
class Erkennung:
    labels: np.ndarray
    kandidaten: np.ndarray
    pores: list[Pore]
    rejected: list[RejectedPore]
    streng: float


def erkennen(gray: np.ndarray, contrast: np.ndarray, background: np.ndarray,
             cfg: PoreConfig, analyse: AnalysisConfig) -> Erkennung:
    probe = np.ones(gray.shape, dtype=bool)
    ergebnis = get_detector(cfg.method).detect(
        DetectionInput(gray=gray, specimen=probe, background=background, contrast=contrast),
        cfg)
    maske = postprocess.clean(ergebnis.mask, cfg)
    labels = postprocess.label_image(maske, cfg)
    poren = measure(labels, gray, contrast, probe)
    behalten, verworfen = filter_anwenden(
        poren, analyse,
        FilterContext(image_area_px=gray.size, specimen_area_px=gray.size, um_per_px=None))
    return Erkennung(labels, ergebnis.mask, behalten, verworfen,
                     float(ergebnis.thresholds.get("streng", float("nan"))))


def bild_laden(pfad: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    bild = read_image(pfad)
    # Der Untergrund hängt an keiner Variante - einmal je Bild genügt.
    background = estimate_background(bild.gray, LocalContrastConfig().background)
    return bild.gray, bild.color, background, contrast_image(bild.gray, background)


def maske_lesen(pfad: Path) -> np.ndarray:
    roh = cv2.imdecode(np.fromfile(pfad, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if roh is None:
        raise OSError(f"{pfad} ist nicht lesbar")
    return roh > 0


def maskenpfad(ordner: Path, bild: Path) -> Path:
    return ordner / "masks" / f"{bild.stem}_processed.tif"


# --------------------------------------------------------------------------------------
# Vergleich mit der Referenz
# --------------------------------------------------------------------------------------


def klasse(flaeche: int) -> str:
    for unten, oben, name in KLASSEN:
        if unten <= flaeche < oben:
            return name
    return KLASSEN[-1][2]


def erkannt_maske(e: Erkennung) -> np.ndarray:
    lut = np.zeros(int(e.labels.max()) + 1, dtype=bool)
    lut[[p.label for p in e.pores]] = True
    return lut[e.labels]


def vergleichen(e: Erkennung, wahr: np.ndarray) -> tuple[dict, dict]:
    """Kennwerte eines Bildes, dazu Größenklassen und Verlustgründe."""
    erkannt = erkannt_maske(e)
    grund_je_label = {r.pore.label: r.filter_name for r in e.rejected}

    tp = int((erkannt & wahr).sum())
    fp = int((erkannt & ~wahr).sum())
    fn = int((~erkannt & wahr).sum())

    n_ref, ref_labels, ref_stats, _ = cv2.connectedComponentsWithStats(
        wahr.astype(np.uint8), connectivity=8)
    gefunden = np.zeros(n_ref, dtype=bool)
    gefunden[np.unique(ref_labels[erkannt])] = True

    klassen = {name: [0, 0] for *_, name in KLASSEN}
    verlust: Counter = Counter()
    for ref in range(1, n_ref):
        name = klasse(int(ref_stats[ref, cv2.CC_STAT_AREA]))
        klassen[name][0] += 1
        if gefunden[ref]:
            klassen[name][1] += 1
            continue
        if name == KLASSEN[0][2]:
            continue      # Einzelpixel der Maske - kein sinnvoller Verlustgrund
        x, y, w, h = (int(v) for v in ref_stats[ref, :4])
        region = ref_labels[y:y + h, x:x + w] == ref
        treffer = e.labels[y:y + h, x:x + w][region]
        treffer = treffer[treffer > 0]
        if treffer.size:
            verlust[f"Filter: {grund_je_label.get(int(np.bincount(treffer).argmax()), '?')}"] += 1
        elif e.kandidaten[y:y + h, x:x + w][region].any():
            verlust["Aufräumen / Rauschgrenze"] += 1
        else:
            verlust["nie Kandidat"] += 1

    fehlfunde = 0
    for pore in e.pores:
        b = pore.bbox
        region = e.labels[b.y:b.y + b.h, b.x:b.x + b.w] == pore.label
        if not wahr[b.y:b.y + b.h, b.x:b.x + b.w][region].any():
            fehlfunde += 1

    relevant = sum(klassen[n][0] for *_, n in KLASSEN[1:])
    relevant_gef = sum(klassen[n][1] for *_, n in KLASSEN[1:])
    zeile = {
        "tp_px": tp, "fp_px": fp, "fn_px": fn,
        "ref_poren_ab10": relevant, "ref_gefunden_ab10": relevant_gef,
        "erkannt_poren": len(e.pores), "fehlfunde": fehlfunde,
        "verworfen": len(e.rejected),
        "strenge_schwelle": e.streng,
        "porositaet_ref_pct": 100.0 * float(wahr.sum()) / wahr.size,
        "porositaet_erkannt_pct": 100.0 * float(erkannt.sum()) / wahr.size,
    }
    zeile.update(quoten(zeile))
    return zeile, {"klassen": klassen, "verlust": verlust}


def quoten(z: dict) -> dict:
    def teil(a, b):
        return a / b if b else float("nan")

    tp, fp, fn = z["tp_px"], z["fp_px"], z["fn_px"]
    rec = teil(z["ref_gefunden_ab10"], z["ref_poren_ab10"])
    prec = teil(z["erkannt_poren"] - z["fehlfunde"], z["erkannt_poren"])
    return {
        "precision_px": teil(tp, tp + fp),
        "recall_px": teil(tp, tp + fn),
        "dice_px": teil(2 * tp, 2 * tp + fp + fn),
        "iou_px": teil(tp, tp + fp + fn),
        "recall_poren_ab10": rec,
        "precision_poren": prec,
        "f1_poren": teil(2 * rec * prec, rec + prec),
    }


# --------------------------------------------------------------------------------------
# Ein Bild, alle Varianten - läuft in einem eigenen Prozess
# --------------------------------------------------------------------------------------


def bild_bewerten(pfad: Path, ordner: Path, cfgs: dict[str, PoreConfig]) -> list[tuple]:
    t0 = time.perf_counter()
    gray, _color, background, contrast = bild_laden(pfad)
    wahr = maske_lesen(maskenpfad(ordner, pfad))
    grund = time.perf_counter() - t0
    analyse = AnalysisConfig()

    aus = []
    for name, cfg in cfgs.items():
        t1 = time.perf_counter()
        e = erkennen(gray, contrast, background, cfg, analyse)
        dauer = grund + time.perf_counter() - t1
        zeile, extra = vergleichen(e, wahr)
        aus.append((name, {"bild": pfad.name, "dauer_s": dauer, **zeile}, extra))
    return aus


# --------------------------------------------------------------------------------------
# Zusammenfassen
# --------------------------------------------------------------------------------------


def zusammenfassen(zeilen: list[dict], klassen: dict, verlust: Counter) -> dict:
    summe = {k: sum(z[k] for z in zeilen) for k in
             ("tp_px", "fp_px", "fn_px", "ref_poren_ab10", "ref_gefunden_ab10",
              "erkannt_poren", "fehlfunde", "verworfen")}
    ref = np.array([z["porositaet_ref_pct"] for z in zeilen])
    erk = np.array([z["porositaet_erkannt_pct"] for z in zeilen])
    ohne_poren = [z for z in zeilen if z["porositaet_ref_pct"] == 0]
    return {
        **summe, **quoten(summe),
        "bilder": len(zeilen),
        "porositaet_ref_mittel_pct": float(ref.mean()),
        "porositaet_erkannt_mittel_pct": float(erk.mean()),
        "porositaet_abw_betrag_mittel_pct": float(np.abs(erk - ref).mean()),
        "porositaet_korrelation": float(np.corrcoef(ref, erk)[0, 1]),
        "dice_px_median_je_bild": float(np.nanmedian([z["dice_px"] for z in zeilen])),
        "bilder_ohne_referenzpore": len(ohne_poren),
        "fehlfunde_je_porenfreiem_bild": (float(np.mean([z["fehlfunde"] for z in ohne_poren]))
                                          if ohne_poren else None),
        "dauer_mittel_s": float(np.mean([z["dauer_s"] for z in zeilen])),
        "recall_je_groesse": {n: {"ref": a, "gefunden": g, "recall": g / a if a else None}
                              for n, (a, g) in klassen.items()},
        "verlustgruende_ab10": dict(verlust.most_common()),
    }


def csv_schreiben(pfad: Path, zeilen: list[dict]) -> None:
    with pfad.open("w", newline="", encoding="utf-8-sig") as f:
        schreiber = csv.DictWriter(f, fieldnames=list(zeilen[0]), delimiter=";")
        schreiber.writeheader()
        schreiber.writerows(zeilen)


# --------------------------------------------------------------------------------------
# Grafiken
# --------------------------------------------------------------------------------------


def ueberlagerung(gray, color, e: Erkennung, wahr: np.ndarray) -> np.ndarray:
    """Grün = richtig, rot = Fehlfund, blau = übersehen."""
    bild = color.copy() if color is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    erkannt = erkannt_maske(e)
    for maske, farbe in ((erkannt & wahr, (40, 170, 40)), (erkannt & ~wahr, (40, 40, 220)),
                         (~erkannt & wahr, (220, 120, 20))):
        if maske.any():
            bild[maske] = farbe
            umriss, _ = cv2.findContours(
                cv2.dilate(maske.astype(np.uint8), np.ones((3, 3), np.uint8)),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bild, umriss, -1, farbe, 1)
    return bild


def _achse(a, gitter="#dddddd") -> None:
    a.grid(color=gitter, lw=0.6, axis="y")
    a.set_axisbelow(True)
    for seite in ("top", "right"):
        a.spines[seite].set_visible(False)


def vergleich_grafik(gesamt: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    akzent, tinte = "#2a6fb0", "#333333"
    namen = list(gesamt)
    tafeln = [
        ("Dice (Pixel)", [gesamt[n]["dice_px"] for n in namen], "{:.2f}", (0, 1)),
        ("Recall Poren ab 10 px", [gesamt[n]["recall_poren_ab10"] for n in namen],
         "{:.0%}", (0, 1)),
        ("Precision Poren", [gesamt[n]["precision_poren"] for n in namen], "{:.0%}", (0, 1)),
        ("Porosität erkannt / Referenz",
         [gesamt[n]["porositaet_erkannt_mittel_pct"] / gesamt[n]["porositaet_ref_mittel_pct"]
          for n in namen], "{:.2f}x", None),
        ("Korrelation Porosität je Bild", [gesamt[n]["porositaet_korrelation"] for n in namen],
         "{:.2f}", (0, 1)),
        ("Fehlfunde je porenfreiem Bild",
         [gesamt[n]["fehlfunde_je_porenfreiem_bild"] or 0 for n in namen], "{:.1f}", None),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for a, (titel, werte, fmt, grenzen) in zip(ax.flat, tafeln, strict=True):
        balken = a.bar(namen, werte, color=akzent, width=0.6)
        oben = grenzen[1] if grenzen else max(werte) * 1.15 or 1
        for b, w in zip(balken, werte, strict=True):
            a.text(b.get_x() + b.get_width() / 2, w + oben * 0.01, fmt.format(w),
                   ha="center", va="bottom", fontsize=9, color=tinte)
        if titel.startswith("Porosität"):
            a.axhline(1.0, color="#999999", lw=1, ls="--")
        a.set_ylim(0, oben * 1.08)
        a.set_title(titel, fontsize=11, loc="left")
        a.tick_params(axis="x", labelsize=9, rotation=15)
        _achse(a)
    fig.suptitle(f"Varianten im Vergleich - {gesamt[namen[0]]['bilder']} Bilder", fontsize=12)
    fig.savefig(out / "vergleich.png", dpi=110, facecolor="white")
    plt.close(fig)


def vergleich_bilder(auswahl: list[tuple[str, str]], ordner: Path,
                     cfgs: dict[str, PoreConfig], out: Path) -> None:
    """Dieselben Bilder in jeder Variante nebeneinander - Grundlage der Entscheidung."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analyse = AnalysisConfig()
    zeilen, spalten = len(auswahl), len(cfgs)
    fig, ax = plt.subplots(zeilen, spalten, figsize=(4.2 * spalten, 3.1 * zeilen),
                           layout="constrained", squeeze=False)
    for r, (name, warum) in enumerate(auswahl):
        pfad = ordner / "images" / name
        gray, color, background, contrast = bild_laden(pfad)
        wahr = maske_lesen(maskenpfad(ordner, pfad))
        for c, (variante, cfg) in enumerate(cfgs.items()):
            e = erkennen(gray, contrast, background, cfg, analyse)
            z, _ = vergleichen(e, wahr)
            a = ax[r, c]
            a.imshow(cv2.cvtColor(ueberlagerung(gray, color, e, wahr), cv2.COLOR_BGR2RGB))
            kopf = f"{variante}\n" if r == 0 else ""
            a.set_title(f"{kopf}{name} ({warum})\nDice {z['dice_px']:.2f}  "
                        f"{z['erkannt_poren']} Poren, {z['fehlfunde']} Fehlfunde",
                        fontsize=8, loc="left")
            a.axis("off")
    fig.suptitle("Grün = richtig   rot = Fehlfund   blau = übersehen", fontsize=11)
    fig.savefig(out / "vergleich_bilder.png", dpi=90, facecolor="white")
    plt.close(fig)


def bilder_waehlen(zeilen: list[dict]) -> list[tuple[str, str]]:
    """Aus der Standardvariante: die typischen Problemfälle und ein Normalfall."""
    ohne = sorted((z for z in zeilen if z["porositaet_ref_pct"] == 0),
                  key=lambda z: -z["fehlfunde"])[:2]
    mit = sorted((z for z in zeilen if z["porositaet_ref_pct"] > 0),
                 key=lambda z: z["dice_px"] if z["dice_px"] == z["dice_px"] else 0)
    mitte = mit[len(mit) // 2 - 1: len(mit) // 2 + 1] if len(mit) >= 2 else mit
    gross = sorted(mit, key=lambda z: -z["porositaet_ref_pct"])
    gross = [z for z in gross if z["bild"].split("_cropped")[0] not in {"35", "43", "46-1"}][:2]
    auswahl = [(z["bild"], "ohne Pore, viele Fehlfunde") for z in ohne]
    auswahl += [(z["bild"], "mittlerer Fall") for z in mitte]
    auswahl += [(z["bild"], "große Poren") for z in gross]
    return auswahl


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ordner", type=Path, default=WURZEL / "test/Benchmark")
    parser.add_argument("--out", type=Path, default=WURZEL / "data/runs/benchmark_poren")
    parser.add_argument("--varianten",
                        default="ohne_untergrenze,standard,halbwert,glatt,kombiniert")
    parser.add_argument("--untergrenze", type=float, default=None,
                        help="Untergrenze der strengen Schwelle in Graustufen. Ohne Angabe "
                             "der Wert aus der Konfiguration (derzeit 40).")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Glättung der groben Suche in Pixeln (Standard 1.0)")
    parser.add_argument("--anteil", type=float, default=0.5,
                        help="Lage des Rands beim Halbwertskriterium (Standard 0.5)")
    parser.add_argument("--ohne", default=DEFEKTE_MASKEN,
                        help=f"Proben ausschließen (Präfix vor '_cropped'). Standard: die "
                             f"mit defekten Masken, {DEFEKTE_MASKEN}. Leer = alle.")
    parser.add_argument("--limit", type=int, default=0, help="nur die ersten n Bilder")
    parser.add_argument("--prozesse", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    alle = varianten(args.untergrenze, args.sigma, args.anteil)
    gewaehlt = [v.strip() for v in args.varianten.split(",") if v.strip()]
    unbekannt = [v for v in gewaehlt if v not in alle]
    if unbekannt:
        parser.error(f"unbekannte Variante(n): {', '.join(unbekannt)} - "
                     f"bekannt: {', '.join(alle)}")
    cfgs = {v: PoreConfig(local_contrast=LocalContrastConfig(**alle[v])) for v in gewaehlt}

    ohne = {s.strip() for s in args.ohne.split(",") if s.strip()}
    bilder = [p for p in find_images(args.ordner / "images")
              if p.name.split("_cropped")[0] not in ohne
              and maskenpfad(args.ordner, p).is_file()]
    if args.limit:
        bilder = bilder[: args.limit]
    if not bilder:
        print("Keine Bilder mit Maske gefunden.", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(bilder)} Bilder, Varianten: {', '.join(cfgs)}, {args.prozesse} Prozesse")

    zeilen = {v: [] for v in cfgs}
    klassen = {v: {n: [0, 0] for *_, n in KLASSEN} for v in cfgs}
    verlust = {v: Counter() for v in cfgs}
    beginn = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.prozesse) as pool:
        auftraege = [pool.submit(bild_bewerten, p, args.ordner, cfgs) for p in bilder]
        for fertig, auftrag in enumerate(as_completed(auftraege), start=1):
            for variante, zeile, extra in auftrag.result():
                zeilen[variante].append(zeile)
                verlust[variante].update(extra["verlust"])
                for n, (a, g) in extra["klassen"].items():
                    klassen[variante][n][0] += a
                    klassen[variante][n][1] += g
            if fertig % 50 == 0 or fertig == len(bilder):
                print(f"  {fertig}/{len(bilder)}  ({time.perf_counter() - beginn:.0f} s)")

    gesamt = {}
    for v in cfgs:
        zeilen[v].sort(key=lambda z: z["bild"])
        csv_schreiben(args.out / f"bilder_{v}.csv", zeilen[v])
        gesamt[v] = {"beschreibung": BESCHREIBUNG[v], "einstellungen": alle[v],
                     **zusammenfassen(zeilen[v], klassen[v], verlust[v])}
    (args.out / "zusammenfassung.json").write_text(
        json.dumps({"ausgeschlossen": sorted(ohne), "varianten": gesamt},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    kurz = ["dice_px", "precision_px", "recall_px", "recall_poren_ab10", "precision_poren",
            "porositaet_ref_mittel_pct", "porositaet_erkannt_mittel_pct",
            "porositaet_korrelation", "fehlfunde_je_porenfreiem_bild", "dauer_mittel_s"]
    csv_schreiben(args.out / "vergleich.csv",
                  [{"variante": v, **{k: gesamt[v][k] for k in kurz}} for v in cfgs])

    vergleich_grafik(gesamt, args.out)
    # Die Bildauswahl aus dem heutigen Standard, sonst aus der ersten Variante.
    basis = "standard" if "standard" in cfgs else next(iter(cfgs))
    vergleich_bilder(bilder_waehlen(zeilen[basis]), args.ordner, cfgs, args.out)

    print(f"\n{'Variante':<12} {'Dice':>6} {'Rec>=10':>8} {'PrecPor':>8} "
          f"{'Por erk/ref':>12} {'r':>6} {'FF/leer':>8}")
    for v in cfgs:
        g = gesamt[v]
        print(f"{v:<12} {g['dice_px']:>6.3f} {g['recall_poren_ab10']:>8.1%} "
              f"{g['precision_poren']:>8.1%} "
              f"{g['porositaet_erkannt_mittel_pct']:>5.2f}/{g['porositaet_ref_mittel_pct']:<5.2f} "
              f"{g['porositaet_korrelation']:>6.2f} "
              f"{(g['fehlfunde_je_porenfreiem_bild'] or 0):>8.1f}")
    print(f"\nErgebnisse in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
