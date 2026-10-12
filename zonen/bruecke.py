"""Anbindung an das Hauptprogramm - wird nur von ``start_with_zones.py`` eingehängt.

* ``pro_bild`` hängt sich als Auswertungsmodul in den Lauf: nach jedem ausgewerteten
  Bild und nach jeder Korrektur im Viewer entstehen CSV und Kontrollbild neu, im Ordner
  des Bildes wie beim Konsolenaufruf.
* ``fuer_viewer`` liefert dem Viewer die Zonen (Linien, Abstände, Porosität je Zone,
  Fehler an der Unterkante) als JSON-taugliches dict.

Die Masken kommen direkt aus dem Speicher. Importiert nichts aus ``poren`` - ``bild`` und
``ctx`` werden nur gelesen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ausgabe, einstellungen, rechnen


def pro_bild(bild, ordner: Path) -> None:
    """``bild``: ``poren.auswertungen.BildDaten``; ``ordner``: ``<ziel>/auswertungen/zonen``."""
    ziel = Path(ordner).parent.parent
    bildordner = _bildordner(ziel, bild)
    stamm = Path(bild.name).stem
    poren, einbettmittel = _masken(bild.labels, bild.poren, bild.einbettmittel, bild.probe)

    try:
        erg = rechnen.berechnen(bild.name, bild.probe, poren, einbettmittel, bild.overlay,
                                bild.um_pro_px, einstellungen.laden())
    except rechnen.KeineKante as grund:
        ausgabe.hinweis_schreiben(bildordner, stamm, bild.name, str(grund))
        print(f"Zonen {bild.name}: keine Zonen - {grund}")
        return
    hintergrund = bild.farbe if bild.farbe is not None else bild.grau
    ausgabe.schreiben(bildordner, stamm, erg, hintergrund, einbettmittel, poren)


def fuer_viewer(ctx) -> dict:
    """``ctx``: korrigierter ``poren.pipeline.PipelineContext``. Längen in µm, ohne
    Maßstab in px; Koordinaten in Bildpixeln wie die Porenumrisse (Pixelmitte = +0,5)."""
    poren, einbettmittel = _masken(ctx.labels, ctx.pores, ctx.resin, ctx.specimen)
    try:
        erg = rechnen.berechnen(ctx.path.name, ctx.specimen, poren, einbettmittel,
                                ctx.excluded, ctx.um_per_px, einstellungen.laden())
    except rechnen.KeineKante as grund:
        return {"fehlt": str(grund)}

    faktor = erg.um_pro_px or 1.0

    def laenge(px: float) -> float:
        return round(px * faktor, 3)

    def strecke(g: rechnen.Gerade) -> dict:
        x1, x2 = erg.x_von, erg.x_bis
        return {"x1": x1 + 0.5, "y1": round(g.y(x1) + 0.5, 2),
                "x2": x2 + 0.5, "y2": round(g.y(x2) + 0.5, 2)}

    grenzen = sorted(set(erg.grenzen_pct) | {0.0, 100.0})
    linien = [{"pct": g, "art": "mitte" if g == 50.0 else "rand" if g in (0.0, 100.0)
               else "grenze", **strecke(erg.grenzlinie(g))}
              for g in sorted(set(erg.grenzen_pct) | {50.0})]
    f = erg.fehler

    def rel(px: float) -> float:
        return round(100.0 * px / erg.dicke_px, 3)

    return {
        "einheit": "µm" if erg.um_pro_px else "px",
        "grenzen": list(erg.grenzen_pct),
        "linien": linien,
        "unterkante": strecke(erg.unten.gerade),
        "abstaende": [{"von": a, "bis": b, "laenge": laenge((b - a) / 100.0 * erg.dicke_px)}
                      for a, b in zip(grenzen, grenzen[1:])],
        "dicke": laenge(erg.dicke_px),
        "zonen": [{"name": z.name, "von": z.von_pct, "bis": z.bis_pct,
                   "probe_px": z.probe_px, "poren_px": z.poren_px,
                   "porositaet": None if z.porositaet_pct is None
                   else round(z.porositaet_pct, 4)} for z in erg.zonen],
        "gesamt": None if erg.gesamt.porositaet_pct is None
        else round(erg.gesamt.porositaet_pct, 4),
        "winkel_oben": round(erg.oben.gerade.winkel_grad, 3),
        "fehler": {"winkel": round(f.winkel_grad, 3),
                   "links": laenge(f.links_px), "links_rel": rel(f.links_px),
                   "rechts": laenge(f.rechts_px), "rechts_rel": rel(f.rechts_px),
                   "max": laenge(f.max_px), "max_rel": rel(f.max_px),
                   "rms": laenge(f.rms_px), "rms_rel": rel(f.rms_px)},
        "warnungen": list(erg.warnungen),
    }


def _masken(labels: np.ndarray, gezaehlte: list, einbettmittel: np.ndarray | None,
            probe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maske der gezählten Poren und Einbettmittel (leer, wenn keins erkannt wurde)."""
    gezaehlt = np.zeros(int(labels.max()) + 1, dtype=bool)
    gezaehlt[[p.label for p in gezaehlte]] = True
    if einbettmittel is None:
        einbettmittel = np.zeros_like(probe)
    return gezaehlt[labels], einbettmittel


def _bildordner(ziel: Path, bild) -> Path:
    """Der Ordner dieses Bildes im Zielordner: ``<stamm>`` oder - bei gleichnamigen Bildern
    aus Unterordnern - ``<stamm>_2``, ``<stamm>_3``, ... Erkannt am Bildnamen und an der
    Probenfläche in ``daten/einbettmittel.json``."""
    stamm = Path(bild.name).stem
    kandidaten, nummer = [ziel / stamm], 2
    while (ziel / f"{stamm}_{nummer}").is_dir():
        kandidaten.append(ziel / f"{stamm}_{nummer}")
        nummer += 1
    passend = []
    for ordner in kandidaten:
        try:
            kopf = json.loads((ordner / "daten" / "einbettmittel.json").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if kopf.get("bild") == bild.name:
            passend.append((kopf.get("probenflaeche_px"), ordner))
    for flaeche, ordner in passend:
        if flaeche == bild.ergebnis.specimen_area_px:
            return ordner
    if passend:
        return passend[0][1]
    raise FileNotFoundError(f"Ordner von {bild.name} unter {ziel} nicht gefunden")
