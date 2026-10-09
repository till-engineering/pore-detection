"""Vergleichsbilder und Bericht der synthetischen Prüfung.

Je Bild ``<stem>_vergleich.png``: links das Bild, rechts dasselbe mit

* hellgrün   wahre Formen
* rot        vom Programm gezählte Poren
* orange     vom Programm verworfene Objekte
* cyan       wahre Probengrenze
* magenta    vom Programm erkannte Probengrenze
* gelbe Nummern   Formen mit WARNUNG oder FEHLER, ``F`` = Falschfund

Dazu ``bericht.html`` (zum Ansehen) und ``bericht.txt`` (zum Vergleichen zweier Läufe) -
beide direkt im Testordner, die Vergleichsbilder in ``pruefung/``.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from .pruefen import FEHLER, HINWEIS, OK, STUFEN, TOLERANZ, WARNUNG, BildErgebnis, Befund

GRUEN, ROT, ORANGE = (90, 220, 90), (40, 40, 255), (0, 150, 255)
CYAN, MAGENTA, GELB = (255, 220, 0), (255, 0, 255), (0, 230, 255)
LUECKE = 10


def vergleichsbild(pfad: Path, bild: np.ndarray, probe_wahr: np.ndarray, formen_px: np.ndarray,
                   ergebnis, ctx, b: BildErgebnis) -> None:
    grau = cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR) if bild.ndim == 2 else bild.copy()
    # Aufgehellt, damit die Linien auf dunklen Poren und hellem Gefüge sichtbar sind.
    rechts = (0.55 * grau + 0.45 * 255).astype(np.uint8)

    _umrisse(rechts, probe_wahr, CYAN, 2)
    _umrisse(rechts, ctx.specimen, MAGENTA, 1)
    _je_label(rechts, formen_px, None, GRUEN)
    _je_label(rechts, ctx.labels, {r.pore.label for r in ergebnis.rejected}, ORANGE)
    _je_label(rechts, ctx.labels, {p.label for p in ergebnis.pores}, ROT)

    nach_id = {f.id: f for f in b.formen}
    fenster = ndi.find_objects(formen_px)
    for fid, f in nach_id.items():
        if f.stufe in (WARNUNG, FEHLER) and fid - 1 < len(fenster) and fenster[fid - 1]:
            ys, xs = fenster[fid - 1]
            _marke(rechts, str(fid), xs.stop + 2, ys.start)
    fenster = ndi.find_objects(ctx.labels)
    for label in b.falschfunde:
        ys, xs = fenster[label - 1]
        _marke(rechts, "F", xs.stop + 2, ys.start)

    steg = np.full((grau.shape[0], LUECKE, 3), 255, np.uint8)
    gesamt = np.hstack([grau, steg, rechts])
    legende = _legende(gesamt.shape[1])
    ok, puffer = cv2.imencode(".png", np.vstack([gesamt, legende]))
    puffer.tofile(str(pfad))


def _umrisse(ziel, maske, farbe, dicke) -> None:
    konturen, _ = cv2.findContours(maske.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ziel, konturen, -1, farbe, dicke)


def _je_label(ziel, labels, nur: set[int] | None, farbe) -> None:
    for label, f in enumerate(ndi.find_objects(labels), start=1):
        if f is None or (nur is not None and label not in nur):
            continue
        konturen, _ = cv2.findContours((labels[f] == label).astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE, offset=(f[1].start, f[0].start))
        cv2.drawContours(ziel, konturen, -1, farbe, 1)


def _marke(ziel, text, x, y) -> None:
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x = min(max(0, x), ziel.shape[1] - w - 4)
    y = min(max(h + 4, y + h + 2), ziel.shape[0] - 2)
    cv2.rectangle(ziel, (x - 2, y - h - 3), (x + w + 2, y + 3), GELB, -1)
    cv2.putText(ziel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _legende(breite: int) -> np.ndarray:
    leiste = np.full((34, breite, 3), 255, np.uint8)
    x = 10
    for farbe, text in ((GRUEN, "wahre Form"), (ROT, "gezaehlt"), (ORANGE, "verworfen"),
                        (CYAN, "wahre Probengrenze"), (MAGENTA, "erkannte Probengrenze"),
                        (GELB, "Nr = Warnung/Fehler, F = Falschfund")):
        cv2.rectangle(leiste, (x, 10), (x + 18, 24), farbe, -1)
        cv2.putText(leiste, text, (x + 24, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)
        x += 24 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 26
    return leiste


# --------------------------------------------------------------------------------------
# Bericht
# --------------------------------------------------------------------------------------


def zaehlen(bilder: list[BildErgebnis], gesamt: list[Befund]) -> dict[str, int]:
    stufen = [x.stufe for b in bilder for x in (*b.befunde, *b.formen)] + [x.stufe for x in gesamt]
    return {s: stufen.count(s) for s in STUFEN}


def text(bilder: list[BildErgebnis], gesamt: list[Befund], kopf: dict) -> str:
    z = zaehlen(bilder, gesamt)
    zeilen = [
        "Synthetische Prüfung - Pore Detection",
        f"Zeit: {kopf['zeit']}   Seed: {kopf['seed']}   Bilder: {len(bilder)}",
        f"Einstellungen: {kopf['einstellungen']}",
        f"Ergebnis: {_gesamtstufe(bilder, gesamt)}   "
        + "   ".join(f"{s}: {z[s]}" for s in STUFEN), "",
    ]
    for b in bilder:
        k = b.kennzahlen
        zeilen.append(f"[{b.stufe:7s}] {b.name}  Probe-IoU {k.get('probe_iou', 0):.4f}  "
                      f"Porosität {_pct(k.get('porositaet_ist'))} (wahr {_pct(k.get('porositaet_soll'))})")
        for x in b.befunde:
            if x.stufe != OK:
                zeilen.append(f"    {x.stufe:7s} {x.pruefung}: {x.text}")
        for f in b.formen:
            if f.stufe != OK:
                zeilen.append(f"    {f.stufe:7s} Form {f.id} ({f.kategorie}, {f.art}): {f.text}")
                for v in f.vergleiche:
                    if v.stufe != OK:
                        zeilen.append(f"            {v.groesse}: soll {v.soll}, ist {v.ist} ({v.abweichung})")
    zeilen.append("")
    for x in gesamt:
        zeilen.append(f"[{x.stufe:7s}] Gesamt {x.pruefung}: {x.text}")
    return "\n".join(zeilen) + "\n"


def html_bericht(bilder: list[BildErgebnis], gesamt: list[Befund], kopf: dict,
                 bildordner: str = "pruefung") -> str:
    """``bildordner``: wo die Vergleichsbilder liegen, relativ zum Bericht."""
    z = zaehlen(bilder, gesamt)
    e = html.escape
    teile = [f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetische Prüfung</title><style>{_CSS}</style></head><body><main>
<h1>Synthetische Prüfung</h1>
<p class="meta">{e(kopf['zeit'])} · Seed {kopf['seed']} · {len(bilder)} Bilder ·
Einstellungen: {e(kopf['einstellungen'])}</p>
<div class="kopf {_gesamtstufe(bilder, gesamt).lower()}"><b>{_gesamtstufe(bilder, gesamt)}</b>
{' · '.join(f'{s}: {z[s]}' for s in STUFEN)}</div>
<table><thead><tr><th>Bild</th><th>Stufe</th><th>Probe-IoU</th><th>Porosität ist</th>
<th>Porosität wahr</th><th>Formen OK</th><th>Falschfunde</th></tr></thead><tbody>"""]
    for b in bilder:
        k = b.kennzahlen
        ok = sum(f.stufe == OK for f in b.formen)
        teile.append(f"<tr><td><a href='#{b.stem}'>{e(b.name)}</a></td>{_zelle(b.stufe)}"
                     f"<td>{k.get('probe_iou', 0):.4f}</td><td>{_pct(k.get('porositaet_ist'))}</td>"
                     f"<td>{_pct(k.get('porositaet_soll'))}</td><td>{ok} / {len(b.formen)}</td>"
                     f"<td>{len(b.falschfunde)}</td></tr>")
    teile.append("</tbody></table>")
    if gesamt:
        teile.append("<h2>Tabellen über alle Bilder</h2><table><tbody>")
        teile += [f"<tr><td>{e(x.pruefung)}</td>{_zelle(x.stufe)}<td>{e(x.text)}</td></tr>"
                  for x in gesamt]
        teile.append("</tbody></table>")

    for b in bilder:
        teile.append(f"<section id='{b.stem}'><h2>{e(b.name)} {_marke_html(b.stufe)}</h2>"
                     f"<a href='{bildordner}/{b.stem}_vergleich.png'><img src='{bildordner}/{b.stem}_vergleich.png' "
                     f"alt='Vergleich {e(b.name)}'></a>"
                     "<h3>Prüfungen</h3><table><tbody>")
        teile += [f"<tr><td>{e(x.pruefung)}</td>{_zelle(x.stufe)}<td>{e(x.text)}</td></tr>"
                  for x in b.befunde]
        teile.append("</tbody></table><h3>Formen</h3><table><thead><tr><th>Nr</th>"
                     "<th>Kategorie</th><th>Art</th><th>Stufe</th><th>IoU</th><th>Ergebnis</th>"
                     "<th>Fläche soll / ist</th><th>Abweichungen</th></tr></thead><tbody>")
        for f in b.formen:
            flaeche = next((v for v in f.vergleiche if v.groesse.startswith("Fläche")), None)
            abw = "; ".join(f"{v.groesse}: {v.soll} → {v.ist} ({v.abweichung})"
                            for v in f.vergleiche if v.stufe != OK)
            teile.append(
                f"<tr><td>{f.id}</td><td>{e(f.kategorie)}</td><td>{e(f.art)}</td>{_zelle(f.stufe)}"
                f"<td>{'' if f.iou is None else f'{f.iou:.3f}'}</td><td>{e(f.text)}</td>"
                f"<td>{'' if flaeche is None else f'{flaeche.soll} / {flaeche.ist:.0f}'}</td>"
                f"<td>{e(abw)}</td></tr>")
        teile.append("</tbody></table></section>")

    teile.append("<h2>Toleranzen</h2><p class='meta'>Bis zur Toleranz OK, bis zum Doppelten "
                 "WARNUNG, darüber FEHLER. Innere Stimmigkeit (Maske, Ergebnisbild, CSV, JSON) "
                 "muss exakt stimmen.</p><table><tbody>")
    teile += [f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in TOLERANZ.items()]
    teile.append("</tbody></table></main></body></html>")
    return "\n".join(teile)


def _gesamtstufe(bilder, gesamt) -> str:
    stufen = [b.stufe for b in bilder] + [x.stufe for x in gesamt]
    s = max(stufen, key=STUFEN.index, default=OK)
    return OK if s == HINWEIS else s


def _pct(wert) -> str:
    return "–" if wert is None else f"{wert:.4f} %"


def _zelle(stufe: str) -> str:
    return f"<td>{_marke_html(stufe)}</td>"


def _marke_html(stufe: str) -> str:
    return f"<span class='stufe {stufe.lower()}'>{stufe}</span>"


_CSS = """
:root{--grund:#f4f6f8;--flaeche:#fff;--tinte:#141820;--leise:#5d6775;--linie:#d5dae1;
--ok:#1f7a3a;--hinweis:#2f6fd0;--warnung:#a8650a;--fehler:#c0262d}
@media (prefers-color-scheme:dark){:root{--grund:#10131a;--flaeche:#181d26;--tinte:#e7ecf2;
--leise:#96a1b0;--linie:#2c3543;--ok:#4cc27a;--hinweis:#6398ea;--warnung:#e0a03c;--fehler:#f0656b}}
body{margin:0;background:var(--grund);color:var(--tinte);font:14px/1.5 system-ui,"Segoe UI",sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px}
h1{margin:0 0 4px;font-size:22px}h2{margin:32px 0 8px;font-size:17px}h3{margin:16px 0 6px;font-size:14px}
.meta{color:var(--leise);margin:0 0 12px}
.kopf{padding:10px 14px;border-radius:6px;background:var(--flaeche);border:1px solid var(--linie);
border-left-width:6px;margin-bottom:16px}
.kopf.ok{border-left-color:var(--ok)}.kopf.warnung{border-left-color:var(--warnung)}
.kopf.fehler{border-left-color:var(--fehler)}
table{border-collapse:collapse;width:100%;background:var(--flaeche);font-size:13px}
th,td{border:1px solid var(--linie);padding:4px 8px;text-align:left;vertical-align:top}
th{background:var(--grund);font-weight:600}
img{max-width:100%;height:auto;border:1px solid var(--linie);display:block}
.stufe{font-weight:600;font-size:12px}.stufe.ok{color:var(--ok)}.stufe.hinweis{color:var(--hinweis)}
.stufe.warnung{color:var(--warnung)}.stufe.fehler{color:var(--fehler)}
section{margin-top:28px}a{color:inherit}
"""
