# Bedienung – alle Skripte auf einen Blick

Alle Befehle im Projektordner, mit der Python der `.venv`:

```powershell
cd E:\Arbeit\Pore_Detection
.venv\Scripts\python <skript> [optionen]
```

## Installation (einmalig)

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
# optional, GPU-Wege + U-Net:
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install "segmentation-models-pytorch>=0.5"
# MicroNet-Gewichte fürs U-Net-Training (kein pip-Paket, ~90 MB in den Torch-Cache):
.venv\Scripts\python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/jstuckner/microscopy-resnet50-micronet/resolve/main/resnet50_micronet_weights.pth')"
```

## Übersicht

| Was | Befehl | Einstellungen | Ergebnis |
|---|---|---|---|
| Porenerkennung (Kommandozeile) | `main.py` | Config im Hauptordner | `output/<ID>_results/` |
| Porenerkennung (GUI) | `gui_main.py` | Config im Hauptordner, Pfade im Launcher | wie oben, danach Viewer |
| Einstellungen von Hand finden | `tools/find_settings.py [bild]` | Config im Hauptordner | beste Parameter im Terminal |
| Auswertung gegen Masken | `dataset_eval/evaluate.py` | `dataset_eval/eval_config.yaml` | `evaluation.xlsx`, `plots/`, `overlays/` |
| Parametersuche (eine Einstellung für alle) | `dataset_eval/tune.py` | `eval_config.yaml`, Abschnitt `tune:` | `tune/trials.xlsx`, `best_pipeline.yaml` |
| U-Net trainieren | `dataset_eval/train_unet.py` | `dataset_eval/train_config.yaml` | `models/<name>/` |
| Synthetische Testbilder erzeugen | `syntetic_test/generate.py` | `syntetic_test/synth_config.yaml` | `syntetic_test/output/` |
| Pipeline auf synthetischen Bildern prüfen | `syntetic_test/evaluate.py` | Config im Hauptordner | Werte im Terminal |

Jedes Skript zeigt mit `--help` seine Optionen. Häufige:

- `evaluate.py --no-run`: nur neu auswerten, die Pipeline läuft nicht nochmal
- `evaluate.py`, `tune.py`, `train_unet.py` mit `--config andere.yaml`: andere Einstellungsdatei

## Verfahren wechseln

Das Programm nimmt die **eine** `config*.yaml` im Hauptordner. Zum Wechseln die aktuelle nach
`configs/` zurückschieben und die gewünschte von dort in den Hauptordner ziehen, z. B.
`configs/config.yaml` (Standard) oder `configs/config_unet.yaml`. Keine oder mehr als eine
`config*.yaml` im Hauptordner = Fehler mit den gefundenen Namen. Der Launcher zeigt die aktive
Config im Reiter „Config (<name>)“ und im Log, `main.py` und `evaluate.py` im Terminal.

Jede Variante ist eine vollständige Pipeline, in der nur Vorverarbeitung, Detektion und
Porenschritte geändert sind. Zum Vergleichen `dataset_eval/evaluate.py` laufen lassen.

Alternativ ohne Kopieren: Die Blöcke unter `pipeline:` in `dataset_eval/eval_config.yaml` eintragen.
Sie überschreiben die Config im Hauptordner nur für die Auswertung.

## U-Net: Ablauf

1. **Daten:** Bilder und Masken in zwei Ordnern mit gleichem Dateinamen. Die Masken sind schwarz, die Poren weiß.
2. **`dataset_eval/train_config.yaml`:** `data.images` und `data.masks` setzen. Wenn die Probe im
   Dateinamen steht, `split.group_pattern` setzen, z. B. `"^[^_]+"` für `P017_patch_03`.
   Standard ist ResNet50 mit MicroNet.
3. **Trainieren:**
   ```powershell
   .venv\Scripts\python dataset_eval/train_unet.py
   ```
   - Dauer auf der RTX 2060 bei ca. 100 Bildern: etwa 1,5–2 h.
   - Strg+C bricht ab, das beste Modell bleibt erhalten.
   - Ergebnis: `models/unet_<datum>_<zeit>/`. Die Schwelle und der Validierungs-Dice stehen in `model.yaml`, die Kurven in `training_curves.png`.
4. **Einsetzen:** In `config_unet.yaml` den Wert `model_dir: models/<name>` eintragen und die Datei als einzige Config in den Hauptordner legen.
5. **Prüfen:** `dataset_eval/evaluate.py` laufen lassen. Ehrlich ist das nur auf Bildern, die **nicht** unter `train:` in `models/<name>/split.yaml` stehen, z. B. in einem eigenen Testordner.

**Wichtig:**
- Das Modell gilt für die Vergrößerung bzw. das Objektiv der Trainingsbilder, denn es wird nicht skaliert.
- Für ein neues Objektiv mit passenden Bildern neu trainieren.
