---
title: EchoLab Fan Anomaly Detection
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
---

# EchoLab: Fan Anomaly Detection (MobileNetV2 + PCA)

Finding faulty fans by listening to their sound · MIMII Fan id_00 · website + training pipeline + Raspberry Pi version.

## Quick start (3 steps)

```bash
pip install -r requirements.txt        # once (Python 3.10-3.12)
python app.py                          # opens http://localhost:7860
```
**First launch has no model** (there is no demo model on purpose). Open the **Setup & Train** tab and press
**Download real data and train the model**. It downloads the MIMII fan data (dev_data_fan.zip, 1.4 GB), trains, measures, and the website fills
with your real accuracy / AUC / F1. The same thing from a terminal: `python echolab_train.py`.
No GPU? Use Colab: upload this zip and run `colab_echolab.ipynb` (T4 GPU, about 1-2 h), then unzip the result into this folder.

**This is a real-dataset project: it uses the MIMII / DCASE 2020 Task 2 fan recordings and nothing else.** If the real data or the
ImageNet weights cannot be obtained it stops with a message; it never generates or substitutes artificial sound.

Already have the dataset? Use it directly (your files are copied, never moved or changed):

```bash
python echolab_train.py --archive D:/datasets/dev_data_fan.zip     # the zip from Zenodo record 3678171
python echolab_train.py --data-dir D:/datasets/fan                  # or an extracted folder
```
(also possible in the website: *Setup & Train* → "Already have the dataset?"). Original MIMII works too (`MIMII/fan/.../id_00/normal/*.wav`). Its recordings have 8 microphone channels: only the first is used,
as in the DCASE data. If several noise versions (-6, 0, 6 dB) are unzipped together, exactly one is used (6 dB by default,
`--snr 0` to change) and the log says which. Original MIMII has no official train/test split here, so normal clips are split
80/20 and all faulty clips go to the test set; the clip counts will then differ from the deck's 1003 / 100 / 341.
Accepted layouts: DCASE
(`fan/train/normal_id_00_*.wav`, `fan/test/normal_id_00_*.wav`, `fan/test/anomaly_id_00_*.wav`) or MIMII
(`id_00/normal/*.wav`, `id_00/abnormal/*.wav`).
`python echolab_train.py --quick` runs a fast check on a small subset of the real data; its model is labelled QUICK everywhere.

## What is in the deck and where it is in the project

| Deck content | Where |
|---|---|
| Fan id_00, 1003 healthy train, 341 faulty + 100 healthy test, 10 s clips | `download_data.py`, `preprocess.py --ids 00`, counts printed and shown in the website |
| 16 kHz, 128 x 313 Mel "sound image" | `features_np.py` (NumPy, identical in training, website and Pi; checked against librosa) |
| Making more data: stretch time, shift pitch, mask parts of the image | `fanlib.deck_augment` |
| Pre-trained MobileNetV2 on the full 10 s | `fanlib.build_model`, `echolab_train.py` (two-phase transfer learning) |
| Fault score, best threshold (ROC) | ROC-best (cross-fitted) and label-free thresholds, `echolab_train.py` |
| Stratified 5-fold CV (802 train / 201 val), final unseen test, average AUC ± spread | `echolab_train.py` → Model & Results tab, `report/REPORT.md` |
| Bias / variance, why not LOOCV / 2-fold / time-series / simple K-fold | `report/REPORT.md` §6, "How it works" tab, `VIVA.md` |
| AUC, pAUC, F1, **accuracy**, confusion matrix, ROC | Model & Results tab (all read from `metrics.json`) |
| Baseline (autoencoder 72.89 %) vs ours | bar chart + target table in Model & Results |
| Grad-CAM ("the model listens at 4-8 kHz") | Analyze tab (live) and `figures/gradcam.png` |
| Quantised 14 MB → 4 MB, Raspberry Pi version | `model.tflite` (dynamic-range quantised), `pi_inference.py`, `benchmark_latency.py`, `HARDWARE.md` |
| Dashboard green / red | Analyze tab verdict card |
| Demo QR + Gradio link | Share tab, `python app.py --share` |
| Future work: lab fan with a mic | Analyze tab records from the microphone; own-fan calibration |
| Future work: phone app + Telegram alerts | Telegram alert in the Analyze tab; the site is phone-friendly (QR) |
| Future work: pumps and valves | `python echolab_train.py --machine pump --ids 00` (also valve, slider, ...) |
| **PCA** (added) | spectrogram PCA + embedding PCA, fused with the CNN; PCA Lab tab |

## How the model learns with healthy sounds only

No faulty clip is used for training. Each batch is 50 % real healthy and 50 % *pseudo-faulty* clips made from the same real healthy
recordings by injecting fault patterns (4-8 kHz rattle, clicks, tones, amplitude modulation, noise floor). Two PCA models fitted on healthy sound add reconstruction-error
scores. The three scores are turned into z-scores against healthy sound and fused with equal weights, decided in advance and never
tuned on test data. Whether the injected patterns transfer to real faults is exactly what the unseen real test clips measure.

## Things in the deck to fix before the examiner sees it

1. **"Stratified" 5-fold on healthy clips only.** With one class, stratification is meaningless. In this project each fold keeps a
   1:1 healthy : pseudo-fault mix (say so on slide 8).
2. **Accuracy is misleading here.** 341 faulty vs 100 healthy means "always faulty" scores 77.3 % accuracy. The website
   shows accuracy *and* balanced accuracy, AUC and F1.
3. **128 x 313** is right (centre-padded frames); the earlier code used 311. Now consistent.
4. **Baseline 72.89 % AUC / 60 % pAUC:** check it against the DCASE 2020 Task 2 results table *for fan id_00*; I could not verify
   it (to my recollection the official per-ID fan figures are different and id_00 is among the harder IDs).
5. **Literature AUCs** (88, 85, 82 %), "id_00 has the least noise", "14 MB → 4 MB", "90+ % target", "25 ms laptop / 90 ms Pi"
   are claims or targets, not measurements. The report marks each target MET / NOT MET from the measured file; the Pi speed shows
   "not measured" until you run `benchmark_latency.py` on a Pi.
6. The deck lists the model as 14 MB. A MobileNetV2 backbone is about 9 MB in fp32 and about 2.5-4 MB quantised;
   the website shows the measured sizes.

## Files

| File | Purpose |
|---|---|
| `app.py` | the website |
| `echolab_train.py` | download → preprocess → 5-fold CV → final model → PCA → thresholds → export → report |
| `echolab_core.py` | inference engine (scores, fusion, calibration, figures, Telegram) |
| `echolab_report.py` | `report/REPORT.md` from `metrics.json` |
| `fanlib.py`, `analyze.py`, `features_np.py`, `mel_basis.npy` | model, augmentation, pseudo-faults, Grad-CAM, PCA, features |
| `download_data.py`, `preprocess.py` | data (`--machine`, `--archive`, `--real-only`) |
| `tflite_runner.py`, `benchmark_latency.py`, `pi_inference.py`, `fan-monitor.service`, `HARDWARE.md` | Raspberry Pi version (optional) |
| `colab_echolab.ipynb` | train on Colab's GPU |
| `VIVA.md` | examiner questions with answers |

## Troubleshooting

* *"No trained model yet"*: run the Setup & Train tab or `python echolab_train.py`.
* *Download blocked*: use `--archive` (see above).
* *"ImageNet weights could not be downloaded"*: check the firewall for storage.googleapis.com, or `--allow-scratch` (lower accuracy).
* *Grad-CAM panel empty*: needs TensorFlow and `models/echolab/model.keras` (the website and trainer run on the same computer).
* *Training interrupted*: run the same command again; finished folds are resumed (`--fresh` to start over).
