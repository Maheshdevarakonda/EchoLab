#!/usr/bin/env python3
"""
echolab_report.py -- writes report/REPORT.md from models/echolab/metrics.json.

Every number comes from the measured file; deck targets are checked automatically and marked
MET / NOT MET.  Called at the end of echolab_train.py, or run on its own:  python echolab_report.py
Convert:  pandoc report/REPORT.md -o report/REPORT.docx --toc
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

M_PATH = Path("models/echolab/metrics.json")
LABEL = {"cnn": "MobileNetV2 fault score", "spca": "PCA on spectrogram", "epca": "PCA on embedding",
         "fused": "EchoLab fused score (CNN + 2 PCA)"}
SCORES = ["cnn", "spca", "epca", "fused"]


def pc(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f} %"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main(path=M_PATH, out="report/REPORT.md"):
    R = json.loads(Path(path).read_text())
    meta, data, mod, cv, fin = R["meta"], R["data"], R["model"], R["cv"], R["final"]
    syn = bool(meta.get("quick"))
    L = []
    w = L.append
    if syn:
        w("> **DRAFT: QUICK RUN.** This report comes from a --quick run on a small subset of the data. "
          "Run `python echolab_train.py` without --quick for the real result and regenerate.\n")
    if not meta.get("pretrained_imagenet") and not syn:
        w("> **Note:** the ImageNet weights were not available; the network was trained from scratch, so accuracy is lower "
          "than with transfer learning.\n")
    w("# Fan Anomaly Detection using MobileNetV2 and PCA (EchoLab)\n")
    w(f"*Generated {date.today():%d %B %Y} from `models/echolab/metrics.json` (trained {meta['created']}, "
      f"data source: **{meta['data_source']}**, machine {meta['machine']} id_{meta['ids']}).*\n")

    o = fin["operating"]["fused"]
    w("## Abstract\n")
    w("Factory and lab fans run all day, and faults such as worn bearings or blocked air are usually noticed only after "
      "overheating or a breakdown. This work detects faulty fans by listening to them. Ten-second recordings "
      "are converted to 128x313 log-Mel spectrograms; an ImageNet-pretrained MobileNetV2 is fine-tuned using healthy "
      "recordings only (faulty clips are held out for testing), exposed to injected fault patterns and the augmentations of "
      "time-stretch, pitch-shift and masking. Two PCA models fitted on healthy sound (on the spectrogram and on the "
      "network embedding) provide complementary reconstruction-error scores; the three scores are fused with fixed "
      f"equal weights. Evaluated on {data['n_test_healthy']} healthy and {data['n_test_faulty']} faulty test clips that "
      f"were never used for training, the fused score reaches AUC {pc(fin['scores']['fused']['auc'])} "
      f"(pAUC {pc(fin['scores']['fused']['pauc'])}); at the ROC-best, cross-fitted threshold it attains accuracy "
      f"{pc(o['roc_crossfit']['accuracy'])}, balanced accuracy {pc(o['roc_crossfit']['balanced_accuracy'])} and F1 "
      f"{f3(o['roc_crossfit']['f1'])}. The quantised model is {mod['tflite_quant_mb']:.1f} MB and runs in "
      f"{mod['host_latency_ms']['median_ms']:.0f} ms per clip on the training computer"
      + (". **(Quick run on a subset: not the final result.)**" if syn else ".") + "\n")

    w("## 1. Problem statement\n")
    w("1. Factory and lab fans run all day; faults like worn bearings or blocked air go unnoticed.\n"
      "2. Today someone checks by hand or waits until it breaks, leading to overheating, downtime and cost "
      "(contact vibration sensors are around Rs. 15k each).\n"
      "3. Challenge: only recordings of healthy fans are available, so the model must learn what normal sounds like.\n"
      "4. Data: MIMII Fan id_00, healthy clips to train and faulty clips to test (10 s each).\n"
      "5. Goal: catch a faulty sound automatically, before the fan fails.\n")
    w("## 2. Motivation and research gap\n")
    w("Monitoring protects equipment and money; faults of a fan produce high-pitched rattle (about 4-8 kHz) that is "
      "clearly visible in a spectrogram; a microphone costs about Rs. 150 and can listen continuously. "
      "The baseline listens to only 0.3 s of sound with a small autoencoder trained from scratch, and offers no "
      "threshold tuning, explanation or Raspberry Pi version. EchoLab uses the full 10 s, a pretrained image network, "
      "tuned thresholds, Grad-CAM explanation and a quantised edge model.\n")

    w("## 3. Literature (as listed in the project deck)\n")
    w(table(["Paper", "Idea we use", "AUC in deck"],
            [["Koizumi 2020", "Baseline autoencoder on 0.3 s; defines the task and scoring", "72.89 %"],
             ["Giri 2020", "MobileNetV2 classifies machine IDs as a trick; MobileNet beats autoencoders", "88 %"],
             ["Ribeiro 2021", "MFCC + CNN with sound augmentation (augmentation idea)", "85 %"],
             ["Lopez 2022", "Shrinks MobileNet to 5 MB for a Raspberry Pi (compression idea)", "82 %"],
             ["Kumar 2024", "Survey: id_00 is the cleanest fan; suggests Grad-CAM", "-"]]))
    w("\n*These figures are quoted from the deck; verify them against the original papers and the DCASE 2020 Task 2 "
      "results table (per machine ID) before submission.*\n")

    w("## 4. Dataset and preprocessing\n")
    w(f"MIMII fan **id_00** (DCASE 2020 Task 2 development set): {data['n_train_healthy']} healthy training clips; "
      f"test: {data['n_test_healthy']} healthy + {data['n_test_faulty']} faulty clips (10 s each, 16 kHz). "
      "id_02, id_04 and id_06 are noisier and are not used. Each clip becomes a 128 x 313 log-Mel image "
      "(frame 1024, hop 512, 128 Slaney-mel bands, dB referenced to the clip maximum, floor -80 dB). The same NumPy "
      "code is used in training and in the website (verified against librosa to < 1e-4 dB).\n")
    w("**Making more data:** time-stretch (0.9-1.1x), pitch-shift (+/-3 Mel bands) and time/frequency masking "
      "(SpecAugment [6]) applied to the spectrogram image.\n")

    w("## 5. Method\n")
    w("**Pipeline:** 10 s sound clip -> resample 16 kHz -> spectrogram (sound image) -> MobileNetV2 (pretrained) -> "
      "fault score -> tuned threshold -> dashboard (green / red).\n")
    w("**Healthy-only learning with pseudo-faults.** Because no faulty clip may be used for training, every training batch "
      "contains 50 % healthy clips and 50 % *pseudo-faulty* clips created from healthy ones by injecting fault "
      "patterns in the power domain (4-8 kHz rattle bursts, broadband clicks, tonal lines, amplitude modulation, raised "
      "noise floor). The network learns to output a high fault score for unusual sound. Real faults are never seen "
      "during training; how well the pseudo-faults transfer to real faults is exactly what the test measures.\n")
    w("**PCA [13, 14].** (i) Spectrogram PCA: 5-frame context vectors (640-d) of healthy clips; "
      f"{mod['spca_components']} components ({pc(mod['spca_variance'])} variance); score = mean reconstruction error. "
      f"(ii) Embedding PCA: 1280-d MobileNetV2 embeddings of healthy clips; {mod['epca_components']} components "
      f"({pc(mod['epca_variance'])} variance); score = reconstruction error. Both need no faulty clip.\n")
    w("**Fusion.** Each score is converted to a z-score against the out-of-fold healthy distribution (median / MAD); "
      "the fused score is their equal-weight mean. The weights were fixed before any test data was seen and never tuned.\n")
    w("**Thresholds.** (a) *Label-free*: 95th percentile of out-of-fold healthy scores (about 5 % false alarms; needs no "
      "faulty clip). (b) *ROC-best*: Youden-optimal threshold, cross-fitted: chosen on 4/5 of the test clips and applied "
      "to the remaining 1/5, repeated over 5 stratified folds, so no clip is judged by a threshold tuned on itself.\n")
    w("**Edge model.** Dynamic-range quantised TensorFlow Lite model with two outputs (fault score and embedding); "
      "Grad-CAM [15] uses the Keras model.\n")

    w("## 6. Evaluation protocol: stratified 5-fold cross-validation, bias and variance\n")
    w(f"The healthy clips are split into {data['folds']} folds ({data['fold_train']} train / {data['fold_val']} validation "
      "per fold); every fold keeps the same 1:1 healthy : pseudo-fault mix. The validation AUC (healthy vs "
      "pseudo-fault) is a *proxy* for model selection and diagnosis. Each fold model is also scored on the untouched "
      "real test clips, giving the average AUC and its spread. A final model is then trained on all healthy clips.\n")
    w("*Why 5-fold:* leave-one-out would need about 1000 separate trainings (too slow) and its training sets overlap almost "
      "completely (high-variance estimate [8, 9]); a 2-fold split trains on only half the data (pessimistic bias); "
      "time-series CV is unnecessary because clips are independent; a simple K-fold without keeping the class mix can "
      "distort the balance. Bias = error from a model that is too simple (under-fitting); variance = sensitivity to the "
      "particular training clips (over-fitting). We diagnose them with the validation level, the train-validation gap "
      "and the spread across folds.\n")

    w("## 7. Results\n")
    w("### 7.1 Cross-validation\n")
    w(table(["Fold", "train", "val", "proxy val AUC (CNN)", "proxy train AUC", "real-test AUC: CNN", "PCA-spec", "PCA-emb", "fused"],
            [[r["fold"], r["n_train"], r["n_val"], f3(r["proxy_val_auc"]["cnn"]), f3(r["proxy_train_auc_cnn"]),
              pc(r["test_auc"]["cnn"]), pc(r["test_auc"]["spca"]), pc(r["test_auc"]["epca"]), pc(r["test_auc"]["fused"])]
             for r in cv["folds"]]))
    w("\nAverage test AUC +/- spread over folds: " + "; ".join(
        f"{LABEL[n]} **{pc(cv['test_auc_mean'][n])} +/- {pc(cv['test_auc_std'][n])}**" for n in SCORES) + ". "
      f"Proxy validation AUC {f3(cv['proxy_val_auc_mean'])} +/- {f3(cv['proxy_val_auc_std'])}; train-validation gap "
      f"{cv['gap_train_minus_val']:+.3f} (" + ("large: over-fitting" if cv["gap_train_minus_val"] > 0.05 else "small: limited over-fitting") + ").\n")
    w("![Training curves](../models/echolab/figures/cv_curves.png)\n")

    w("### 7.2 Final model on the unseen test clips\n")
    w(table(["Score", "AUC", "pAUC (FPR <= 0.1)", "AP"],
            [[LABEL[n], pc(fin["scores"][n]["auc"]), pc(fin["scores"][n]["pauc"]), pc(fin["scores"][n]["ap"])] for n in SCORES]))
    w(f"\nDeck baseline (autoencoder): AUC {R['baseline']['auc']:.2f} %, pAUC {R['baseline']['pauc']:.0f} % *(as stated in the deck)*.\n")
    w("![ROC](../models/echolab/figures/roc.png)\n")
    w("![Baseline comparison](../models/echolab/figures/baseline_compare.png)\n")

    w("### 7.3 Accuracy, F1 and the operating point\n")
    rows = []
    for n in SCORES:
        for key, nm in (("label_free", "label-free (95th pct healthy)"), ("roc_crossfit", "ROC-best, cross-fitted")):
            m = fin["operating"][n][key]
            rows.append([LABEL[n], nm, f"{m['threshold']:.2f}", pc(m["accuracy"]), pc(m["balanced_accuracy"]), pc(m["precision"]),
                         pc(m["recall"]), pc(m["specificity"]), f3(m["f1"])])
    w(table(["Score", "Threshold rule", "z-thr", "Accuracy", "Balanced acc.", "Precision", "Recall", "Specificity", "F1"], rows))
    w(f"\n**Reading accuracy correctly:** the test set has {data['n_test_faulty']} faulty and {data['n_test_healthy']} healthy "
      f"clips, so a system that says \"faulty\" for every clip already scores {pc(fin['trivial_accuracy_predict_all_faulty'])} accuracy. "
      "Balanced accuracy (the mean of recall and specificity), AUC and F1 are therefore the figures to quote.\n")
    w("![Confusion matrices](../models/echolab/figures/confusion.png)\n")
    w("![Score distributions](../models/echolab/figures/scores.png)\n")

    w("### 7.4 PCA analysis and explanation\n")
    w("![PCA lab](../models/echolab/figures/pca_lab.png)\n")
    w("The embedding PCA map shows where healthy and faulty test clips lie relative to the healthy training sound. "
      "Grad-CAM shows where the network listens; the dotted line marks 4 kHz.\n")
    w("![Grad-CAM](../models/echolab/figures/gradcam.png)\n")
    w("*Caution:* the network was trained with injected fault patterns, many of which are placed at 4-8 kHz, so attention in "
      "that band is expected and is not independent evidence about real faults.\n")

    w("### 7.5 Deck targets versus measurements\n")
    fa, fp = fin["scores"]["fused"]["auc"], fin["scores"]["fused"]["pauc"]
    f1 = o["roc_crossfit"]["f1"]
    ms = mod["host_latency_ms"]["median_ms"]
    chk = lambda c: "MET" if c else "NOT MET"
    w(table(["Deck claim", "Target", "Measured", "Verdict"],
            [["AUC (fused)", "90-92 %", pc(fa), chk(fa >= 0.90)],
             ["pAUC (fused)", "about 78 %", pc(fp), chk(fp >= 0.78)],
             ["F1 (fused, ROC-best)", "0.87", f3(f1), chk(f1 >= 0.87)],
             ["Better than the autoencoder baseline", "> 72.89 %", pc(fa), chk(fa * 100 > R["baseline"]["auc"])],
             ["Model size, quantised", "about 4 MB", f"{mod['tflite_quant_mb']:.1f} MB (fp32 TFLite {mod['tflite_fp32_mb']:.1f} MB, Keras file {mod['keras_mb']:.1f} MB)",
              chk(mod["tflite_quant_mb"] <= 4.5)],
             ["Speed on a laptop", "about 25 ms", f"{ms:.0f} ms (this computer; model only)", chk(ms <= 25)],
             ["Speed on Raspberry Pi 4", "about 90 ms", "not measured on a Pi", "NOT MEASURED"]]))
    w("\nTargets are the deck's goals; they are reported as met or not met without adjustment.\n")

    w("## 8. Uniqueness and end users\n")
    w("Clean data (id_00); visible faults (bright high-pitched lines); works with little data (about 1000 healthy clips, "
      "pretrained network); lightweight (quantised model for a Raspberry Pi). End users: college labs, building HVAC "
      "fans, factory cooling fans, researchers.\n")
    w("## 9. Limitations\n")
    w("* Pseudo-faults are injected patterns, not real faults; transfer to real faults is measured, not assumed (Section 7.1: fold-model test AUC).\n"
      "* Only machine id_00 is used; other IDs and machine types (pump, valve, slider) need re-training and testing.\n"
      "* The ROC-best threshold needs some labelled faulty clips; the label-free threshold does not, at the cost of a fixed false-alarm rate.\n"
      "* Quantised-model size and speed on a Raspberry Pi must be measured on the device.\n"
      "* A different microphone, room or fan shifts the sound: use the website's own-fan calibration.\n"
      "* Alerts are a screening signal, not a diagnosis.\n")
    w("## 10. Conclusion and future work\n")
    w("EchoLab adapts a pretrained image network and PCA to fan sound, with healthy-only learning, a tuned threshold and "
      "an edge-ready model. Section 7.5 states which deck targets the measurements support. Future work (deck): test on a real "
      "lab fan with a microphone (the website accepts recordings and live microphone input), add more faults "
      "(loose belt, blockage), phone access and Telegram alerts (implemented in the website), and extend to pumps and valves "
      "(`--machine pump|valve|slider`).\n")
    w("## References\n")
    w("*(Verify bibliographic details before submission.)*\n")
    refs = [
        "Purohit, H. et al., \"MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machines Investigation and Inspection,\" DCASE Workshop, 2019.",
        "Koizumi, Y. et al., \"Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring,\" DCASE Workshop, 2020.",
        "Koizumi, Y. et al., \"ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection,\" WASPAA, 2019.",
        "Sandler, M. et al., \"MobileNetV2: Inverted Residuals and Linear Bottlenecks,\" CVPR, 2018.",
        "He, K. et al., \"Deep Residual Learning for Image Recognition,\" CVPR, 2016.",
        "Park, D. S. et al., \"SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,\" Interspeech, 2019.",
        "Jacob, B. et al., \"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,\" CVPR, 2018.",
        "Kohavi, R., \"A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,\" IJCAI, 1995.",
        "Hastie, T., Tibshirani, R., Friedman, J., *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 7).",
        "Yosinski, J. et al., \"How transferable are features in deep neural networks?\" NeurIPS, 2014.",
        "Hershey, S. et al., \"CNN Architectures for Large-Scale Audio Classification,\" ICASSP, 2017.",
        "Fawcett, T., \"An introduction to ROC analysis,\" Pattern Recognition Letters 27(8), 2006.",
        "Jolliffe, I. T., *Principal Component Analysis*, 2nd ed., Springer, 2002.",
        "Shlens, J., \"A Tutorial on Principal Component Analysis,\" arXiv:1404.1100, 2014.",
        "Selvaraju, R. R. et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,\" ICCV, 2017.",
        "Giri, R. et al., \"Self-supervised classification for detecting anomalous sounds,\" DCASE Workshop, 2020. (cited in the deck)",
    ]
    for i, r in enumerate(refs, 1):
        w(f"[{i}] {r}\n")
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text("\n".join(L))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
