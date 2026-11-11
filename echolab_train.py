#!/usr/bin/env python3
"""
echolab_train.py -- EchoLab: the training pipeline of the project deck, end to end.

    python echolab_train.py            # download MIMII fan id_00 -> train -> evaluate -> export -> report
    python echolab_train.py --archive dev_data_fan.zip     # zip you downloaded yourself
    python echolab_train.py --data-dir D:/MIMII/fan        # folder you already have
    python echolab_train.py --quick                        # fast check on a small subset (NOT the final model)

PROTOCOL (what the deck says, made precise)
  data      DCASE-2020 / MIMII fan id_00: ~1003 HEALTHY training clips, test = 100 healthy + ~341 faulty.
            Faulty clips are NEVER used for training, so the model must learn what "normal" sounds like.
  features  16 kHz, 10 s -> 128 x 313 log-Mel "sound image".
  model     ImageNet-pretrained MobileNetV2 on the full 10 s image (transfer learning).
  learning  Each batch = 50 % healthy + 50 % PSEUDO-FAULTY clips made from healthy ones (injected rattle at
            4-8 kHz, clicks, tones, modulation, noise), with the deck's augmentations (time stretch, pitch shift,
            masking).  The network outputs a "fault score".
  CV        Stratified 5-fold over the healthy clips (802 train / 201 validation per fold; every fold keeps the
            same 1:1 healthy : pseudo-fault mix).  Every fold model is ALSO scored on the untouched real test
            clips -> "average AUC +/- spread".
  PCA       (1) spectrogram PCA and (2) embedding PCA, both fitted on healthy sound only (reconstruction error),
            fused with the network score.  The fusion is fixed in advance (equal weights, no tuning on test data).
  threshold (a) label-free: 95th percentile of out-of-fold HEALTHY scores (needs no faulty clip);
            (b) ROC-best: Youden-optimal, cross-fitted on the test clips (threshold chosen on 4/5 of the test
            clips and applied to the remaining 1/5, so it is never judged on data it was tuned on).
  export    Keras model + quantised TFLite (outputs fault score AND embedding) + PCA files + metrics.json.

Everything the web app displays (accuracy, F1, AUC, ROC ...) is read from models/echolab/metrics.json, which is
written only by this script.  The project uses REAL recordings only; a --quick run is labelled as such everywhere.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import fanlib as fl
from analyze import PCADetector, stack_frames
from echolab_core import SCORE_LABEL, SCORES, fit_norm, raw_to_t, to_z
from features_np import N_FRAMES, N_MELS

OUT = Path("models/echolab")
DECK_BASELINE = dict(name="Koizumi 2020 autoencoder baseline", auc=72.89, pauc=60.0,
                     note="figures as stated in the project deck; verify against the DCASE 2020 Task 2 results table")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def ensure_data(args):
    """Make sure dataset/cache matches the request; download + preprocess if needed (real recordings only)."""
    cache = Path(args.cache)
    want_ids = sorted({i.strip().zfill(2) for i in args.ids.split(",")}) if args.ids else None
    info_p = cache / "info.json"
    ok = False
    if info_p.is_file() and not args.refresh:
        info = json.loads(info_p.read_text())
        if info.get("source") == "synthetic":              # cache left by an older version of the project
            log("Removing a leftover cache of ARTIFICIAL sound from an older version (real data only now).")
            shutil.rmtree(cache, ignore_errors=True)
        else:
            ok = info.get("machine") == args.machine and info.get("ids") == want_ids
    if not ok:
        log("=" * 70, "\nSTEP 1/5  dataset\n" + "=" * 70)
        cmd = [sys.executable, "download_data.py", "--machine", args.machine]
        if args.archive:
            cmd += ["--archive", args.archive]
        if args.data_dir:
            cmd += ["--data-dir", args.data_dir]
        if args.snr:
            cmd += ["--snr", str(args.snr)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            log("\nStopped: the real dataset could not be obtained (see the message above).\n"
                "If you downloaded dev_data_fan.zip yourself:  python echolab_train.py --archive path/to/dev_data_fan.zip")
            sys.exit(2)
        pre = [sys.executable, "preprocess.py", "--machine", args.machine]
        if args.ids:
            pre += ["--ids", args.ids]
        if subprocess.run(pre).returncode != 0:
            log("\nStopped: preprocessing failed.")
            sys.exit(2)
    X, y, meta, info = fl.load_cache(args.cache)
    return X, y, meta, info


def split_data(X, y, meta, args):
    split = meta["split"].values
    healthy = np.where((split == "train") & (y == 0))[0]
    test = np.where(split == "test")[0]
    rng = np.random.default_rng(args.seed)
    if args.quick:
        healthy = np.sort(rng.choice(healthy, size=min(96, len(healthy)), replace=False))
        tn = rng.choice(test[y[test] == 0], size=min(24, int((y[test] == 0).sum())), replace=False)
        ta = rng.choice(test[y[test] == 1], size=min(48, int((y[test] == 1).sum())), replace=False)
        test = np.sort(np.concatenate([tn, ta]))
    if len(healthy) < 20 or len(set(y[test].tolist())) < 2:
        log("Not enough data: need healthy training clips and a test set with both healthy and faulty clips.")
        sys.exit(2)
    return healthy, test


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def op_metrics(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return dict(accuracy=(tp + tn) / max(len(y), 1), balanced_accuracy=(sens + spec) / 2, precision=prec,
                recall=sens, specificity=spec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def youden_thr(y, s):
    from sklearn.metrics import roc_curve
    f, t, th = roc_curve(y, s)
    j = t - f
    j[~np.isfinite(th)] = -1
    return float(th[int(np.argmax(j))])


def crossfit_roc(y, s, k=5, seed=0):
    """Threshold chosen on k-1 folds of the test clips, applied to the held-out fold (stratified)."""
    from sklearn.model_selection import StratifiedKFold
    pred = np.zeros(len(y), int)
    thrs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(s, y):
        thr = youden_thr(y[tr], s[tr])
        thrs.append(thr)
        pred[te] = (s[te] >= thr).astype(int)
    return pred, thrs


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def build_val_arrays(X, va_idx, seed):
    vr = np.random.default_rng(seed + 7)
    Hdb = [X[i].astype(np.float32) for i in va_idx]
    Fdb = [fl.inject_pseudo_fault(d, vr) for d in Hdb]
    return Hdb, Fdb


def train_one(X, tr_idx, va_arrays, args, rng, seed, label):
    import keras
    from sklearn.metrics import roc_auc_score

    keras.utils.set_random_seed(seed)
    model, base, got = fl.build_model(N_FRAMES, pretrained=not args.no_pretrained, alpha=args.alpha, dropout=args.dropout)
    ds = fl.make_pf_sequence(X, tr_idx, args.batch, seed)
    hist = []
    if va_arrays is not None:
        Vh = np.stack([fl.norm(d) for d in va_arrays[0]])
        Vf = np.stack([fl.norm(d) for d in va_arrays[1]])
        yv = np.r_[np.zeros(len(Vh)), np.ones(len(Vf))]

    class CB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            row = dict(epoch=len(hist), loss=float((logs or {}).get("loss", np.nan)),
                       train_auc_aug=float((logs or {}).get("auc", np.nan)), val_auc=float("nan"))
            if va_arrays is not None and args.eval_every and (len(hist) + 1) % args.eval_every == 0:
                p = np.r_[fl.predict_arrays(self.model, Vh), fl.predict_arrays(self.model, Vf)]
                row["val_auc"] = float(roc_auc_score(yv, p))
            hist.append(row)

    t0 = time.time()
    auc = lambda: keras.metrics.AUC(name="auc")
    if got:
        base.trainable = False
        model.compile(keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[auc()])
        model.fit(ds, epochs=args.epochs_head, verbose=2, shuffle=False, callbacks=[CB()])
        fl.set_finetune(base, args.unfreeze_last)
        lr = keras.optimizers.schedules.CosineDecay(1e-4, decay_steps=max(1, args.epochs_ft * len(ds)), alpha=0.05)
        epochs = args.epochs_ft
    else:
        log("  [warn] training the whole network FROM SCRATCH (no ImageNet weights). "
            "Accuracy will be lower than with transfer learning; this is stated in the saved metrics.")
        base.trainable = True
        epochs = args.epochs_head + args.epochs_ft
        lr = keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=max(1, epochs * len(ds)), alpha=0.02)
    model.compile(keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=[auc()])
    model.fit(ds, epochs=epochs, verbose=2, shuffle=False, callbacks=[CB()])
    log(f"  [{label}] trained in {time.time()-t0:.0f}s  ({'ImageNet transfer learning' if got else 'from scratch'})")
    del ds
    gc.collect()
    return model, hist, got


def fit_pcas(model, X, tr_idx, args, rng):
    import keras
    emb_model = keras.Model(model.input, fl.find_gap(model).output)
    frames = []
    for i in tr_idx:
        f = stack_frames(X[i].astype(np.float32))
        frames.append(f[rng.choice(len(f), size=min(args.frames_per_clip, len(f)), replace=False)])
    spca = PCADetector(args.pca_var, args.pca_max_k).fit(np.concatenate(frames))
    epca = PCADetector(args.pca_var, args.pca_max_k_emb).fit(fl.embed(emb_model, X, tr_idx))
    return emb_model, spca, epca


def score_clips(model, emb_model, spca, epca, X, idx):
    return dict(cnn=fl.predict_probs(model, X, idx, "full"),
                spca=np.array([spca.score_db(X[i].astype(np.float32)) for i in idx]),
                epca=epca.frame_errors(fl.embed(emb_model, X, idx)))


def score_arrays_db(model, emb_model, spca, epca, dbs):
    A = np.stack([fl.norm(d) for d in dbs])
    E = np.concatenate([np.asarray(emb_model(A[s:s + 32, :, :, None], training=False)) for s in range(0, len(A), 32)])
    return dict(cnn=fl.predict_arrays(model, A), spca=np.array([spca.score_db(d) for d in dbs]),
                epca=epca.frame_errors(E))


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_all(model, X, healthy, test, args):
    import keras
    import tensorflow as tf
    from tflite_runner import TFLiteModel

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(OUT / "model.keras")
    exp = keras.Model(model.input, [model.output, fl.find_gap(model).output])
    gc.collect()
    blob32 = tf.lite.TFLiteConverter.from_keras_model(exp).convert()
    n32 = len(blob32)
    del blob32                                       # only the size is needed; free memory before the next conversion
    gc.collect()
    conv = tf.lite.TFLiteConverter.from_keras_model(exp)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    blobq = conv.convert()
    (OUT / "model.tflite").write_bytes(blobq)
    # agreement between Keras and the quantised TFLite model on test clips
    tm = TFLiteModel(OUT / "model.tflite", 1)
    sel = test[: min(40, len(test))]
    pk = fl.predict_probs(model, X, sel, "full")
    pt = np.array([tm.predict(fl.norm(X[i])[None, :, :, None]) for i in sel])
    lat = TFLiteModel(OUT / "model.tflite", 1).benchmark(runs=20)
    return dict(keras_mb=(OUT / "model.keras").stat().st_size / 1e6, tflite_fp32_mb=n32 / 1e6,
                tflite_quant_mb=len(blobq) / 1e6, host_latency_ms=lat,
                tflite_vs_keras_max_prob_diff=float(np.abs(pk - pt).max()),
                tflite_vs_keras_decision_agreement=float(((pk >= 0.5) == (pt >= 0.5)).mean()))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(R, X, y, test, model, hist_all, z_test, thr, viz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fd = OUT / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    yt = y[test]
    cols = dict(cnn="#2563eb", spca="#14b8a6", epca="#a855f7", fused="#f59e0b")

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for n in SCORES:
        f, t, _ = roc_curve(yt, z_test[n])
        ax.plot(f, t, color=cols[n], lw=2.6 if n == "fused" else 1.6,
                label=f"{SCORE_LABEL[n]}  (AUC {100*R['final']['scores'][n]['auc']:.1f} %)")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.axvline(0.1, color="gray", lw=0.6, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves on the unseen test clips")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "roc.png", dpi=170); plt.close(fig)

    # confusion matrices (fused, both operating points)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, key, title in ((axs[0], "label_free", "label-free threshold"), (axs[1], "roc_crossfit", "ROC-best (cross-fitted)")):
        m = R["final"]["operating"]["fused"][key]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["healthy", "faulty"], yticklabels=["healthy", "faulty"],
               xlabel="Predicted", ylabel="True", title=f"{title}\nacc {100*m['accuracy']:.1f} %  F1 {m['f1']:.3f}")
    fig.tight_layout(); fig.savefig(fd / "confusion.png", dpi=170); plt.close(fig)

    # fused-score histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    lo, hi = np.percentile(z_test["fused"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 45)
    ax.hist(z_test["fused"][yt == 0], bins=bins, alpha=0.65, color="#16a34a", label="healthy test clips")
    ax.hist(z_test["fused"][yt == 1], bins=bins, alpha=0.65, color="#dc2626", label="faulty test clips")
    ax.axvline(thr["fused"]["label_free"], color="k", ls=":", label="label-free threshold")
    ax.axvline(thr["fused"]["roc"], color="k", ls="--", label="ROC-best threshold")
    ax.set(xlabel="EchoLab fused fault score (z-units of healthy sound)", ylabel="clips", title="Score distributions")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "scores.png", dpi=170); plt.close(fig)

    # training curves (bias / variance view)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for k, h in enumerate(hist_all):
        ax[0].plot([r["loss"] for r in h], alpha=0.8, label=f"fold {k+1}")
        v = [(r["epoch"], r["val_auc"]) for r in h if not np.isnan(r["val_auc"])]
        if v:
            ax[1].plot(*zip(*v), marker="o", alpha=0.8, label=f"fold {k+1}")
    ax[0].set(xlabel="epoch", ylabel="training loss", title="Training loss per fold")
    ax[1].set(xlabel="epoch", ylabel="validation AUC (healthy vs pseudo-fault)", title="Validation AUC per fold")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fd / "cv_curves.png", dpi=170); plt.close(fig)

    # PCA lab
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    rat = viz["ratio"]
    ax[0].plot(range(1, len(rat) + 1), rat, "k-")
    ax[0].axhline(0.9, color="gray", ls=":")
    ax[0].set(xlabel="principal component", ylabel="cumulative explained variance", title="Embedding PCA: scree curve")
    tp, tyv = viz["test_pc"], viz["test_y"]
    ax[1].scatter(*viz["train_pc"].T, s=10, c="#16a34a", alpha=0.25, label="healthy (training)")
    ax[1].scatter(*tp[tyv == 0].T, s=18, c="#15803d", marker="^", label="healthy (test)")
    ax[1].scatter(*tp[tyv == 1].T, s=18, c="#dc2626", marker="x", label="faulty (test)")
    ax[1].set(xlabel=f"PC1 ({100*viz['var1']:.1f} %)", ylabel=f"PC2 ({100*viz['var2']:.1f} %)", title="Embedding PCA map")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fd / "pca_lab.png", dpi=170); plt.close(fig)

    # baseline comparison
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    names = ["Deck baseline\n(autoencoder)"] + [SCORE_LABEL[n].replace(" ", "\n", 1) for n in SCORES]
    vals = [DECK_BASELINE["auc"]] + [100 * R["final"]["scores"][n]["auc"] for n in SCORES]
    bars = ax.bar(names, vals, color=["#6b7280"] + [cols[n] for n in SCORES])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(90, color="#dc2626", ls="--", lw=1, label="deck target: 90 %")
    ax.set(ylim=(0, 105), ylabel="AUC (%)", title="Measured AUC vs the deck baseline and target")
    ax.legend(fontsize=8); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(fd / "baseline_compare.png", dpi=170); plt.close(fig)

    # Grad-CAM examples
    fused = z_test["fused"]
    pick = list(test[yt == 1][np.argsort(-fused[yt == 1])[:3]]) + list(test[yt == 0][np.argsort(fused[yt == 0])[:3]])
    titles = ["faulty, highest score"] * 3 + ["healthy, lowest score"] * 3
    fig, axs = plt.subplots(2, 6, figsize=(19, 5.6))
    for c, (i, ttl) in enumerate(zip(pick, titles)):
        db = X[i].astype(np.float32)
        cam = fl.gradcam(model, fl.norm(db)[None, :, :, None])
        axs[0, c].imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0)
        axs[1, c].imshow(db, origin="lower", aspect="auto", cmap="gray", vmin=-80, vmax=0)
        axs[1, c].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5, vmin=0, vmax=1)
        for r in (0, 1):
            axs[r, c].axhline(99, color="white", ls=":", lw=0.8)
        axs[0, c].set_title(ttl, fontsize=9)
        axs[1, c].set_xlabel("time frame")
        if c == 0:
            axs[0, c].set_ylabel("Mel band (dotted line = 4 kHz)"); axs[1, c].set_ylabel("Grad-CAM")
    fig.tight_layout(); fig.savefig(fd / "gradcam.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="EchoLab training (deck protocol)")
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--ids", default="00", help="machine id(s); the deck uses id_00 only")
    ap.add_argument("--cache", default="dataset/cache")
    ap.add_argument("--archive", default=None, help="manually downloaded dev_data_<machine>.zip")
    ap.add_argument("--data-dir", default=None, help="folder you already have (extracted DCASE 2020 / MIMII)")
    ap.add_argument("--snr", default=None, help="original MIMII with several noise versions: 6 (default), 0 or -6")
    ap.add_argument("--refresh", action="store_true", help="rebuild the cache")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-ft", type=int, default=10)
    ap.add_argument("--unfreeze-last", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--pca-var", type=float, default=0.90)
    ap.add_argument("--pca-max-k", type=int, default=80, help="cap for the spectrogram PCA")
    ap.add_argument("--pca-max-k-emb", type=int, default=64, help="cap for the embedding PCA")
    ap.add_argument("--frames-per-clip", type=int, default=30)
    ap.add_argument("--p-label-free", type=float, default=95.0, help="percentile of healthy scores for the label-free threshold")
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch on purpose")
    ap.add_argument("--allow-scratch", action="store_true",
                    help="continue from scratch if the ImageNet weights cannot be downloaded (lower accuracy)")
    ap.add_argument("--quick", action="store_true", help="software test: tiny subset, alpha 0.35, 3 folds")
    ap.add_argument("--fresh", action="store_true", help="ignore saved fold results and retrain everything")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.quick:
        args.folds, args.epochs_head, args.epochs_ft, args.batch, args.eval_every, args.alpha = 3, 2, 8, 16, 5, 0.35
        args.unfreeze_last = 20
    if args.no_pretrained:
        args.allow_scratch = True
    t_start = time.time()

    X, y, meta, info = ensure_data(args)
    healthy, test = split_data(X, y, meta, args)
    if not args.no_pretrained:                      # pre-flight: fail in seconds, not after hours
        import keras
        log("Checking the ImageNet weights for MobileNetV2 ...")
        _, _, have = fl.build_model(N_FRAMES, pretrained=True, alpha=args.alpha)
        keras.backend.clear_session()
        if not have and not args.allow_scratch:
            log("\nStopped: the pretrained MobileNetV2 (ImageNet) weights could not be downloaded "
                "(they come from storage.googleapis.com).\n"
                "  * Check your internet connection / firewall and run again, or\n"
                "  * run with --allow-scratch to train from scratch (lower accuracy; the website will say so).")
            sys.exit(2)
    yt = y[test]
    log("=" * 70, f"\nEchoLab  |  data source: {info['source'].upper()}  |  healthy train {len(healthy)}  |  test "
        f"{int((yt==0).sum())} healthy + {int((yt==1).sum())} faulty", "\n" + "=" * 70)
    if args.quick:
        log("!! QUICK RUN: small subset of the real data and a small network. Good for checking the setup, "
            "NOT the final model. Run without --quick for the real result.")

    # ------------------------------------------------------------------ 5-fold CV on healthy clips
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(args.seed)
    folds = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(healthy))
    oof = {n: np.full(len(healthy), np.nan) for n in ("cnn", "spca", "epca")}
    fold_rows, hist_all, fold_test_raw, got_weights = [], [], [], True
    from sklearn.metrics import roc_auc_score
    log("\nSTEP 2/5  stratified %d-fold cross-validation (healthy clips; every fold 1:1 healthy : pseudo-fault)" % args.folds)
    fp = hashlib.md5(json.dumps([args.machine, args.ids, args.folds, args.seed, args.alpha, args.epochs_head, args.epochs_ft,
                                 args.batch, args.quick, args.no_pretrained, len(healthy), len(test), info["n"], info["source"]]).encode()).hexdigest()[:10]
    work = Path("models/echolab_work") / fp          # finished folds survive a crash / Colab disconnect
    if args.fresh and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    for k, (a, b) in enumerate(folds):
        tr, va = healthy[a], healthy[b]
        saved = work / f"fold{k}.pkl"
        if saved.is_file():
            d = pickle.loads(saved.read_bytes())
            for n in oof:
                oof[n][b] = d["sh"][n]
            fold_rows.append(d["row"]); hist_all.append(d["hist"]); fold_test_raw.append(d["st"]); got_weights = got_weights and d["got"]
            log(f"\n--- fold {k+1}/{args.folds}: resumed from a previous run (use --fresh to retrain)")
            continue
        log(f"\n--- fold {k+1}/{args.folds}: train {len(tr)} healthy | validate {len(va)} healthy (+{len(va)} pseudo-faulty)")
        va_arr = build_val_arrays(X, va, args.seed + k)
        model, hist, got = train_one(X, tr, va_arr, args, rng, args.seed + k, f"fold{k+1}")
        got_weights = got_weights and got
        emb_model, spca, epca = fit_pcas(model, X, tr, args, rng)
        sh = score_clips(model, emb_model, spca, epca, X, va)
        sf = score_arrays_db(model, emb_model, spca, epca, va_arr[1])
        for n in oof:
            oof[n][b] = sh[n]
        st = score_clips(model, emb_model, spca, epca, X, test)
        fold_test_raw.append(st)
        sub = rng.choice(len(tr), size=min(150, len(tr)), replace=False)
        tr_arr = build_val_arrays(X, tr[sub], args.seed + 100 + k)
        trh = score_arrays_db(model, emb_model, spca, epca, tr_arr[0]); trf = score_arrays_db(model, emb_model, spca, epca, tr_arr[1])
        proxy_val = {n: float(roc_auc_score(np.r_[np.zeros(len(va)), np.ones(len(va))], np.r_[sh[n], sf[n]])) for n in oof}
        proxy_tr = float(roc_auc_score(np.r_[np.zeros(len(sub)), np.ones(len(sub))], np.r_[trh["cnn"], trf["cnn"]]))
        hist_all.append(hist)
        fold_rows.append(dict(fold=k + 1, n_train=int(len(tr)), n_val=int(len(va)), proxy_val_auc=proxy_val,
                              proxy_train_auc_cnn=proxy_tr, test_auc_raw={n: fl.auc_metrics(yt, st[n])["auc"] for n in oof}))
        log(f"    proxy validation AUC (healthy vs pseudo-fault): " + ", ".join(f"{n} {v:.3f}" for n, v in proxy_val.items()))
        log(f"    real test AUC of this fold's model:            " + ", ".join(f"{n} {v:.3f}" for n, v in fold_rows[-1]['test_auc_raw'].items()))
        saved.write_bytes(pickle.dumps(dict(sh=sh, row=fold_rows[-1], hist=hist, st=st, got=got)))
        import keras
        keras.backend.clear_session()

    norm = fit_norm(oof)
    z_oof = to_z(norm, oof)
    for r, st in zip(fold_rows, fold_test_raw):
        zt = to_z(norm, st)
        r["test_auc"] = {n: fl.auc_metrics(yt, zt[n])["auc"] for n in SCORES}
    cv = dict(folds=fold_rows)
    for n in SCORES:
        v = np.array([r["test_auc"][n] for r in fold_rows])
        cv.setdefault("test_auc_mean", {})[n] = float(v.mean())
        cv.setdefault("test_auc_std", {})[n] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    pv = np.array([r["proxy_val_auc"]["cnn"] for r in fold_rows]); pt = np.array([r["proxy_train_auc_cnn"] for r in fold_rows])
    cv["proxy_val_auc_mean"], cv["proxy_val_auc_std"] = float(pv.mean()), float(pv.std(ddof=1)) if len(pv) > 1 else float("nan")
    cv["proxy_train_auc_mean"], cv["gap_train_minus_val"] = float(pt.mean()), float(pt.mean() - pv.mean())

    # ------------------------------------------------------------------ final model on all healthy clips
    log("\nSTEP 3/5  final model on ALL healthy clips")
    model, hist_final, got = train_one(X, healthy, None, args, np.random.default_rng(args.seed + 999), args.seed + 999, "final")
    emb_model, spca, epca = fit_pcas(model, X, healthy, args, np.random.default_rng(args.seed + 5))
    z_test = to_z(norm, score_clips(model, emb_model, spca, epca, X, test))

    thr = {n: dict(label_free=float(np.percentile(z_oof[n], args.p_label_free))) for n in SCORES}
    final = dict(scores={}, operating={})
    for n in SCORES:
        final["scores"][n] = fl.auc_metrics(yt, z_test[n])
        pred_lf = (z_test[n] >= thr[n]["label_free"]).astype(int)
        pred_cf, thrs = crossfit_roc(yt, z_test[n], 5, args.seed)
        thr[n]["roc"] = float(np.median(thrs))
        final["operating"][n] = dict(
            label_free=dict(op_metrics(yt, pred_lf), threshold=thr[n]["label_free"],
                            how=f"{args.p_label_free:.0f}th percentile of out-of-fold healthy scores (no faulty clip used)"),
            roc_crossfit=dict(op_metrics(yt, pred_cf), threshold=thr[n]["roc"], fold_thresholds=thrs,
                              how="Youden-optimal ROC threshold, cross-fitted over 5 stratified folds of the test clips"))
    final["trivial_accuracy_predict_all_faulty"] = float((yt == 1).mean())

    # ------------------------------------------------------------------ export
    log("\nSTEP 4/5  export (Keras + quantised TFLite + PCA files)")
    exp = export_all(model, X, healthy, test, args)
    spca.save(OUT / "pca_spec.npz", thr["spca"]["label_free"], dict(data_source=info["source"]))
    epca.save(OUT / "pca_emb.npz", thr["epca"]["label_free"], dict(data_source=info["source"]))
    Eh = fl.embed(emb_model, X, healthy[: min(500, len(healthy))])
    Et = fl.embed(emb_model, X, test)
    proj = lambda E: (E - epca.mean) @ epca.components[:2].T
    ratio = epca.explained_ratio()[: min(100, len(epca.eigvals))]
    viz = dict(train_pc=proj(Eh), test_pc=proj(Et), test_y=yt, ratio=ratio,
               var1=float(epca.eigvals[0] / epca.eigvals.sum()), var2=float(epca.eigvals[1] / epca.eigvals.sum()))
    np.savez(OUT / "viz.npz", **viz)

    spec_k = int(spca.k); emb_k = int(epca.k)
    R = dict(
        meta=dict(created=datetime.now().isoformat(timespec="seconds"), data_source=info["source"], machine=args.machine,
                  ids=args.ids, pretrained_imagenet=bool(got and got_weights), quick=bool(args.quick),
                  minutes=round((time.time() - t_start) / 60, 1), args={k: v for k, v in vars(args).items()}),
        data=dict(n_train_healthy=int(len(healthy)), n_test_healthy=int((yt == 0).sum()), n_test_faulty=int((yt == 1).sum()),
                  folds=args.folds, fold_train=int(np.mean([r["n_train"] for r in fold_rows])),
                  fold_val=int(np.mean([r["n_val"] for r in fold_rows]))),
        model=dict(alpha=args.alpha, **exp, spca_components=spec_k, epca_components=emb_k,
                   spca_variance=float(spca.explained_ratio()[spec_k - 1]), epca_variance=float(epca.explained_ratio()[emb_k - 1])),
        cv=cv, final=final, thresholds=thr, norm=norm, baseline=DECK_BASELINE)
    make_figures(R, X, y, test, model, hist_all, z_test, thr, viz)

    cfg = dict(version=1, data_source=info["source"], quick=bool(args.quick), score_names=SCORES, norm=norm, thresholds=thr,
               default_score="fused", default_operating="roc", input_width=N_FRAMES,
               trained_on=dict(machine=args.machine, ids=args.ids), created=R["meta"]["created"])
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(R, indent=2, default=float))

    # ------------------------------------------------------------------ report
    log("\nSTEP 5/5  report")
    try:
        import echolab_report
        echolab_report.main()
    except Exception as e:                                            # report must never break training
        log(f"  (report skipped: {type(e).__name__}: {e})")

    o = final["operating"]["fused"]
    log("\n" + "=" * 70 + "\nRESULTS (unseen test clips: %d healthy + %d faulty)" % ((yt == 0).sum(), (yt == 1).sum()))
    for n in SCORES:
        s_ = final["scores"][n]
        log(f"  {SCORE_LABEL[n]:26s} AUC {100*s_['auc']:5.1f} %   pAUC {100*s_['pauc']:5.1f} %   "
            f"CV-fold AUC {100*cv['test_auc_mean'][n]:5.1f} +/- {100*cv['test_auc_std'][n]:.1f} %")
    for key in ("label_free", "roc_crossfit"):
        m = o[key]
        log(f"  fused @ {key:13s} accuracy {100*m['accuracy']:5.1f} %  balanced {100*m['balanced_accuracy']:5.1f} %  "
            f"F1 {m['f1']:.3f}  recall {100*m['recall']:.1f} %  specificity {100*m['specificity']:.1f} %")
    log(f"  (predicting 'faulty' for every clip would score {100*final['trivial_accuracy_predict_all_faulty']:.1f} % accuracy)")
    log(f"  model: {exp['tflite_quant_mb']:.2f} MB quantised TFLite  |  {exp['host_latency_ms']['median_ms']:.0f} ms / clip on this computer")
    if args.quick:
        log("\n!! QUICK RUN: these numbers come from a small subset. Run without --quick for the real result.")
    elif not R["meta"]["pretrained_imagenet"]:
        log("\n!! ImageNet weights were not available: trained from scratch (lower accuracy expected).")
    shutil.rmtree(work, ignore_errors=True)
    log(f"\nSaved: {OUT}/  (model.tflite, model.keras, config.json, metrics.json, figures/)   total {R['meta']['minutes']} min")
    log("Open the website:   python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
