"""
fanlib.py -- shared building blocks for training / evaluation.

  * data cache loading and leakage-aware splitting
  * MobileNetV2 transfer-learning model (full 10 s window or 2 s ablation window)
  * SpecAugment-style augmentation + tf.data generator
  * metrics (AUC, pAUC@0.1, F1 ...) and F1-optimal threshold search
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from features_np import N_FRAMES, N_MELS, WIN_2S, WIN_STARTS


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_cache(cache: str = "dataset/cache"):
    c = Path(cache)
    if not (c / "X.npy").is_file():
        raise FileNotFoundError("dataset/cache/X.npy missing. Run preprocess.py first.")
    X = np.load(c / "X.npy")
    y = np.load(c / "y.npy").astype(np.int64)
    meta = pd.read_csv(c / "meta.csv", dtype={"machine_id": str})
    info = json.loads((c / "info.json").read_text())
    return X, y, meta, info


def _strat_key(y, mid):
    return np.array([f"{a}_{b}" for a, b in zip(y, mid)])


def holdout_split(y, mid, test_size=0.2, seed=42):
    """Stratified (label x machine) hold-out; the test part is touched ONLY for the final report."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    key = _strat_key(y, mid)
    try:
        dev, test = train_test_split(idx, test_size=test_size, stratify=key, random_state=seed)
    except ValueError:
        dev, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    return np.sort(dev), np.sort(test)


def cv_folds(idx, y, mid, mode="stratified", k=5, seed=42):
    """List of (train_idx, val_idx) in GLOBAL index space."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    idx = np.asarray(idx)
    if mode == "group":                                  # leave-machine(s)-out
        n_groups = len(set(mid[idx]))
        g = GroupKFold(n_splits=min(k, n_groups))
        return [(idx[a], idx[b]) for a, b in g.split(idx, y[idx], groups=mid[idx])]
    key = _strat_key(y[idx], mid[idx])
    if pd.Series(key).value_counts().min() < k:          # too few per (label,machine) cell
        key = y[idx]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(idx[a], idx[b]) for a, b in skf.split(idx, key)]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(width: int = N_FRAMES, pretrained: bool = True, alpha: float = 1.0, dropout: float = 0.3):
    """Input (128, width, 1) dB-normalised to [-1,1]  ->  P(abnormal).
    The 1-channel spectrogram is replicated to 3 channels so ImageNet weights can be used."""
    import keras
    from keras import layers

    inp = keras.Input((N_MELS, width, 1), name="logmel")
    x = layers.Concatenate(axis=-1, name="to_rgb")([inp, inp, inp])
    base, got_weights = None, False
    if pretrained:
        try:
            base = keras.applications.MobileNetV2(input_shape=(N_MELS, width, 3), include_top=False,
                                                  weights="imagenet", alpha=alpha)
            got_weights = True
        except Exception as e:                       # offline -> no cached ImageNet weights
            print(f"[warn] could not load ImageNet weights ({type(e).__name__}); training from scratch.")
    if base is None:
        base = keras.applications.MobileNetV2(input_shape=(N_MELS, width, 3), include_top=False,
                                              weights=None, alpha=alpha)
    if not got_weights:
        # keras.applications uses BN momentum 0.999 (moving stats need ~1000s of steps to converge);
        # a from-scratch run on a small dataset needs faster-tracking statistics.
        for l in base.layers:
            if isinstance(l, keras.layers.BatchNormalization):
                l.momentum = 0.9
    # Pretrained backbone: BatchNorm kept in inference mode (standard fine-tuning practice).
    # Random-init backbone: BatchNorm must be allowed to train, otherwise activations collapse.
    x = base(x, training=False) if got_weights else base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_abnormal")(x)
    return keras.Model(inp, out, name="fan_mobilenetv2"), base, got_weights


def set_finetune(base, n_last: int):
    """Unfreeze the last n_last layers of the backbone, but keep BatchNorm frozen."""
    import keras
    layers = base.layers
    base.trainable = True
    for i, l in enumerate(layers):
        l.trainable = (i >= len(layers) - n_last) and not isinstance(l, keras.layers.BatchNormalization)


# --------------------------------------------------------------------------- #
# augmentation + input pipeline
# --------------------------------------------------------------------------- #
def norm(db):
    return (np.asarray(db, dtype=np.float32) + 40.0) / 40.0          # [-80,0] dB -> [-1,1]


def spec_augment(x, rng, n_freq=2, max_f=12, n_time=2):
    """SpecAugment (Park et al., 2019): frequency + time masking, fill = clip mean."""
    x = x.copy()
    fill = float(x.mean())
    for _ in range(n_freq):
        f = int(rng.integers(0, max_f + 1))
        f0 = int(rng.integers(0, N_MELS - f + 1))
        x[f0:f0 + f, :] = fill
    W = x.shape[1]
    max_t = max(2, W // 10)
    for _ in range(n_time):
        t = int(rng.integers(0, max_t + 1))
        t0 = int(rng.integers(0, W - t + 1))
        x[:, t0:t0 + t] = fill
    return x


def make_train_ds(X, y, idx, batch, window, augment, rng):
    import tensorflow as tf
    W = N_FRAMES if window == "full" else WIN_2S
    idx = np.asarray(idx)

    def gen():                                       # infinite, reshuffled every pass
        while True:
            order = rng.permutation(idx)
            for s in range(0, len(order) - batch + 1, batch):
                b = order[s:s + batch]
                xb = np.empty((batch, N_MELS, W, 1), np.float32)
                for k, i in enumerate(b):
                    x = norm(X[i])
                    if window == "2s":
                        st = int(rng.integers(0, N_FRAMES - W + 1))
                        x = x[:, st:st + W]
                    elif augment and rng.random() < 0.5:
                        x = np.roll(x, int(rng.integers(0, N_FRAMES)), axis=1)   # stationary sound -> safe
                    if augment and rng.random() < 0.8:
                        x = spec_augment(x, rng)
                    xb[k, :, :, 0] = x
                yield xb, y[b].astype(np.float32)

    sig = (tf.TensorSpec((None, N_MELS, W, 1), tf.float32), tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2), len(idx) // batch


def predict_probs(model, X, idx, window, batch=32):
    """Clip-level P(abnormal). 2 s mode averages the probabilities of 5 windows."""
    idx = np.asarray(idx)
    out = []
    for s in range(0, len(idx), batch):
        xs = norm(X[idx[s:s + batch]])               # (B,128,311)
        if window == "full":
            p = np.asarray(model(xs[..., None], training=False)).ravel()
        else:
            p = np.mean([np.asarray(model(xs[:, :, st:st + WIN_2S, None], training=False)).ravel()
                         for st in WIN_STARTS], axis=0)
        out.append(p)
    return np.concatenate(out) if out else np.array([])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auc_metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return dict(auc=float("nan"), pauc=float("nan"), ap=float("nan"))
    return dict(auc=float(roc_auc_score(y, p)),
                pauc=float(roc_auc_score(y, p, max_fpr=0.1)),     # DCASE-style pAUC, p = 0.1
                ap=float(average_precision_score(y, p)))


def metrics_at(y, p, thr):
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(thr=float(thr), tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                accuracy=(tp + tn) / max(len(y), 1), fpr=fp / (fp + tn) if fp + tn else 0.0)


def f1_curve(y, p, grid=None):
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    f1 = np.array([metrics_at(y, p, t)["f1"] for t in grid])
    return grid, f1


def best_threshold(y, p):
    """F1-maximising threshold; if several tie (plateau) take the plateau's middle -> more robust."""
    grid, f1 = f1_curve(y, p)
    best = np.where(f1 >= f1.max() - 1e-9)[0]
    return float(grid[best[len(best) // 2]]), float(f1.max())


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


# --------------------------------------------------------------------------- #
# EchoLab (deck protocol): healthy-only training with injected-fault exposure
# --------------------------------------------------------------------------- #
FRAME_S = 512 / 16000          # seconds per spectrogram frame


def time_stretch(x, rate, rng):
    """Spectrogram-domain time stretch (deck: 'stretch time'). x: (128, T) normalised. rate>1 = faster."""
    T = x.shape[1]
    Tn = max(8, int(round(T / rate)))
    src = np.linspace(0, T - 1, Tn)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).astype(np.float32)
    y = x[:, i0] * (1 - w) + x[:, i1] * w
    if Tn >= T:                                           # slower -> random crop
        s = int(rng.integers(0, Tn - T + 1))
        return y[:, s:s + T]
    return np.pad(y, ((0, 0), (0, T - Tn)), mode="edge")  # faster -> pad with last frame


def pitch_shift(x, k):
    """Spectrogram-domain pitch shift (deck: 'shift pitch'): move all Mel bands up/down by k bins."""
    if k == 0:
        return x
    y = np.empty_like(x)
    if k > 0:
        y[k:] = x[:-k]; y[:k] = x[:1]
    else:
        y[:k] = x[-k:]; y[k:] = x[-1:]
    return y


def deck_augment(x, rng):
    """The three augmentations of the deck, applied with independent probabilities."""
    if rng.random() < 0.5:
        x = time_stretch(x, float(rng.uniform(0.9, 1.1)), rng)
    if rng.random() < 0.5:
        x = pitch_shift(x, int(rng.integers(-3, 4)))
    if rng.random() < 0.8:
        x = spec_augment(x, rng)                          # 'mask parts of the image'
    return x


FAULT_KINDS = ["rattle", "clicks", "tone", "am", "noise_floor"]


def inject_pseudo_fault(db, rng, kinds=None):
    """Create a pseudo-faulty version of a HEALTHY log-Mel (dB, max = 0 dB, shape (128, T)).
    Faults are added in the POWER domain relative to each Mel band's own healthy level (median over
    time), so every fault is visible whatever the clip's spectral shape; the clip is then re-referenced.
      rattle      high-pitched burst noise in Mel bins ~94-128 (about 4-8 kHz, slide 3 of the deck)
      clicks      broadband impulses (loose/worn parts)
      tone        a narrow tonal line, optionally modulated (bearing tone)
      am          amplitude modulation (blade imbalance)
      noise_floor raised broadband noise (contamination / clogging)
    These are INJECTED PATTERNS added to real healthy recordings. They teach the network 'what unusual sound
    looks like'; they are not real faults."""
    P = 10.0 ** (db.astype(np.float64) / 10.0)
    M, T = P.shape
    ref = np.maximum(np.median(P, axis=1, keepdims=True), 1e-10)      # healthy level per band (M, 1)
    t = np.arange(T) * FRAME_S
    add = np.zeros_like(P)
    mult = np.ones(T)
    db_up = lambda lo, hi: 10.0 ** (rng.uniform(lo, hi) / 10.0)         # a factor, given in dB
    kinds = kinds or list(rng.choice(FAULT_KINDS, size=int(rng.integers(1, 3)), replace=False))
    for k in kinds:
        if k == "rattle":
            b0 = int(rng.integers(94, 108)); b1 = int(rng.integers(b0 + 8, M + 1))
            gate = np.sin(2 * np.pi * rng.uniform(0.3, 4.0) * t + rng.uniform(0, 6.3)) > rng.uniform(-0.7, 0.4)
            add[b0:b1, :] += ref[b0:b1] * db_up(4, 18) * gate[None, :] * rng.uniform(0.6, 1.4, (b1 - b0, T))
        elif k == "clicks":
            for c in rng.integers(0, max(T - 2, 1), size=int(rng.integers(3, 25))):
                add[:, c:c + 2] += ref * db_up(8, 24)
        elif k == "tone":
            b = int(rng.integers(8, 112))
            mod = 1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 3.0) * t)
            add[b - 1:b + 2, :] += ref[b - 1:b + 2] * db_up(8, 25) * mod[None, :]
        elif k == "am":
            mult = mult * (1.0 + rng.uniform(0.4, 0.95) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t + rng.uniform(0, 6.3)))
        elif k == "noise_floor":
            add += ref * rng.uniform(0.5, 4.0) * rng.uniform(0.6, 1.4, (M, T))
    P2 = np.maximum((P + add) * mult[None, :], 1e-12)
    out = 10.0 * np.log10(P2)
    out -= out.max()
    return np.maximum(out, -80.0).astype(np.float32)


def make_pf_sequence(X, idx, batch, seed, augment=True, workers=2):
    """Keras data sequence (no tf.data generator, so no py_func memory growth).
    Each batch = 50 % healthy (label 0) + 50 % pseudo-faulty (label 1) clips, all built from HEALTHY clips,
    with the deck's augmentations.  Every batch has its own RNG, so results are reproducible and thread-safe."""
    import keras
    idx = np.asarray(idx)
    half = max(1, batch // 2)

    class PseudoFaultSequence(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=8)
            self.epoch = 0
            self.order = np.random.default_rng([seed, 0]).permutation(idx)

        def __len__(self):
            return max(1, len(idx) // half)

        def on_epoch_end(self):
            self.epoch += 1
            self.order = np.random.default_rng([seed, self.epoch]).permutation(idx)

        def __getitem__(self, i):
            rng = np.random.default_rng([seed, self.epoch, i])
            b = self.order[i * half:(i + 1) * half]
            if len(b) < half:                                   # last, short batch: pad with random clips
                b = np.concatenate([b, rng.choice(idx, size=half - len(b))])
            xb = np.empty((2 * half, N_MELS, N_FRAMES, 1), np.float32)
            yb = np.zeros(2 * half, np.float32)
            yb[half:] = 1.0
            for k, j in enumerate(b):
                db = X[j].astype(np.float32)
                h, f = norm(db), norm(inject_pseudo_fault(db, rng))
                if augment:
                    h, f = deck_augment(h, rng), deck_augment(f, rng)
                xb[k, :, :, 0] = h
                xb[half + k, :, :, 0] = f
            return xb, yb

    return PseudoFaultSequence()


def predict_arrays(model, A, bs=32):
    """A: (N, 128, T) already normalised -> (N,) probabilities."""
    return np.concatenate([np.asarray(model(A[s:s + bs, :, :, None], training=False)).ravel()
                           for s in range(0, len(A), bs)]) if len(A) else np.array([])


def embed(emb_model, X, idx, bs=32):
    """GAP embeddings (N, 1280) of real clips X[idx]."""
    idx = np.asarray(idx)
    return np.concatenate([np.asarray(emb_model(norm(X[idx[s:s + bs]])[..., None], training=False))
                           for s in range(0, len(idx), bs)])


def find_gap(model):
    import keras
    return next(l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))


def gradcam(model, x):
    """Grad-CAM (Selvaraju et al.): where the network 'listens'. x: (1,128,T,1) normalised.
    Returns heat-map (128, T) in [0,1]: rows = Mel bands (low -> high), columns = time."""
    import keras
    import tensorflow as tf
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    dense = model.get_layer("p_abnormal")
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        fmap = base(tf.concat([x, x, x], axis=-1), training=False)
        tape.watch(fmap)
        out = dense(tf.reduce_mean(fmap, axis=(1, 2)))
    g = tape.gradient(out, fmap)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-9)
    cam = tf.image.resize(cam[None, :, :, None], (x.shape[1], x.shape[2]))[0, :, :, 0].numpy()
    return np.clip(cam, 0, 1)
