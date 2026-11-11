#!/usr/bin/env python3
"""
echolab_core.py -- the EchoLab inference engine, shared by the website (app.py) and the command line.

    python echolab_core.py recording.wav [more.wav ...]

Loads the model bundle written by echolab_train.py (models/echolab/) and, for any audio file:
    audio -> 16 kHz mono -> 10 s windows -> 128x313 log-Mel -> quantised MobileNetV2 (fault score + embedding)
          -> spectrogram-PCA error, embedding-PCA error -> z-scores against HEALTHY sound -> fused score
          -> window flags -> HEALTHY / POSSIBLE FAULT / FAULTY.
There is NO built-in demo model: if the bundle is missing, `Bundle.exists()` is False and the app shows the
setup screen.  A bundle from a --quick run carries `quick = True` and is labelled everywhere.
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from analyze import PCADetector, load_audio, make_windows, stack_frames  # noqa: F401
from features_np import N_FRAMES, N_MELS, N_SAMPLES, SR, WIN_STARTS, logmel, to_model_input
from tflite_runner import TFLiteModel

BUNDLE = Path("models/echolab")
SCORES = ["cnn", "spca", "epca", "fused"]
SCORE_LABEL = {"cnn": "MobileNetV2 fault score", "spca": "PCA on spectrogram", "epca": "PCA on embedding",
               "fused": "EchoLab fused score"}
FILES = ["model.tflite", "config.json", "metrics.json", "pca_spec.npz", "pca_emb.npz", "viz.npz"]


# --------------------------------------------------------------------------- #
# score transforms -- the single source of truth (echolab_train.py imports these)
# --------------------------------------------------------------------------- #
def raw_to_t(name, s):
    s = np.asarray(s, dtype=np.float64)
    if name == "cnn":
        p = np.clip(s, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))                          # logit of the network's fault probability
    return np.log(np.maximum(s, 1e-12))                      # PCA reconstruction errors are positive, heavy-tailed


def fit_norm(oof_healthy: dict):
    """Robust centre/scale of each score on out-of-fold HEALTHY clips."""
    out = {}
    for n in ("cnn", "spca", "epca"):
        t = raw_to_t(n, oof_healthy[n])
        med = float(np.median(t))
        mad = float(1.4826 * np.median(np.abs(t - med)))
        out[n] = dict(mu=med, sigma=max(mad, 0.1 * float(t.std()), 1e-3))
    return out


def to_z(norm, raw: dict, shift: dict | None = None):
    """raw scores -> z-units of healthy sound; fused = equal-weight mean (fixed a priori, never tuned)."""
    shift = shift or {}
    z = {n: (raw_to_t(n, raw[n]) - norm[n]["mu"]) / norm[n]["sigma"] - shift.get(n, 0.0) for n in ("cnn", "spca", "epca")}
    z["fused"] = (z["cnn"] + z["spca"] + z["epca"]) / 3.0
    return z


# --------------------------------------------------------------------------- #
class Bundle:
    def __init__(self, path=BUNDLE):
        self.path = Path(path)
        self.cfg = json.loads((self.path / "config.json").read_text())
        self.metrics = json.loads((self.path / "metrics.json").read_text())
        self.tfl = TFLiteModel(self.path / "model.tflite", num_threads=2)
        self.spca, _, _ = PCADetector.load(self.path / "pca_spec.npz")
        self.epca, _, _ = PCADetector.load(self.path / "pca_emb.npz")
        self.viz = {k: v for k, v in np.load(self.path / "viz.npz").items()}
        self.norm, self.thr = self.cfg["norm"], self.cfg["thresholds"]
        self.quick = bool(self.cfg.get("quick"))                 # fast check on a small subset
        self.synthetic = bool(self.cfg.get("synthetic"))         # bundle from an older version of the project
        self._keras = None

    @staticmethod
    def exists(path=BUNDLE) -> bool:
        return all((Path(path) / f).is_file() for f in FILES)

    # ---- scoring
    def raw_scores(self, db):
        """db: (128,313) -> raw scores + embedding."""
        p, e = self.tfl.predict_both(to_model_input(db))
        return dict(cnn=p, spca=self.spca.score_db(db), epca=float(self.epca.frame_errors(e[None, :])[0])), e

    def project(self, emb):
        return (emb - self.epca.mean) @ self.epca.components[:2].T

    def threshold(self, score, operating):
        return float(self.thr[score]["roc" if operating == "roc" else "label_free"])

    # ---- Grad-CAM (needs TensorFlow + model.keras; silently unavailable otherwise)
    def gradcam(self, db):
        try:
            if self._keras is None:
                import keras
                self._keras = keras.saving.load_model(self.path / "model.keras")
            import fanlib as fl
            return fl.gradcam(self._keras, fl.norm(db)[None, :, :, None])
        except Exception:
            return None


# --------------------------------------------------------------------------- #
def analyze_audio(bundle: Bundle, src, name="", score=None, operating="roc", thr=None, hop_s=5.0,
                  min_rms=0.002, shift=None) -> dict:
    score = score or bundle.cfg.get("default_score", "fused")
    wav, info = load_audio(src, name)
    starts, wav, looped = make_windows(wav, hop_s)
    thr = bundle.threshold(score, operating) if thr is None else float(thr)
    rows, best = [], dict(z=-1e9)
    for s in starts:
        chunk = wav[s:s + N_SAMPLES]
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        if rms < min_rms:
            rows.append(dict(start_s=s / SR, rms=rms, **{f"z_{n}": np.nan for n in SCORES}, **{f"raw_{n}": np.nan for n in SCORES[:3]}))
            continue
        db = logmel(chunk)
        raw, emb = bundle.raw_scores(db)
        z = to_z(bundle.norm, {k: np.array([v]) for k, v in raw.items()}, shift)
        z = {k: float(v[0]) for k, v in z.items()}
        rows.append(dict(start_s=s / SR, rms=rms, **{f"z_{n}": z[n] for n in SCORES}, **{f"raw_{n}": raw[n] for n in SCORES[:3]}))
        if z[score] > best["z"]:
            best = dict(z=z[score], db=db, emb=emb, start_s=s / SR)
    import pandas as pd
    df = pd.DataFrame(rows)
    valid = df[f"z_{score}"].notna()
    df["flag"] = (df[f"z_{score}"] >= thr) & valid
    df["state"] = np.where(~valid, "NO SIGNAL", np.where(df["flag"], "FAULTY", "healthy"))
    n_valid, n_hit = int(valid.sum()), int(df["flag"].sum())
    verdict = ("NO SIGNAL" if n_valid == 0 else "FAULTY" if n_hit / n_valid >= 0.5 else
               "POSSIBLE FAULT" if n_hit else "HEALTHY")

    per_score = {}
    for n in SCORES:
        t_n = bundle.threshold(n, operating)
        col = df[f"z_{n}"].dropna()
        per_score[n] = dict(mean_z=float(col.mean()) if len(col) else None, max_z=float(col.max()) if len(col) else None,
                            threshold=t_n, flagged_fraction=float((col >= t_n).mean()) if len(col) else 0.0)
    warn = []
    if bundle.synthetic:
        warn.append("This model comes from an older version trained on ARTIFICIAL sound. Retrain on the real data.")
    elif bundle.quick:
        warn.append("QUICK model (small subset of the real data): not the final model. Retrain without --quick.")
    if looped:
        warn.append(f"Recording is {info['duration_s']:.1f} s (< 10 s): it was repeated to fill one window. "
                    "Use at least 10 s for reliable results.")
    if info["peak"] >= 0.99:
        warn.append("The recording is clipped (peak at full scale); distortion can look like a fault.")
    if info["peak"] < 0.01:
        warn.append("The recording is very quiet; check the microphone gain.")
    if info["orig_sr"] < 16000:
        warn.append(f"Original sample rate is {info['orig_sr']} Hz (< 16 kHz): the 4-8 kHz band is missing, "
                    "which weakens detection of high-pitched faults.")
    if n_valid == 0:
        warn.append("Every window is below the silence floor (fan off or microphone muted).")
    elif n_valid < len(df):
        warn.append(f"{len(df) - n_valid} silent window(s) were skipped.")
    if shift:
        warn.append("Own-fan calibration is active: scores were re-centred on your normal recordings.")
    return dict(name=name, verdict=verdict, score=score, operating=operating, threshold=thr, windows=df,
                n_windows=len(df), n_valid=n_valid, n_flagged=n_hit, flagged_fraction=n_hit / n_valid if n_valid else 0.0,
                peak_z=None if best["z"] < -1e8 else best["z"], per_score=per_score, warnings=warn, info=info,
                wav=wav, best=best if best["z"] > -1e8 else None)


def calibrate(bundle: Bundle, normal_sources, hop_s=5.0, min_rms=0.002):
    """Own-fan calibration: re-centre every score on the median of YOUR normal recordings (needs >= 3 windows).
    A different microphone, room or fan mostly shifts the baseline; this removes that shift."""
    T = {n: [] for n in ("cnn", "spca", "epca")}
    for nm, src in normal_sources:
        wav, _ = load_audio(src, nm)
        starts, wav, _ = make_windows(wav, hop_s)
        for s in starts:
            chunk = wav[s:s + N_SAMPLES]
            if np.sqrt(np.mean(np.square(chunk, dtype=np.float64))) < min_rms:
                continue
            raw, _ = bundle.raw_scores(logmel(chunk))
            for n in T:
                T[n].append(float(((raw_to_t(n, raw[n]) - bundle.norm[n]["mu"]) / bundle.norm[n]["sigma"])))
    n_win = len(T["cnn"])
    if n_win < 3:
        raise ValueError(f"Need at least 3 non-silent 10 s windows of normal audio; got {n_win}. "
                         "Upload 30 s or more of your fan running normally.")
    shift = {n: float(np.median(v)) for n, v in T.items()}
    return dict(shift=shift, n_windows=n_win, median_fused_before=float(np.mean(list(shift.values()))))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
COL = {"HEALTHY": "#16a34a", "FAULTY": "#dc2626", "POSSIBLE FAULT": "#f59e0b", "NO SIGNAL": "#64748b"}

# Dark theme style applied to all matplotlib figures
_DARK = dict(facecolor="#0f1b3d", edgecolor="none")
_AX_DARK = dict(facecolor="#132040", labelcolor="#cbd5e1", titlecolor="#e2e8f0")


def _apply_dark(fig, axes):
    """Apply dark theme to a matplotlib Figure and its Axes."""
    fig.patch.set_facecolor("#0f1b3d")
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor("#132040")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a8a")


def _hz_to_bin(f, fmax=8000.0):
    mel = lambda x: x / (200 / 3) if x < 1000 else 15 + math.log(x / 1000) / (math.log(6.4) / 27)
    return N_MELS * mel(f) / mel(fmax)


def fig_timeline(res):
    from matplotlib.figure import Figure
    df, sc, thr = res["windows"], res["score"], res["threshold"]
    fig = Figure(figsize=(9.4, 6.4), constrained_layout=True)
    ax = fig.subplots(2, 1)
    _apply_dark(fig, ax)
    wav = res["wav"]
    t = np.arange(len(wav)) / SR
    step = max(1, len(wav) // 30000)
    ax[0].plot(t[::step], wav[::step], lw=0.4, color="#60a5fa")
    for _, r in df.iterrows():
        if r["flag"]:
            ax[0].axvspan(r["start_s"], r["start_s"] + N_SAMPLES / SR, color="#dc2626", alpha=0.25)
    ax[0].set(title="Waveform (red = windows flagged faulty)", xlabel="time (s)", ylabel="amplitude")
    ax[0].set_xlim(0, t[-1] if len(t) else 10)
    mids = df["start_s"] + N_SAMPLES / SR / 2
    cols = [COL["FAULTY"] if h else COL["HEALTHY"] for h in df["flag"]]
    wid = 0.9 * (float(mids.diff().median()) if len(mids) > 1 else 4.5)
    ax[1].bar(mids, df[f"z_{sc}"].fillna(0), width=max(0.4, min(wid, 4.8)), color=cols)
    ax[1].axhline(thr, color="#f59e0b", ls="--", lw=1.3, label=f"threshold {thr:.2f}")
    ax[1].axhline(0, color="#475569", lw=0.8)
    ax[1].set(title=f"{SCORE_LABEL[sc]} per 10 s window (z-units of healthy sound; higher = more unusual)",
              xlabel="window centre (s)", ylabel="z-score")
    leg = ax[1].legend(loc="upper right", facecolor="#0f1b3d", edgecolor="#1e3a8a")
    for t_obj in leg.get_texts(): t_obj.set_color("#e2e8f0")
    return fig


def fig_spectrogram(res, cam=None):
    from matplotlib.figure import Figure
    b = res["best"]
    fig = Figure(figsize=(9.4, 3.9 if cam is None else 7.4), constrained_layout=True)
    if b is None:
        ax = fig.subplots(1, 1)
        _apply_dark(fig, [ax])
        ax.text(0.5, 0.5, "no analysable window", ha="center", color="#94a3b8")
        return fig
    axs = fig.subplots(1 if cam is None else 2, 1)
    axs = [axs] if cam is None else list(axs)
    _apply_dark(fig, axs)
    ticks = [0, 500, 1000, 2000, 4000, 8000]
    for a in axs:
        a.set_yticks([_hz_to_bin(f) if f else 0 for f in ticks])
        a.set_yticklabels([f"{f/1000:g}k" if f >= 1000 else str(f) for f in ticks])
        a.axhline(_hz_to_bin(4000), color="#f59e0b", ls=":", lw=0.9)
        a.set_ylabel("frequency (Hz)"); a.set_xlabel("time within window (s)")
    im = axs[0].imshow(b["db"], origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0, extent=[0, 10, 0, N_MELS])
    axs[0].set_title(f"Log-Mel spectrogram of the most unusual window (starts at {b['start_s']:.0f} s); dotted = 4 kHz")
    cb = fig.colorbar(im, ax=axs[0], format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color="#94a3b8")
    cb.ax.tick_params(labelcolor="#94a3b8")
    if cam is not None:
        axs[1].imshow(b["db"], origin="lower", aspect="auto", cmap="gray", vmin=-80, vmax=0, extent=[0, 10, 0, N_MELS])
        axs[1].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5, vmin=0, vmax=1, extent=[0, 10, 0, N_MELS])
        axs[1].set_title("Grad-CAM: where the network is listening (red = most influential)")
    return fig


def fig_pca_map(bundle, emb=None, extra=None):
    from matplotlib.figure import Figure
    v = bundle.viz
    fig = Figure(figsize=(7.2, 5.4), constrained_layout=True)
    ax = fig.subplots(1, 1)
    _apply_dark(fig, [ax])
    ax.scatter(*v["train_pc"].T, s=10, c="#22c55e", alpha=0.30, label="healthy (training)")
    ty = np.asarray(v["test_y"])
    test_pc = np.asarray(v["test_pc"])
    if len(test_pc) == len(ty):
        h_pts = test_pc[ty == 0]
        f_pts = test_pc[ty == 1]
        if len(h_pts): ax.scatter(*h_pts.T, s=22, c="#4ade80", marker="^", label="healthy (test)")
        if len(f_pts): ax.scatter(*f_pts.T, s=24, c="#f87171", marker="x", label="faulty (test)")
    if emb is not None:
        p = bundle.project(emb)
        ax.scatter([p[0]], [p[1]], s=260, c="#fbbf24", marker="*", edgecolors="#fff", zorder=5, label="your recording")
    ax.set(xlabel=f"PC1 ({100*float(v['var1']):.1f} % of variance)",
           ylabel=f"PC2 ({100*float(v['var2']):.1f} % of variance)",
           title="Embedding PCA map: where does your recording fall?")
    leg = ax.legend(fontsize=8, facecolor="#0f1b3d", edgecolor="#1e3a8a")
    for t_obj in leg.get_texts(): t_obj.set_color("#e2e8f0")
    return fig


# --------------------------------------------------------------------------- #
# Telegram alerts (deck: future work "Phone app + Telegram alerts")
# --------------------------------------------------------------------------- #
def send_telegram(token: str, chat_id: str, text: str, api_base: str = "https://api.telegram.org", timeout=10):
    """-> (ok, message). Token and chat id are never stored or logged."""
    if not token or not chat_id:
        return False, "Telegram token and chat id are required."
    try:
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        with urllib.request.urlopen(urllib.request.Request(f"{api_base}/bot{token}/sendMessage", data=data), timeout=timeout) as r:
            ok = 200 <= r.status < 300
            return ok, "alert sent" if ok else f"Telegram answered HTTP {r.status}"
    except Exception as e:
        return False, f"could not send alert: {type(e).__name__}"


def alert_text(res) -> str:
    r = res
    return (f"EchoLab alert: {r['name'] or 'recording'} -> {r['verdict']}\n"
            f"{r['n_flagged']}/{r['n_valid']} windows above threshold ({r['threshold']:.2f}); peak z-score "
            f"{r['peak_z']:.2f}" if r["peak_z"] is not None else f"EchoLab: {r['name']} -> {r['verdict']}")


# --------------------------------------------------------------------------- #
def main():
    import sys
    if not Bundle.exists():
        print("No trained EchoLab model found in models/echolab/.  Train it with:  python echolab_train.py")
        return 1
    b = Bundle()
    if b.synthetic or b.quick:
        print("!! Not a final model (" + ("older artificial-data version" if b.synthetic else "--quick run") + ").\n")
    for f in sys.argv[1:]:
        r = analyze_audio(b, f, Path(f).name)
        print(f"{r['name']:34s} {r['verdict']:15s} flagged {r['n_flagged']}/{r['n_valid']} windows  "
              f"peak z {r['peak_z'] if r['peak_z'] is None else round(r['peak_z'], 2)}  (threshold {r['threshold']:.2f})")
        for w in r["warnings"]:
            print("   -", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
