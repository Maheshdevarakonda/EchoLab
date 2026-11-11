#!/usr/bin/env python3
"""
analyze.py -- audio helpers and the PCA detector used by EchoLab (library; the website is app.py).

  load_audio     any audio file / bytes -> mono 16 kHz float32 (WAV/FLAC/OGG; MP3/M4A via librosa+ffmpeg)
  make_windows   10 s analysis windows (short clips are repeated, the end of a long clip is always covered)
  stack_frames   log-Mel -> 5-frame context vectors (640-d) for the spectrogram PCA
  PCADetector    PCA fitted on HEALTHY sound; score = reconstruction error (no faulty example needed)
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from math import gcd
from pathlib import Path

import numpy as np

from features_np import N_SAMPLES, SR

CONTEXT = 5          # stack 5 consecutive Mel frames -> 128*5 = 640-dim vectors (as in the DCASE baseline)


def _resample(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(sr_in, SR)
    return resample_poly(x, SR // g, sr_in // g).astype(np.float32)


def load_audio(src, name: str = "") -> tuple[np.ndarray, dict]:
    """src: path or raw bytes.  Returns (mono float32 @16 kHz, info dict).  WAV/FLAC/OGG always;
    MP3/M4A if libsndfile>=1.1 or librosa+ffmpeg is available."""
    import soundfile as sf
    data = None
    try:
        data, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src),
                           dtype="float32", always_2d=True)
        data = data.T                                                # (channels, samples)
    except Exception:
        try:
            import librosa
            path = str(src)
            tmp = None
            if isinstance(src, (bytes, bytearray)):
                tmp = tempfile.NamedTemporaryFile(suffix=Path(name or "x.mp3").suffix or ".mp3", delete=False)
                tmp.write(src); tmp.close(); path = tmp.name
            data, sr = librosa.load(path, sr=None, mono=False)
            data = np.atleast_2d(data)
            if tmp:
                os.unlink(tmp.name)
        except Exception as e:
            raise ValueError(f"Cannot read '{name or src}': unsupported or corrupt audio. "
                             f"Please convert it to WAV. ({type(e).__name__})") from e
    info = dict(orig_sr=int(sr), channels=int(data.shape[0]), duration_s=data.shape[1] / sr,
                peak=float(np.abs(data).max()) if data.size else 0.0)
    mono = data[0] if data.shape[0] > 2 else data.mean(axis=0)     # mic arrays: first channel (as in DCASE); stereo: average
    return _resample(mono, sr), info


def make_windows(wav: np.ndarray, hop_s: float = 5.0):
    """-> (list of start samples, wav possibly looped, looped flag)."""
    n = len(wav)
    if n < SR:
        raise ValueError("Recording is shorter than 1 second.")
    looped = False
    if n < N_SAMPLES:                                               # 1-10 s: repeat to fill one window
        wav = np.tile(wav, int(np.ceil(N_SAMPLES / n)))[:N_SAMPLES]
        looped, n = True, N_SAMPLES
    hop = max(1, int(hop_s * SR))
    starts = list(range(0, n - N_SAMPLES + 1, hop))
    if starts[-1] != n - N_SAMPLES:
        starts.append(n - N_SAMPLES)                                # always cover the end
    return starts, wav, looped


def stack_frames(db: np.ndarray) -> np.ndarray:
    """log-Mel (128, T) -> (T-4, 640) context vectors, scaled to about [-1, 1]."""
    n = db.shape[1] - CONTEXT + 1
    x = (db + 40.0) / 40.0
    return np.concatenate([x[:, i:i + n].T for i in range(CONTEXT)], axis=1).astype(np.float64)


class PCADetector:
    """Learn the subspace of NORMAL sound; a frame that the subspace reconstructs badly is anomalous.
    score(window) = mean over frames of the squared reconstruction error."""

    def __init__(self, var: float = 0.90, max_k: int = 50):
        self.var, self.max_k = var, max_k
        self.mean = self.components = self.eigvals = None
        self.k = 0

    def fit(self, frames: np.ndarray):
        self.mean = frames.mean(axis=0)
        X = frames - self.mean
        cov = X.T @ X / max(len(X) - 1, 1)
        w, V = np.linalg.eigh(cov)
        w, V = np.clip(w[::-1], 0, None), V[:, ::-1]
        ratio = np.cumsum(w) / max(w.sum(), 1e-12)
        self.k = int(min(np.searchsorted(ratio, self.var) + 1, self.max_k, len(w)))
        self.components, self.eigvals = V[:, :self.k].T.copy(), w
        return self

    def frame_errors(self, frames: np.ndarray) -> np.ndarray:
        X = frames - self.mean
        rec = (X @ self.components.T) @ self.components
        return ((X - rec) ** 2).mean(axis=1)

    def score_db(self, db: np.ndarray) -> float:
        return float(self.frame_errors(stack_frames(db)).mean())

    def explained_ratio(self) -> np.ndarray:
        return np.cumsum(self.eigvals) / max(self.eigvals.sum(), 1e-12)

    def save(self, path, threshold: float, meta: dict | None = None):
        np.savez(path, mean=self.mean, components=self.components, eigvals=self.eigvals, k=self.k,
                 var=self.var, threshold=threshold, meta=json.dumps(meta or {}))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        d = cls(float(z["var"]), int(z["k"]))
        d.mean, d.components, d.eigvals, d.k = z["mean"], z["components"], z["eigvals"], int(z["k"])
        return d, float(z["threshold"]), json.loads(str(z["meta"]))
