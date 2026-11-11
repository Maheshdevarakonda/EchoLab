#!/usr/bin/env python3
"""
features_np.py -- NumPy-only log-Mel spectrogram (128 x 313) for a 10 s / 16 kHz clip.

WHY THIS FILE EXISTS
  The SAME function is used (a) to pre-compute training tensors and (b) on the
  Raspberry Pi at run time.  Identical code => zero train/serve skew, and the Pi
  needs only NumPy (no librosa / numba / llvmlite).

  The Mel filterbank is exported ONCE from librosa into mel_basis.npy (262 KB) and
  shipped with the project.  Parity with librosa is verified by:
        python features_np.py --test

Definition (matches librosa.feature.melspectrogram(center=True, power=2.0)
followed by power_to_db(ref=np.max, top_db=80)):
  frame = 1024 samples, hop = 512, periodic Hann window, |rFFT|^2,
  128-band Slaney mel filterbank, 10*log10, referenced to the clip maximum,
  floored at -80 dB.   10 s @ 16 kHz  ->  1 + 160000//512 = 313 frames (centre=True, 512 zero samples padded at both ends;
  the 128 x 313 'sound image' of the project deck).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 128
N_SAMPLES = 160000
N_FRAMES = 1 + N_SAMPLES // HOP                  # 313 (centre-padded)
WIN_2S = 63                                      # 63 frames * 512 / 16000 = 2.0 s (ablation window)
WIN_STARTS = [0, 62, 125, 188, 250]              # 5 windows of 63 frames covering all 313 frames
TOP_DB = 80.0
AMIN = 1e-10

_BASIS_PATH = Path(__file__).with_name("mel_basis.npy")
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)   # periodic Hann
_basis_cache = None


def export_mel_basis(path: Path = _BASIS_PATH) -> Path:
    """Run once on a PC that has librosa installed."""
    import librosa
    b = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    np.save(path, b)
    return path


def _basis() -> np.ndarray:
    global _basis_cache
    if _basis_cache is None:
        if not _BASIS_PATH.is_file():
            try:
                export_mel_basis()
            except ImportError as e:
                raise FileNotFoundError(
                    f"{_BASIS_PATH.name} missing. Run `python features_np.py --export` "
                    "on a machine with librosa and copy the file next to this script.") from e
        _basis_cache = np.load(_BASIS_PATH).astype(np.float64)
    return _basis_cache


def fix_length(wav: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    return wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))


def logmel(wav: np.ndarray) -> np.ndarray:
    """waveform (any length, 16 kHz, mono) -> float32 array (128, 313), values in [-80, 0] dB."""
    wav = np.pad(fix_length(wav), (N_FFT // 2, N_FFT // 2))      # centre=True zero padding
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
    frames = wav[idx] * _WINDOW                                  # (313, 1024)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2             # (313, 513)
    mel = _basis() @ power.T                                     # (128, 313)
    db = 10.0 * np.log10(np.maximum(mel, AMIN))
    db -= 10.0 * np.log10(max(mel.max(), AMIN))
    db = np.maximum(db, db.max() - TOP_DB)
    return db.astype(np.float32)


def to_model_input(db: np.ndarray) -> np.ndarray:
    """dB in [-80,0] -> [-1,1] (MobileNetV2 input convention). Shape (128, W) -> (1,128,W,1)."""
    return ((db.astype(np.float32) + 40.0) / 40.0)[None, :, :, None]


def _selftest() -> int:
    import librosa
    rng = np.random.default_rng(0)
    worst = 0.0
    for kind in ("noise", "tones", "short"):
        if kind == "noise":
            w = rng.normal(0, 0.1, N_SAMPLES)
        elif kind == "tones":
            t = np.arange(N_SAMPLES) / SR
            w = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t) + 0.01 * rng.normal(size=N_SAMPLES)
        else:
            w = rng.normal(0, 0.1, 100000)
        w = w.astype(np.float32)
        ref = librosa.power_to_db(
            librosa.feature.melspectrogram(y=np.pad(w, (0, N_SAMPLES - len(w))) if len(w) < N_SAMPLES else w,
                                           sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                           center=True, power=2.0), ref=np.max)
        mine = logmel(w)
        assert mine.shape == (N_MELS, N_FRAMES) == ref.shape, (mine.shape, ref.shape)
        d = float(np.abs(mine - ref).max())
        worst = max(worst, d)
        print(f"  {kind:6s}: shape {mine.shape}  max|diff| vs librosa = {d:.2e} dB")
    ok = worst < 1e-2
    print("PARITY", "PASS" if ok else "FAIL", f"(worst {worst:.2e} dB)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write mel_basis.npy using librosa")
    ap.add_argument("--test", action="store_true", help="parity test vs librosa")
    a = ap.parse_args()
    if a.export:
        print("wrote", export_mel_basis())
    if a.test:
        raise SystemExit(_selftest())
