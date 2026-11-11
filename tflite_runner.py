"""
tflite_runner.py -- thin wrapper around the TFLite interpreter.

Uses `tflite_runtime` (tiny, for the Raspberry Pi) if installed, else full TensorFlow.
Handles float and integer-quantised input/output tensors transparently.
"""
from __future__ import annotations

import time

import numpy as np


def _interpreter_cls():
    try:
        from tflite_runtime.interpreter import Interpreter          # Raspberry Pi (classic)
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter          # Raspberry Pi (LiteRT, successor)
        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf                                          # PC / Colab
    return tf.lite.Interpreter


class TFLiteModel:
    """Works with 1-output models (P(abnormal)) and with the EchoLab 2-output export
    [P(abnormal), 1280-d embedding]; outputs are identified by size, not by position."""

    def __init__(self, path, num_threads: int = 4):
        self.path = str(path)
        self.it = _interpreter_cls()(model_path=self.path, num_threads=num_threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        self.out = next(o for o in outs if int(np.prod(o["shape"])) == 1)           # probability
        self.emb_out = next((o for o in outs if int(np.prod(o["shape"])) > 1), None)  # embedding (optional)
        self.width = int(self.inp["shape"][2])                      # 313 (10 s) or 63 (2 s)

    def _read(self, d):
        y = self.it.get_tensor(d["index"])
        if d["dtype"] != np.float32:
            scale, zp = d["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        return y

    def _run(self, x):
        d = self.inp
        if d["dtype"] != np.float32:                                 # quantised input
            scale, zp = d["quantization"]
            info = np.iinfo(d["dtype"])
            x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(d["dtype"])
        self.it.set_tensor(d["index"], x)
        self.it.invoke()

    def predict(self, x: np.ndarray) -> float:
        """x: float32 (1, 128, W, 1) scaled to [-1, 1]. Returns P(abnormal)."""
        self._run(x)
        return float(np.ravel(self._read(self.out))[0])

    def predict_both(self, x: np.ndarray):
        """-> (P(abnormal), embedding vector or None)."""
        self._run(x)
        p = float(np.ravel(self._read(self.out))[0])
        e = np.ravel(self._read(self.emb_out)).astype(np.float64) if self.emb_out is not None else None
        return p, e

    def benchmark(self, runs: int = 100, warmup: int = 10) -> dict:
        x = np.random.default_rng(0).uniform(-1, 1, tuple(self.inp["shape"])).astype(np.float32)
        for _ in range(warmup):
            self.predict(x)
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict(x)
            t.append((time.perf_counter() - t0) * 1000)
        t = np.array(t)
        return dict(mean_ms=float(t.mean()), median_ms=float(np.median(t)),
                    p95_ms=float(np.percentile(t, 95)), min_ms=float(t.min()), runs=runs)
