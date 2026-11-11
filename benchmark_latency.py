#!/usr/bin/env python3
"""
benchmark_latency.py -- measure inference latency of one or more .tflite models.

RUN THIS ON THE RASPBERRY PI to get the numbers for the report (PC numbers are not Pi numbers).

  python benchmark_latency.py models/deploy_model.tflite models/baseline_resnet50_fp32.tflite
  python benchmark_latency.py models/deploy_model.tflite --threads 4 --runs 200 --json pi_bench.json

Reports for each model: size (MB), model-only latency (mean/median/p95), feature-extraction
latency (10 s log-Mel in NumPy) and end-to-end = features + model.  Also logs CPU temperature
and throttling state on a Raspberry Pi, because thermal throttling silently inflates latency.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from features_np import logmel
from tflite_runner import TFLiteModel


def pi_status() -> dict:
    out = dict(machine=platform.machine(), python=platform.python_version())
    try:
        out["cpu_temp_c"] = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:
        pass
    try:
        out["throttled"] = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                          text=True, timeout=3).stdout.strip()
    except Exception:
        pass
    try:
        out["model"] = Path("/proc/device-tree/model").read_text().strip("\x00")
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    wav = rng.normal(0, 0.1, 160000).astype(np.float32)
    logmel(wav)                                                        # warm-up
    f = []
    for _ in range(30):
        t0 = time.perf_counter(); logmel(wav); f.append((time.perf_counter() - t0) * 1000)
    feat_ms = float(np.median(f))

    status = pi_status()
    print("System:", status)
    print(f"Feature extraction (10 s log-Mel, NumPy): median {feat_ms:.1f} ms\n")
    print(f"{'model':42s} {'MB':>6s} {'median ms':>10s} {'p95 ms':>8s} {'e2e ms':>8s}")
    rows = {}
    for m in a.models:
        tm = TFLiteModel(m, num_threads=a.threads)
        b = tm.benchmark(runs=a.runs)
        size = Path(m).stat().st_size / 1e6
        rows[m] = dict(size_mb=size, **b, feature_ms=feat_ms, end_to_end_ms=b["median_ms"] + feat_ms,
                       threads=a.threads)
        print(f"{Path(m).name:42s} {size:6.2f} {b['median_ms']:10.1f} {b['p95_ms']:8.1f} {rows[m]['end_to_end_ms']:8.1f}")
    status_after = pi_status()
    print("\nAfter benchmark:", {k: v for k, v in status_after.items() if k in ("cpu_temp_c", "throttled")})
    if a.json:
        Path(a.json).write_text(json.dumps(dict(system=status, after=status_after, results=rows), indent=2))
        print("saved", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
