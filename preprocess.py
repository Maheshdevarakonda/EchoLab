#!/usr/bin/env python3
"""
preprocess.py -- convert every clip listed in dataset/fan/manifest.csv into a
log-Mel spectrogram (128 x 311) and cache them in one array.

Outputs (dataset/cache/):
  X.npy        float16  (N, 128, 313)   log-Mel in dB, range [-80, 0]
  y.npy        int8     (N,)            0 = normal, 1 = abnormal
  meta.csv     path, label, split, machine_id  (row i  <->  X[i])
  info.json    source (dcase2020|kaggle|user_dataset), machine, ids, shapes, counts

Usage:  python preprocess.py [--root dataset/fan]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from download_data import load_audio, read_source
from features_np import N_FRAMES, N_MELS, SR, logmel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--root", default=None, help="default: dataset/<machine>")
    ap.add_argument("--out", default="dataset/cache")
    ap.add_argument("--ids", default=None, help="comma-separated machine IDs to keep, e.g. 00 (deck: MIMII fan id_00)")
    a = ap.parse_args()
    root, out = Path(a.root or f"dataset/{a.machine}"), Path(a.out)

    man = root / "manifest.csv"
    if not man.is_file():
        print(f"{man} not found. Run download_data.py first.")
        return 1
    rows = list(csv.DictReader(open(man)))
    if a.ids:
        keep = {i.strip().zfill(2) for i in a.ids.split(",")}
        rows = [r for r in rows if r["machine_id"].zfill(2) in keep]
        print(f"keeping machine ids {sorted(keep)}: {len(rows)} clips")
        if not rows:
            print("No clips left after the --ids filter.")
            return 1
    n = len(rows)
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float16, shape=(n, N_MELS, N_FRAMES))
    y = np.zeros(n, dtype=np.int8)
    kept, skipped, t0 = [], 0, time.time()
    j = 0
    for i, r in enumerate(rows):
        try:
            wav, sr = load_audio(root / r["path"], sr=SR)
            if len(wav) < SR:                      # < 1 s: corrupt / truncated
                raise ValueError("clip shorter than 1 s")
            X[j] = logmel(wav).astype(np.float16)
            y[j] = int(r["label"])
            kept.append(r)
            j += 1
        except Exception as e:
            skipped += 1
            print(f"  [skip] {r['path']}: {e}")
        if (i + 1) % 250 == 0 or i + 1 == n:
            print(f"  {i+1}/{n} clips  ({time.time()-t0:.0f}s)", flush=True)
    X.flush()
    del X
    if skipped:                                     # shrink arrays to the kept rows
        Xf = np.load(out / "X.npy", mmap_mode="r")[:j]
        Xc = np.array(Xf)          # real copy before overwriting the mapped file
        del Xf
        np.save(out / "X.npy", Xc)
    y = y[:j]
    np.save(out / "y.npy", y)
    with open(out / "meta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "split", "machine_id"])
        w.writeheader()
        w.writerows(kept)
    info = dict(source=read_source(root), machine=a.machine, ids=(sorted({i.strip().zfill(2) for i in a.ids.split(",")}) if a.ids else None), n=int(j), n_normal=int((y == 0).sum()), n_abnormal=int((y == 1).sum()),
                shape=[N_MELS, N_FRAMES], skipped=skipped)
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
