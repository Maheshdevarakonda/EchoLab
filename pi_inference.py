#!/usr/bin/env python3
"""
pi_inference.py -- real-time fan-fault monitor for Raspberry Pi 4.

  USB mic -> 10 s ring buffer -> NumPy log-Mel (128x311) -> TFLite MobileNetV2 -> P(abnormal)
          -> k-of-n debounce -> LED / buzzer / CSV log / web status page

Run
  python pi_inference.py                                  # live microphone
  python pi_inference.py --list-devices                   # find the USB mic index
  python pi_inference.py --simulate dataset/fan/test      # no hardware: replay wav files
  python pi_inference.py --simulate some.wav --realtime   # replay one file with real pacing

Decision logic
  * every --hop seconds (default 5) the latest 10 s are analysed
  * RMS below --min-rms  -> state NO_SIGNAL (fan off / mic unplugged); the model is NOT run,
    because per-clip normalisation would otherwise turn background noise into a "spectrogram"
  * prob >= threshold    -> "hit".  ALARM when >= K of the last N decisions are hits
    (default 3 of 5) -> suppresses one-off spikes (door slam, dropped tool)
  * scores, thresholds and PCA files come from models/echolab/ (written by echolab_train.py)

Hardware (BCM numbering; see HARDWARE.md):  green LED 22, red LED 17, buzzer 27, ack button 23 -> GND
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import signal
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from features_np import N_SAMPLES, SR, WIN_STARTS, logmel, to_model_input



# --------------------------------------------------------------------------- #
# GPIO (optional -- script also runs on a laptop)
# --------------------------------------------------------------------------- #
class Indicators:
    def __init__(self, green, red, buzzer, button):
        self.ok_led = self.red_led = self.buzzer = self.button = None
        self.acked = False
        try:
            from gpiozero import LED, Button, Buzzer
            if green >= 0: self.ok_led = LED(green)
            if red >= 0: self.red_led = LED(red)
            if buzzer >= 0: self.buzzer = Buzzer(buzzer)
            if button >= 0:
                self.button = Button(button, pull_up=True, bounce_time=0.05)
                self.button.when_pressed = self._ack
            print("[gpio] indicators ready")
        except Exception as e:
            print(f"[gpio] disabled ({type(e).__name__}: {e}) -- console/log only")

    def _ack(self):
        self.acked = True

    def show(self, state: str):
        on = lambda d, v: (d.on() if v else d.off()) if d else None
        on(self.ok_led, state == "OK")
        on(self.red_led, state == "ALARM")
        if state == "ALARM":
            on(self.buzzer, not self.acked)
        else:
            on(self.buzzer, False)
            self.acked = False

    def close(self):
        for d in (self.ok_led, self.red_led, self.buzzer):
            try:
                d and d.off() and d.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# audio sources
# --------------------------------------------------------------------------- #
def _resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class MicSource:
    """Continuous capture into a 10 s ring buffer (kept at the device's native rate)."""

    def __init__(self, device=None, hop=5.0):
        import sounddevice as sd
        self.sd = sd
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        self.native = SR
        try:
            self.stream = self._open(device, SR)
        except Exception:
            self.native = int(sd.query_devices(device, "input")["default_samplerate"])
            print(f"[mic] {SR} Hz not supported by device, capturing at {self.native} Hz and resampling")
            self.stream = self._open(device, self.native)
        self.ring = np.zeros(self.native * 10, np.float32)
        self.w = 0
        self.total = 0
        self.lock = threading.Lock()
        self.hop = hop
        self.stream.start()

    def _open(self, device, sr):
        return self.sd.InputStream(device=device, channels=1, samplerate=sr, dtype="float32",
                                   blocksize=sr // 4, callback=self._cb)

    def _cb(self, indata, frames, t, status):
        x = indata[:, 0]
        n = len(x)
        with self.lock:
            end = self.w + n
            if end <= len(self.ring):
                self.ring[self.w:end] = x
            else:
                k = len(self.ring) - self.w
                self.ring[self.w:] = x[:k]
                self.ring[:n - k] = x[k:]
            self.w = end % len(self.ring)
            self.total += n

    def windows(self, stop: threading.Event):
        need = self.native * 10
        nxt = need
        while not stop.is_set():
            if self.total >= nxt:
                with self.lock:
                    snap = np.concatenate([self.ring[self.w:], self.ring[:self.w]]).copy()
                nxt += int(self.hop * self.native)
                yield _resample(snap, self.native), datetime.now()
            else:
                stop.wait(0.2)

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass


class FileSource:
    """Replay a wav file (sliding 10 s windows) or a directory (one window per file)."""

    def __init__(self, path, hop=5.0, realtime=False):
        self.path, self.hop, self.realtime = Path(path), hop, realtime

    def _read(self, f):
        import soundfile as sf
        x, sr = sf.read(str(f), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return _resample(x, sr)

    def windows(self, stop: threading.Event):
        files = sorted(self.path.rglob("*.wav")) if self.path.is_dir() else [self.path]
        for f in files:
            try:
                x = self._read(f)
            except Exception as e:
                print(f"[sim] skipping {f.name}: {type(e).__name__}")
                continue
            starts = [0] if self.path.is_dir() else range(0, max(1, len(x) - N_SAMPLES + 1), int(self.hop * SR))
            for s in starts:
                if stop.is_set():
                    return
                print(f"[sim] {f.name} @ {s/SR:.0f}s")
                yield x[s:s + N_SAMPLES], datetime.now()
                if self.realtime:
                    stop.wait(self.hop)

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
class Detector:
    """Uses the same EchoLab bundle and fused score as the website (numpy + tflite only, no TensorFlow)."""

    def __init__(self, bundle, score="fused", operating="roc", k=3, n=5, min_rms=0.003):
        self.b, self.score_name = bundle, score
        self.thr = bundle.threshold(score, operating)
        self.k, self.min_rms = k, min_rms
        self.hist = collections.deque(maxlen=n)

    def step(self, wav: np.ndarray) -> dict:
        rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
        if rms < self.min_rms:
            self.hist.clear()
            return dict(state="NO_SIGNAL", prob=None, rms=rms, latency_ms=0.0, hits=0)
        from echolab_core import to_z
        t0 = time.perf_counter()
        raw, _ = self.b.raw_scores(logmel(wav))
        z = float(to_z(self.b.norm, {k_: np.array([v]) for k_, v in raw.items()})[self.score_name][0])
        ms = (time.perf_counter() - t0) * 1000
        self.hist.append(z >= self.thr)
        hits = int(sum(self.hist))
        state = "ALARM" if hits >= self.k else ("WARN" if self.hist[-1] else "OK")
        return dict(state=state, prob=z, rms=rms, latency_ms=ms, hits=hits)


# --------------------------------------------------------------------------- #
# status web page + logging
# --------------------------------------------------------------------------- #
STATUS = dict(state="STARTING", prob=None, rms=None, threshold=None, updated="-", history=[])
STATUS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with STATUS_LOCK:
            s = json.loads(json.dumps(STATUS))
        if self.path.startswith("/status.json"):
            body, ctype = json.dumps(s).encode(), "application/json"
        else:
            colour = dict(OK="#2e7d32", WARN="#f9a825", ALARM="#c62828").get(s["state"], "#546e7a")
            rows = "".join(f"<tr><td>{h[0]}</td><td>{h[1]}</td><td>{h[2]}</td></tr>" for h in s["history"][-15:][::-1])
            p = "-" if s["prob"] is None else f"{s['prob']:.3f}"
            body = (f"<html><head><meta http-equiv='refresh' content='5'><meta name='viewport' "
                    f"content='width=device-width,initial-scale=1'><title>Fan monitor</title></head>"
                    f"<body style='font-family:sans-serif;max-width:560px;margin:2em auto'>"
                    f"<h2>Industrial fan acoustic monitor</h2>"
                    f"<div style='background:{colour};color:#fff;padding:1em;border-radius:8px;font-size:2em'>"
                    f"{s['state']}</div><p>P(abnormal) = <b>{p}</b> &nbsp; threshold = {s['threshold']} "
                    f"&nbsp; updated {s['updated']}</p><table border=1 cellpadding=4><tr><th>time</th>"
                    f"<th>state</th><th>P</th></tr>{rows}</table></body></html>").encode()
            ctype = "text/html"
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)


def open_log(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    f = log_dir / f"events_{datetime.now():%Y%m%d}.csv"
    new = not f.exists()
    fh = open(f, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["timestamp", "state", "prob", "threshold", "rms", "hits_in_window", "latency_ms"])
    return fh, w


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/echolab", help="folder written by echolab_train.py")
    ap.add_argument("--score", default="fused", choices=["fused", "cnn", "spca", "epca"])
    ap.add_argument("--operating", default="roc", choices=["roc", "label_free"])
    ap.add_argument("--threshold", type=float, default=None, help="override the z-score threshold")
    ap.add_argument("--device", default=None, help="sounddevice index or name")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--simulate", default=None, help="wav file or folder instead of microphone")
    ap.add_argument("--realtime", action="store_true", help="pace --simulate like a live stream")
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-rms", type=float, default=0.003)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8080, help="status web page (0 = off)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--gpio", default="22,17,27,23", help="green,red,buzzer,button BCM pins (-1 = unused)")
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    from echolab_core import Bundle
    if not Bundle.exists(a.bundle):
        print(f"No trained model in {a.bundle}/. Train on a PC (python echolab_train.py) and copy the folder here "
              "(model.tflite, config.json, metrics.json, pca_spec.npz, pca_emb.npz, viz.npz).")
        return 1
    bundle = Bundle(a.bundle)
    if bundle.synthetic or bundle.quick:
        print("!! this is not a final model (older artificial-data version or --quick run)")
    det = Detector(bundle, a.score, a.operating, a.k, a.n, a.min_rms)
    if a.threshold is not None:
        det.thr = a.threshold
    thr = det.thr
    print(f"[model] {a.bundle}  score={a.score}  z-threshold {thr:.2f}  ({a.k} of {a.n} windows to alarm)")

    g = [int(v) for v in a.gpio.split(",")]
    ind = Indicators(*g)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    if a.port:
        try:
            srv = HTTPServer(("0.0.0.0", a.port), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            print(f"[web] status page on http://<pi-ip>:{a.port}/")
        except OSError as e:
            print(f"[web] disabled: {e}")

    src = FileSource(a.simulate, a.hop, a.realtime) if a.simulate else MicSource(a.device, a.hop)
    if not a.simulate:
        print(f"[mic] listening; first result after 10 s of audio.  Ctrl+C to stop.")
    fh, w = open_log(Path(a.log_dir))
    try:
        for wav, ts in src.windows(stop):
            r = det.step(wav)
            ind.show(r["state"])
            p = "" if r["prob"] is None else f"{r['prob']:.4f}"
            w.writerow([ts.isoformat(timespec="seconds"), r["state"], p, f"{thr:.3f}", f"{r['rms']:.5f}",
                        r["hits"], f"{r['latency_ms']:.1f}"]); fh.flush()
            print(f"{ts:%H:%M:%S}  {r['state']:9s} z={p or '  -  '}  rms={r['rms']:.4f}  "
                  f"hits={r['hits']}/{a.n}  infer={r['latency_ms']:.0f} ms")
            with STATUS_LOCK:
                STATUS.update(state=r["state"], prob=r["prob"], rms=r["rms"], threshold=round(thr, 3),
                              updated=f"{ts:%H:%M:%S}")
                STATUS["history"].append([f"{ts:%H:%M:%S}", r["state"], p or "-"])
                del STATUS["history"][:-100]
    finally:
        ind.close(); src.close(); fh.close()
        print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
