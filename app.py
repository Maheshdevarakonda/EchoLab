#!/usr/bin/env python3
"""
app.py -- EchoLab website (Gradio).      python app.py          ->  http://localhost:7860

Tabs:  Analyze (upload or record) | Batch | Model & Results (accuracy, F1, AUC ...) | PCA Lab |
       How it works | Setup & Train (downloads MIMII and trains the real model from the browser) | Share (QR)

There is NO demo model.  Until models/echolab/ exists (created by echolab_train.py or by the
"Setup & Train" tab) the website shows the setup screen.  Every accuracy figure shown is read from
models/echolab/metrics.json, which only the training script writes.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

import echolab_core as core
from echolab_core import SCORE_LABEL, SCORES, Bundle

try:
    import gradio as gr
except ImportError:                                              # pragma: no cover
    sys.exit("Gradio is not installed.  Run:  pip install -r requirements.txt")

GR_MAJOR = int(gr.__version__.split(".")[0])
FIG_DIR = core.BUNDLE / "figures"
OPS = {"ROC-best threshold (tuned with labelled faults)": "roc", "Label-free threshold (needs no faulty clip)": "label_free"}
SCORE_CHOICES = {SCORE_LABEL[n]: n for n in ("fused", "cnn", "spca", "epca")}
STATE = dict(bundle=None, error=None, share_url=None, train=None)

# --------------------------------------------------------------------------- #
# look and feel (EchoLab deck: navy, blue, teal, amber, green / red)
# --------------------------------------------------------------------------- #
CSS = """
.gradio-container {max-width: 1280px !important; margin: auto; font-family: Lato, 'Segoe UI', system-ui, sans-serif;}

/* ---- Hero banner ---- */
.el-hero {background: linear-gradient(135deg,#0f1b3d 0%,#16275a 100%); color:#fff; border-radius:16px; padding:22px 28px; margin-bottom:10px;}
.el-hero .brand {letter-spacing:.35em; font-size:.78rem; color:#60a5fa; font-weight:700;}
.el-hero h1 {margin:.25em 0 .1em 0; font-size:2rem; line-height:1.15; color:#fff;}
.el-hero p {margin:0; color:#cbd5e1; font-size:.95rem;}
.el-kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:14px;}
.el-kpi {background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); border-radius:10px; padding:10px 14px;}
.el-kpi b {display:block; font-size:1.45rem; color:#fff;} .el-kpi span {font-size:.74rem; color:#93c5fd;}

/* ---- Alert banners ---- */
.el-banner {border-radius:10px; padding:10px 16px; margin:6px 0; font-weight:600;}
.el-warn  {background:#3b1111; color:#fca5a5; border:1px solid #7f1d1d;}
.el-info  {background:#0f2033; color:#93c5fd; border:1px solid #1e40af;}
.el-amber {background:#2d1f00; color:#fcd34d; border:1px solid #92400e;}

/* ---- Verdict box ---- */
.el-verdict {border-radius:16px; padding:20px 26px; color:#fff; margin:6px 0;}
.el-verdict h2 {margin:0; font-size:2.3rem; color:#fff;}
.el-verdict p  {margin:.3em 0 0 0; color:#f8fafc; opacity:.95;}

/* ---- Dark-theme data table ---- */
table.el-t {border-collapse:collapse; width:100%; font-size:.9rem; background:#0f1b3d; color:#e2e8f0; border-radius:10px; overflow:hidden;}
table.el-t th {background:#0a1428; color:#93c5fd; padding:8px 11px; text-align:left; border-bottom:2px solid #1e3a8a;}
table.el-t td {padding:7px 11px; border-bottom:1px solid #1e2d4d; color:#e2e8f0;}
table.el-t tr:nth-child(even) td {background:#132040;}
table.el-t tr:hover td {background:#1e3056;}
table.el-t p {color:#94a3b8; font-size:.8rem; margin:.4em 0 0 0;}

/* ---- Steps & card ---- */
.el-step {display:inline-block; background:#2563eb; color:#fff; border-radius:8px; padding:8px 12px; margin:3px; font-weight:600;}
.el-arrow {color:#60a5fa; font-weight:700; margin:0 2px;}
.el-card  {border:1px solid #1e3a8a; border-radius:12px; padding:12px 16px; background:#0f2033; color:#cbd5e1; margin:6px 0;}
.el-card b {color:#93c5fd;}

/* ---- Force Gradio output HTML containers to dark background ---- */
.gradio-html, .gradio-html > div { background: transparent !important; }
"""


def make_theme():
    # (fonts are set in CSS: custom theme fonts trip a Gradio 6 comparison bug)
    return gr.themes.Base(primary_hue="blue", secondary_hue="amber", neutral_hue="slate")


# --------------------------------------------------------------------------- #
# model state
# --------------------------------------------------------------------------- #
def load_bundle():
    STATE["bundle"], STATE["error"] = None, None
    if Bundle.exists():
        try:
            STATE["bundle"] = Bundle()
        except Exception as e:                                   # corrupt / incomplete bundle
            STATE["error"] = f"{type(e).__name__}: {e}"
    return STATE["bundle"]


def pc(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f} %"


def hero_html():
    b = STATE["bundle"]
    cards = ""
    if b is not None:
        m, d = b.metrics, b.metrics["data"]
        o = m["final"]["operating"]["fused"]["roc_crossfit"]
        items = [(d["n_train_healthy"], "healthy training clips"), (d["n_test_faulty"], "faulty test clips"),
                 (f"{m['model']['tflite_quant_mb']:.1f} MB", "quantised model"), (pc(m["final"]["scores"]["fused"]["auc"]), "test AUC"),
                 (pc(o["accuracy"]), "accuracy"), (pc(o["balanced_accuracy"]), "balanced accuracy"), (f"{o['f1']:.3f}", "F1 score")]
        cards = '<div class="el-kpis">' + "".join(f'<div class="el-kpi"><b>{v}</b><span>{k}</span></div>' for v, k in items) + "</div>"
    banner = ""
    if b is None:
        banner = ('<div class="el-banner el-amber">No trained model yet: open the <b>Setup &amp; Train</b> tab. '
                  'It downloads the real MIMII fan data and trains the model (one click).</div>')
    elif b.synthetic:
        banner = ('<div class="el-banner el-warn">This model comes from an older version trained on ARTIFICIAL sound. '
                  'Retrain on the real MIMII data in <b>Setup &amp; Train</b>.</div>')
    elif b.quick:
        banner = ('<div class="el-banner el-amber">QUICK model: trained on a small subset of the real data to check the setup. '
                  'Run the full training in <b>Setup &amp; Train</b> for the final result.</div>')
    elif not b.metrics["meta"].get("pretrained_imagenet"):
        banner = ('<div class="el-banner el-amber">The ImageNet weights were not available, so the network was trained '
                  'from scratch: lower accuracy than with transfer learning.</div>')
    return (f'<div class="el-hero"><div class="brand">ECHOLAB</div><h1>Fan Anomaly Detection</h1>'
            f'<p>Finding faulty fans by listening to their sound &nbsp;·&nbsp; MIMII Fan id_00 &nbsp;·&nbsp; MobileNetV2 + PCA</p>{cards}</div>{banner}')


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
VCOL = core.COL
VICON = {"HEALTHY": "✔", "FAULTY": "✖", "POSSIBLE FAULT": "⚠", "NO SIGNAL": "∅"}


def verdict_html(res):
    c = VCOL[res["verdict"]]
    pz = "–" if res["peak_z"] is None else f"{res['peak_z']:.2f}"
    return (f'<div class="el-verdict" style="background:{c}"><h2>{VICON[res["verdict"]]} {res["verdict"]}</h2>'
            f'<p>{res["n_flagged"]} of {res["n_valid"]} analysed 10-second windows flagged &nbsp;·&nbsp; '
            f'peak score z = {pz} &nbsp;·&nbsp; threshold z = {res["threshold"]:.2f} &nbsp;·&nbsp; {html.escape(res["name"])}</p></div>')


def scores_html(res):
    rows = ""
    for n in SCORES:
        s = res["per_score"][n]
        flag = s["flagged_fraction"]
        col = "#dc2626" if flag >= 0.5 else "#f59e0b" if flag > 0 else "#16a34a"
        mz = "–" if s["mean_z"] is None else f"{s['mean_z']:.2f}"
        xz = "–" if s["max_z"] is None else f"{s['max_z']:.2f}"
        bold = "font-weight:700" if n == res["score"] else ""
        rows += (f'<tr style="{bold}"><td>{SCORE_LABEL[n]}</td><td>{mz}</td><td>{xz}</td><td>{s["threshold"]:.2f}</td>'
                 f'<td><span style="color:{col}">{100*flag:.0f} %</span></td></tr>')
    return ('<table class="el-t"><tr><th>Score</th><th>Mean z</th><th>Peak z</th><th>Threshold z</th><th>Windows flagged</th></tr>'
            + rows + '</table><p style="font-size:.8rem;color:#64748b">z = distance from healthy sound in healthy-standard-deviations '
            '(0 = typical healthy fan). Each score is shown at its own threshold; the verdict uses the selected score.</p>')


def warnings_html(res):
    return "".join(f'<div class="el-banner el-amber">{html.escape(w)}</div>' for w in res["warnings"])


def empty_outputs(msg):
    return (f'<div class="el-banner el-amber">{msg}</div>', "", None, None, None, None, None, "")


def run_analysis(path, score_label, op_label, offset, hop, cal, tg_token, tg_chat, tg_on):
    b = STATE["bundle"]
    if b is None:
        return empty_outputs("No trained model yet. Open the <b>Setup &amp; Train</b> tab.")
    if not path:
        return empty_outputs("Upload a recording or record with the microphone, then press <b>Analyze</b>.")
    score, op = SCORE_CHOICES[score_label], OPS[op_label]
    thr = b.threshold(score, op) + float(offset)
    try:
        res = core.analyze_audio(b, path, Path(path).name, score, op, thr, float(hop), shift=(cal or {}).get("shift"))
    except ValueError as e:
        return empty_outputs(html.escape(str(e)))
    cam = b.gradcam(res["best"]["db"]) if res["best"] is not None else None
    fig_t, fig_s = core.fig_timeline(res), core.fig_spectrogram(res, cam)
    fig_p = core.fig_pca_map(b, res["best"]["emb"] if res["best"] is not None else None)
    df = res["windows"].copy()
    df["end_s"] = df["start_s"] + 10
    show = df[["start_s", "end_s", "rms", "z_cnn", "z_spca", "z_epca", "z_fused", "state"]].round(3)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, prefix="echolab_")
    show.to_csv(tmp.name, index=False)
    note = ""
    if tg_on and res["verdict"] in ("FAULTY", "POSSIBLE FAULT"):
        ok, msg = core.send_telegram(tg_token, tg_chat, core.alert_text(res))
        note = f'<div class="el-banner {"el-info" if ok else "el-warn"}">Telegram: {html.escape(msg)}</div>'
    return (verdict_html(res) + note, scores_html(res) + warnings_html(res), fig_t, fig_s, fig_p, show, tmp.name,
            "" if cam is not None else "Grad-CAM needs TensorFlow and models/echolab/model.keras; it is not available in this environment.")


def run_calibration(files):
    b = STATE["bundle"]
    if b is None:
        return None, "No trained model yet."
    if not files:
        return None, "No calibration files: scores use the dataset's healthy reference."
    try:
        cal = core.calibrate(b, [(Path(f).name, f) for f in files])
    except ValueError as e:
        return None, f"⚠ {e}"
    sh = ", ".join(f"{n}: {v:+.2f}" for n, v in cal["shift"].items())
    return cal, f"✔ Calibrated on {cal['n_windows']} normal windows. Score shifts removed (z-units): {sh}."


def run_batch(files, score_label, op_label, offset, hop, cal):
    b = STATE["bundle"]
    if b is None:
        return None, None, "No trained model yet."
    if not files:
        return None, None, "Upload one or more recordings."
    score, op = SCORE_CHOICES[score_label], OPS[op_label]
    thr = b.threshold(score, op) + float(offset)
    import pandas as pd
    rows = []
    for f in files:
        nm = Path(f).name
        try:
            r = core.analyze_audio(b, f, nm, score, op, thr, float(hop), shift=(cal or {}).get("shift"))
            rows.append(dict(file=nm, verdict=r["verdict"], windows=r["n_valid"], flagged=r["n_flagged"],
                             flagged_pct=round(100 * r["flagged_fraction"]), peak_z=None if r["peak_z"] is None else round(r["peak_z"], 2),
                             duration_s=round(r["info"]["duration_s"], 1)))
        except Exception as e:
            rows.append(dict(file=nm, verdict=f"ERROR: {e}"))
    df = pd.DataFrame(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, prefix="echolab_batch_")
    df.to_csv(tmp.name, index=False)
    n_f = int((df["verdict"] == "FAULTY").sum())
    return df, tmp.name, f"{len(df)} files analysed, {n_f} flagged FAULTY."


def project_only(path):
    b = STATE["bundle"]
    if b is None:
        return None, "No trained model yet."
    if not path:
        return core.fig_pca_map(b), "Upload a recording to place it on the map."
    try:
        r = core.analyze_audio(b, path, Path(path).name)
    except ValueError as e:
        return core.fig_pca_map(b), str(e)
    return core.fig_pca_map(b, r["best"]["emb"] if r["best"] else None), f"Placed the most unusual window of {r['name']} (verdict: {r['verdict']})."


# --------------------------------------------------------------------------- #
# results tab
# --------------------------------------------------------------------------- #
def results_html():
    b = STATE["bundle"]
    if b is None:
        return '<div class="el-banner el-amber">Train the model first (Setup &amp; Train tab). All figures here come from the measured results file.</div>'
    m = b.metrics
    fin, cv, d, mod = m["final"], m["cv"], m["data"], m["model"]
    out = ""
    if b.synthetic or b.quick:
        out += '<div class="el-banner el-amber">QUICK / OLD model: these numbers are not the final result.</div>'
    out += (f'<div class="el-card"><b>Test set (never used for training):</b> {d["n_test_healthy"]} healthy + {d["n_test_faulty"]} faulty clips. '
            f'<b>Training:</b> {d["n_train_healthy"]} healthy clips only. Trained {m["meta"]["created"]} on <b>{m["meta"]["data_source"]}</b> data.</div>')
    rows = ""
    for n in SCORES:
        s = fin["scores"][n]
        rows += (f'<tr><td>{SCORE_LABEL[n]}</td><td>{pc(s["auc"])}</td><td>{pc(s["pauc"])}</td><td>{pc(s["ap"])}</td>'
                 f'<td>{pc(cv["test_auc_mean"][n])} ± {pc(cv["test_auc_std"][n])}</td></tr>')
    out += ('<h3>Ranking quality (threshold-free)</h3><table class="el-t"><tr><th>Score</th><th>AUC</th><th>pAUC (FPR ≤ 0.1)</th><th>AP</th>'
            f'<th>5-fold average AUC ± spread</th></tr>{rows}</table>')
    rows = ""
    for n in SCORES:
        for key, nm in (("roc_crossfit", "ROC-best (cross-fitted)"), ("label_free", "Label-free (95th pct healthy)")):
            o = fin["operating"][n][key]
            bold = "font-weight:700" if n == "fused" else ""
            rows += (f'<tr style="{bold}"><td>{SCORE_LABEL[n]}</td><td>{nm}</td><td>{pc(o["accuracy"])}</td><td>{pc(o["balanced_accuracy"])}</td>'
                     f'<td>{pc(o["precision"])}</td><td>{pc(o["recall"])}</td><td>{pc(o["specificity"])}</td><td>{o["f1"]:.3f}</td>'
                     f'<td>{o["tp"]} / {o["fp"]} / {o["fn"]} / {o["tn"]}</td></tr>')
    out += ('<h3>Accuracy and F1 at the operating threshold</h3><table class="el-t"><tr><th>Score</th><th>Threshold rule</th><th>Accuracy</th>'
            f'<th>Balanced accuracy</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>TP / FP / FN / TN</th></tr>{rows}</table>')
    out += (f'<div class="el-banner el-info">Read accuracy with care: the test set is {pc(fin["trivial_accuracy_predict_all_faulty"])} faulty, '
            f'so answering "faulty" for every clip would already score {pc(fin["trivial_accuracy_predict_all_faulty"])} accuracy. '
            'Balanced accuracy, AUC and F1 are the honest figures.</div>')
    fa, fp = fin["scores"]["fused"]["auc"], fin["scores"]["fused"]["pauc"]
    f1 = fin["operating"]["fused"]["roc_crossfit"]["f1"]
    ms = mod["host_latency_ms"]["median_ms"]
    chk = lambda c: '<b style="color:#16a34a">MET</b>' if c else '<b style="color:#dc2626">NOT MET</b>'
    out += ('<h3>Project targets (from the deck) vs measured</h3><table class="el-t"><tr><th>Target</th><th>Goal</th><th>Measured</th><th></th></tr>'
            f'<tr><td>AUC</td><td>90–92 %</td><td>{pc(fa)}</td><td>{chk(fa >= .90)}</td></tr>'
            f'<tr><td>pAUC</td><td>≈ 78 %</td><td>{pc(fp)}</td><td>{chk(fp >= .78)}</td></tr>'
            f'<tr><td>F1</td><td>0.87</td><td>{f1:.3f}</td><td>{chk(f1 >= .87)}</td></tr>'
            f'<tr><td>Beats the baseline autoencoder</td><td>&gt; {m["baseline"]["auc"]:.2f} %</td><td>{pc(fa)}</td><td>{chk(fa*100 > m["baseline"]["auc"])}</td></tr>'
            f'<tr><td>Quantised model size</td><td>≈ 4 MB</td><td>{mod["tflite_quant_mb"]:.1f} MB (fp32 TFLite {mod["tflite_fp32_mb"]:.1f} MB)</td><td>{chk(mod["tflite_quant_mb"] <= 4.5)}</td></tr>'
            f'<tr><td>Speed on a laptop</td><td>≈ 25 ms</td><td>{ms:.0f} ms (this computer)</td><td>{chk(ms <= 25)}</td></tr>'
            '<tr><td>Speed on Raspberry Pi 4</td><td>≈ 90 ms</td><td>not measured on a Pi</td><td>–</td></tr></table>')
    return out


def fig_paths():
    names = ["roc.png", "baseline_compare.png", "confusion.png", "scores.png", "cv_curves.png", "gradcam.png"]
    return [str(FIG_DIR / n) if (FIG_DIR / n).is_file() else None for n in names]


def pca_lab_png():
    p = FIG_DIR / "pca_lab.png"
    return str(p) if p.is_file() else None


# --------------------------------------------------------------------------- #
# static explanation
# --------------------------------------------------------------------------- #
PIPE = ["10 s sound clip", "16 kHz", "Spectrogram (sound image)", "MobileNetV2 (pre-trained)", "+ PCA scores", "Fault score", "Tuned threshold", "Dashboard: green / red"]
HOW_HTML = (
    '<h3>Pipeline</h3><div>' + '<span class="el-arrow">→</span>'.join(f'<span class="el-step">{s}</span>' for s in PIPE) + '</div>'
    '<h3>The problem</h3><div class="el-card">Factory and lab fans run all day; faults like worn bearings or blocked air go unnoticed until '
    'overheating and downtime. Contact vibration sensors cost about Rs. 15k each; a microphone costs about Rs. 150 and can listen 24/7. '
    'Only recordings of <b>healthy</b> fans are available for training, so the model must learn what normal sounds like.</div>'
    '<h3>What EchoLab does</h3><div class="el-card">'
    '<b>Full 10 s</b> of sound (not 0.3 s) → <b>sound image</b> (128 × 313 Mel spectrogram). '
    '<b>MobileNetV2</b> pretrained on ImageNet is fine-tuned on real healthy fans plus <b>injected fault patterns</b>; '
    '<b>PCA</b> models of healthy sound (on the spectrogram and on the network embedding) add independent scores; '
    'a <b>tuned threshold</b> turns the fused score into a green/red decision; <b>Grad-CAM</b> shows where the model listens; '
    'the model is <b>quantised</b> for a Raspberry Pi.</div>'
    '<h3>Fair testing: stratified 5-fold cross-validation</h3><div class="el-card">Healthy clips are split into 5 folds (about 802 train / 201 validation); '
    'every fold keeps the same healthy : pseudo-fault mix. The final test uses real faulty and healthy clips never seen in training. '
    'Result = average AUC ± spread over the 5 folds. Leave-one-out was skipped (about 1000 separate trainings), 2-fold (too little training data), '
    'time-series CV (clips are independent) and simple K-fold (can upset the healthy/faulty balance).</div>'
    '<h3>Bias and variance</h3><div class="el-card"><b>Bias</b>: model too simple → errors on both training and new data (under-fitting). '
    '<b>Variance</b>: model too sensitive to its training clips → fails on new data (over-fitting). '
    'We check the validation level, the train–validation gap and the spread across folds (Model &amp; Results).</div>'
    '<h3>Literature (as listed in the project deck; verify before citing)</h3>'
    '<table class="el-t"><tr><th>Paper</th><th>Idea used</th><th>AUC in deck</th></tr>'
    '<tr><td>Koizumi 2020</td><td>Baseline autoencoder on 0.3 s; sets the task and scoring</td><td>72.89 %</td></tr>'
    '<tr><td>Giri 2020</td><td>MobileNetV2 classifies machine IDs; beats autoencoders</td><td>88 %</td></tr>'
    '<tr><td>Ribeiro 2021</td><td>MFCC + CNN with sound augmentation</td><td>85 %</td></tr>'
    '<tr><td>Lopez 2022</td><td>Shrinks MobileNet to 5 MB for a Raspberry Pi</td><td>82 %</td></tr>'
    '<tr><td>Kumar 2024</td><td>Survey: id_00 cleanest; suggests Grad-CAM</td><td>–</td></tr></table>'
    '<h3>Future work from the deck: status</h3><table class="el-t"><tr><th>Item</th><th>Status</th></tr>'
    '<tr><td>Test on a real lab fan with a mic</td><td>Ready: record in the Analyze tab or upload; use own-fan calibration</td></tr>'
    '<tr><td>Add more faults (loose belt, blockage)</td><td>Extend <code>FAULT_KINDS</code> in fanlib.py and retrain</td></tr>'
    '<tr><td>Phone app + Telegram alerts</td><td>Telegram alerts in the Analyze tab; the site works on a phone; QR in the Share tab</td></tr>'
    '<tr><td>Extend to pumps and valves</td><td><code>python echolab_train.py --machine pump --ids 00</code> (also valve, slider)</td></tr></table>'
    '<h3>End users</h3><div class="el-card">College labs · Building HVAC fans · Factory cooling fans · Researchers</div>')


# --------------------------------------------------------------------------- #
# setup & train (runs echolab_train.py from the browser)
# --------------------------------------------------------------------------- #
def status_html():
    rows = []
    info_p = Path("dataset/cache/info.json")
    if info_p.is_file():
        i = json.loads(info_p.read_text())
        rows.append(("Dataset cache", f"{i['n']} clips ({i['n_normal']} healthy, {i['n_abnormal']} faulty), source: <b>{i['source']}</b>"))
    else:
        rows.append(("Dataset cache", "not downloaded yet"))
    b = STATE["bundle"]
    if b is not None:
        rows.append(("Trained model", f"loaded: trained {b.metrics['meta']['created']} on <b>{b.metrics['meta']['data_source']}</b> data"
                                       + (" — <b style='color:#b45309'>QUICK RUN (small subset)</b>" if b.quick else "")
                                       + (" — <b style='color:#dc2626'>OLD ARTIFICIAL-DATA MODEL</b>" if b.synthetic else "")))
    elif STATE["error"]:
        rows.append(("Trained model", f"found but unreadable: {html.escape(STATE['error'])}"))
    else:
        rows.append(("Trained model", "none yet"))
    try:
        import tensorflow as tf
        rows.append(("TensorFlow", f"{tf.__version__}, GPU: {'yes' if tf.config.list_physical_devices('GPU') else 'no (training on CPU is slow)'}"))
    except ImportError:
        rows.append(("TensorFlow", "not installed: needed for training (pip install tensorflow)"))
    return '<table class="el-t">' + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def train_stream(quick, allow_scratch, data_path):
    p = STATE["train"]
    if p is not None and p.poll() is None:
        yield "A training run is already in progress.", gr.update(), gr.update()
        return
    cmd = [sys.executable, "-u", "echolab_train.py"]
    if quick:
        cmd += ["--quick"]
    dp = (data_path or "").strip().strip('"')
    if dp:
        cmd += ["--archive", dp] if dp.lower().endswith(".zip") else ["--data-dir", dp]
    if allow_scratch:
        cmd += ["--allow-scratch"]
    buf = deque(maxlen=300)
    buf.append("$ " + " ".join(cmd))
    STATE["train"] = p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
    last = 0.0
    for line in p.stdout:
        buf.append(line.rstrip())
        if time.time() - last > 0.7:
            last = time.time()
            yield "\n".join(buf), gr.update(), gr.update()
    rc = p.wait()
    load_bundle()
    buf.append(f"\n--- finished (exit code {rc}) ---" + ("  Model loaded: see the Model & Results tab." if rc == 0 else
                                                          "  Training did not finish: read the message above."))
    yield "\n".join(buf), hero_html(), status_html()


def stop_training():
    p = STATE["train"]
    if p is not None and p.poll() is None:
        p.terminate()
        return "Stopping ..."
    return "Nothing is running."


def reload_model():
    load_bundle()
    return hero_html(), status_html(), results_html()


# --------------------------------------------------------------------------- #
# share / QR
# --------------------------------------------------------------------------- #
def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def make_qr(url):
    import qrcode
    return qrcode.make(url).convert("RGB")


def share_info(port=7860):
    local = f"http://{lan_ip()}:{port}"
    urls = [("Same Wi-Fi / network", local)]
    if STATE["share_url"]:
        urls.insert(0, ("Public link (Gradio share)", STATE["share_url"]))
    md = "\n".join(f"* **{k}:** {u}" for k, u in urls)
    try:
        return md, make_qr(urls[0][1])
    except Exception:
        return md, None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui(port=7860):
    kw = {} if GR_MAJOR >= 6 else dict(theme=make_theme(), css=CSS)
    with gr.Blocks(title="EchoLab: Fan Anomaly Detection", **kw) as demo:
        hero = gr.HTML(hero_html())
        cal_state = gr.State(None)
        with gr.Tabs():
            # ---------------------------------------------------------------- Analyze
            with gr.Tab("🎧 Analyze"):
                with gr.Row():
                    with gr.Column(scale=4):
                        audio = gr.Audio(label="Fan recording (upload or record)", sources=["upload", "microphone"], type="filepath")
                        go = gr.Button("Analyze", variant="primary", size="lg")
                        with gr.Accordion("Settings", open=False):
                            def_sc = STATE["bundle"].cfg.get("default_score", "cnn") if STATE["bundle"] else "cnn"
                            score_dd = gr.Dropdown(list(SCORE_CHOICES), value=SCORE_LABEL.get(def_sc, SCORE_LABEL["cnn"]), label="Score used for the verdict")
                            op_rd = gr.Radio(list(OPS), value=list(OPS)[0], label="Threshold rule")
                            offset = gr.Slider(-3, 3, value=0, step=0.1, label="Threshold offset (z-units; higher = fewer alarms)")
                            hop = gr.Radio([2.5, 5.0, 10.0], value=5.0, label="Window hop (s) for recordings longer than 10 s")
                        with gr.Accordion("Calibrate on my own fan (optional)", open=False):
                            gr.Markdown("Upload ≥ 30 s of **your fan running normally**. Scores are re-centred on it, which removes the "
                                        "shift caused by a different microphone, room or fan.")
                            cal_files = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Normal recordings")
                            cal_btn = gr.Button("Calibrate")
                            cal_msg = gr.Markdown("")
                        with gr.Accordion("Telegram alert (optional)", open=False):
                            tg_on = gr.Checkbox(False, label="Send an alert when a recording is faulty")
                            tg_token = gr.Textbox(label="Bot token", type="password")
                            tg_chat = gr.Textbox(label="Chat id")
                            tg_test = gr.Button("Send test message")
                            tg_msg = gr.Markdown("")
                    with gr.Column(scale=7):
                        verdict = gr.HTML()
                        scores = gr.HTML()
                        plot_t = gr.Plot(label="Timeline")
                        plot_s = gr.Plot(label="Spectrogram and Grad-CAM")
                        cam_note = gr.Markdown("")
                        plot_p = gr.Plot(label="PCA map")
                        table = gr.Dataframe(label="Per-window scores", interactive=False, wrap=True)
                        csv = gr.File(label="Download windows (CSV)")
                go.click(run_analysis, [audio, score_dd, op_rd, offset, hop, cal_state, tg_token, tg_chat, tg_on],
                         [verdict, scores, plot_t, plot_s, plot_p, table, csv, cam_note], api_name="analyze")
                cal_btn.click(run_calibration, [cal_files], [cal_state, cal_msg], api_name="calibrate")
                tg_test.click(lambda t, c: core.send_telegram(t, c, "EchoLab test message: alerts are working.")[1], [tg_token, tg_chat], [tg_msg])

            # ---------------------------------------------------------------- Batch
            with gr.Tab("📂 Batch"):
                bfiles = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Several recordings")
                bgo = gr.Button("Analyze all", variant="primary")
                bmsg = gr.Markdown("")
                btable = gr.Dataframe(label="Summary", interactive=False, wrap=True)
                bcsv = gr.File(label="Download summary (CSV)")
                bgo.click(run_batch, [bfiles, score_dd, op_rd, offset, hop, cal_state], [btable, bcsv, bmsg], api_name="batch")

            # ---------------------------------------------------------------- Results
            with gr.Tab("📊 Model & Results"):
                res_html = gr.HTML(results_html())
                gr.Markdown("### Figures")
                imgs = [gr.Image(label=n, show_label=True, interactive=False)
                        for n in ("ROC curves", "Measured AUC vs deck baseline", "Confusion matrices", "Score distributions",
                                  "Training curves (bias / variance)", "Grad-CAM examples")]
            # ---------------------------------------------------------------- PCA lab
            with gr.Tab("🧭 PCA Lab"):
                gr.Markdown(
                    "**What PCA does here.** Principal Component Analysis learns the main directions of variation of *healthy* sound. "
                    "Healthy sounds are reconstructed well from a few components; a sound with something new in it (a rattle, a click) is not. "
                    "The reconstruction error is a fault score that needs no faulty example. EchoLab applies PCA twice (on the spectrogram and on the "
                    "network's embedding) and fuses both with the MobileNetV2 fault score.")
                pca_img = gr.Image(label="Scree curve and embedding map from training", interactive=False)
                gr.Markdown("**Place your own recording on the map** (star):")
                pca_audio = gr.Audio(label="Recording", sources=["upload", "microphone"], type="filepath")
                pca_btn = gr.Button("Project")
                pca_plot = gr.Plot()
                pca_msg = gr.Markdown("")
                pca_btn.click(project_only, [pca_audio], [pca_plot, pca_msg], api_name="pca_map")
            # ---------------------------------------------------------------- How it works
            with gr.Tab("📖 How it works"):
                gr.HTML(HOW_HTML)
            # ---------------------------------------------------------------- Setup & Train
            with gr.Tab("⚙️ Setup & Train"):
                gr.Markdown("### Get the real model\nOne click downloads the **MIMII fan** data from Zenodo (dev_data_fan.zip is 1.4 GB), "
                            "extracts id_00, trains MobileNetV2 + PCA with 5-fold cross-validation and measures everything on the unseen test clips. "
                            "A GPU is strongly recommended (about 1 hour on a T4 versus many hours on a CPU).\n\n"
                            "The project uses **real recordings only**: if the download fails it stops and tells you how to fix it. "
                            "If you already have the dataset, type its path in the box below.")
                status = gr.HTML(status_html())
                with gr.Row():
                    allow_scratch = gr.Checkbox(False, label="If the ImageNet weights cannot be downloaded, train from scratch (lower accuracy)")
                    quick = gr.Checkbox(False, label="Quick check on a small subset (not the final model)")
                data_path = gr.Textbox(label="Already have the dataset? Path to dev_data_fan.zip or to the extracted folder (optional)",
                                       placeholder="e.g. D:/datasets/dev_data_fan.zip")
                with gr.Row():
                    train_btn = gr.Button("Download real data and train the model", variant="primary")
                    stop_btn = gr.Button("Stop")
                    reload_btn = gr.Button("Reload model")
                log = gr.Textbox(label="Training log", lines=18, max_lines=18, autoscroll=True, interactive=False)
                train_btn.click(train_stream, [quick, allow_scratch, data_path], [log, hero, status], api_name="train")
                stop_btn.click(stop_training, None, [log])
                reload_btn.click(reload_model, None, [hero, status, res_html])
            # ---------------------------------------------------------------- Share
            with gr.Tab("📱 Share"):
                share_md = gr.Markdown("")
                qr_img = gr.Image(label="Scan to open on a phone", interactive=False, width=300)
                gr.Markdown("Start with `python app.py --share` for a public link that works from anywhere.")
        demo.load(lambda: (hero_html(), results_html(), status_html(), *fig_paths(), pca_lab_png(), *share_info(port)),
                  None, [hero, res_html, status, *imgs, pca_img, share_md, qr_img])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices on your network")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    b = load_bundle()
    print("EchoLab model:", ("loaded" + (" (QUICK run)" if b.quick else "")) if b else
          "none yet: open the Setup & Train tab")
    demo = build_ui(a.port)
    demo.queue()
    lk = dict(server_name=a.host, server_port=a.port, share=a.share, inbrowser=not a.no_browser, prevent_thread_lock=True)
    if GR_MAJOR >= 6:
        lk.update(theme=make_theme(), css=CSS)
    demo.launch(**lk)
    STATE["share_url"] = getattr(demo, "share_url", None)
    url = STATE["share_url"] or f"http://{lan_ip()}:{a.port}"
    print(f"\nEchoLab is running:  http://localhost:{a.port}   |   network: http://{lan_ip()}:{a.port}"
          + (f"   |   public: {STATE['share_url']}" if STATE["share_url"] else ""))
    try:
        import qrcode
        q = qrcode.QRCode(border=1); q.add_data(url); q.make(fit=True); q.print_ascii(invert=True)
        make_qr(url).save("echolab_qr.png")
        print("QR code saved to echolab_qr.png")
    except Exception:
        pass
    demo.block_thread()


if __name__ == "__main__":
    main()
