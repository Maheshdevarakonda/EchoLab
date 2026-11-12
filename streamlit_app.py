#!/usr/bin/env python3
"""
streamlit_app.py -- EchoLab Streamlit web app for Streamlit Community Cloud.
"""
from __future__ import annotations

import html
import io
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import echolab_core as core
from echolab_core import SCORE_LABEL, SCORES, Bundle

# Set page config
st.set_page_config(
    page_title="EchoLab: Fan Anomaly Detection",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark theme CSS
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hero Banner */
    .el-hero {
        background: linear-gradient(135deg, #0f1b3d 0%, #16275a 100%);
        border: 1px solid #1e3a8a;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .el-hero .brand {
        letter-spacing: .35em;
        font-size: .8rem;
        color: #60a5fa;
        font-weight: 700;
        text-transform: uppercase;
    }
    .el-hero h1 {
        margin: .25em 0 .15em 0;
        font-size: 2.2rem;
        color: #ffffff;
        font-weight: 800;
    }
    .el-hero p {
        margin: 0;
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* KPI Grid */
    .el-kpis {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-top: 18px;
    }
    .el-kpi {
        background: rgba(255, 255, 255, .05);
        border: 1px solid rgba(255, 255, 255, .1);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
    }
    .el-kpi b {
        display: block;
        font-size: 1.5rem;
        color: #38bdf8;
        font-weight: 700;
    }
    .el-kpi span {
        font-size: .75rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    
    /* Verdict Cards */
    .el-verdict-healthy {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border: 1px solid #059669;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0;
        color: white;
    }
    .el-verdict-faulty {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        border: 1px solid #dc2626;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0;
        color: white;
    }
    .el-verdict-possible {
        background: linear-gradient(135deg, #78350f 0%, #b45309 100%);
        border: 1px solid #d97706;
        border-radius: 14px;
        padding: 20px 24px;
        margin: 16px 0;
        color: white;
    }
    
    /* Custom table */
    .el-table {
        width: 100%;
        border-collapse: collapse;
        background: #0f172a;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #1e293b;
        margin: 14px 0;
    }
    .el-table th {
        background: #1e293b;
        color: #38bdf8;
        padding: 10px 14px;
        text-align: left;
        font-size: 0.85rem;
    }
    .el-table td {
        padding: 10px 14px;
        border-top: 1px solid #334155;
        color: #f1f5f9;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    if not Bundle.exists():
        return None
    return Bundle()


def pc(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f} %"


def render_hero(bundle):
    if bundle is not None:
        m, d = bundle.metrics, bundle.metrics["data"]
        o = m["final"]["operating"]["fused"]["roc_crossfit"]
        items = [
            (d["n_train_healthy"], "healthy training clips"),
            (d["n_test_faulty"], "faulty test clips"),
            (f"{m['model']['tflite_quant_mb']:.1f} MB", "quantised model"),
            (pc(m["final"]["scores"]["fused"]["auc"]), "test AUC"),
            (pc(o["accuracy"]), "accuracy"),
            (pc(o["balanced_accuracy"]), "balanced accuracy"),
            (f"{o['f1']:.3f}", "F1 score"),
        ]
        kpi_html = "".join(f'<div class="el-kpi"><b>{v}</b><span>{k}</span></div>' for v, k in items)
    else:
        kpi_html = ""

    st.markdown(f"""
    <div class="el-hero">
        <div class="brand">EchoLab · Edge AI</div>
        <h1>Fan Anomaly Detection</h1>
        <p>Industrial anomaly detection from sound using MobileNetV2 + PCA · MIMII Fan id_00 dataset</p>
        <div class="el-kpis">{kpi_html}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    bundle = get_bundle()
    render_hero(bundle)

    if bundle is None:
        st.error("Model bundle not found. Please verify models/echolab/ exists.")
        return

    tab_analyze, tab_results, tab_pca, tab_about = st.tabs([
        "🎧 Analyze Audio", "📊 Model & Results", "🧭 PCA Lab", "📖 How It Works"
    ])

    with tab_analyze:
        col_left, col_right = st.columns([1, 1.3], gap="large")

        with col_left:
            st.subheader("1. Fan Recording")
            uploaded_file = st.file_uploader(
                "Upload a fan recording (WAV format, 16 kHz recommended)",
                type=["wav", "mp3", "ogg", "flac"]
            )

            # Sample audio picker if user hasn't uploaded
            st.markdown("Or choose a sample recording:")
            sample_dir_norm = Path("dataset/fan/test/normal")
            sample_dir_anom = Path("dataset/fan/test/abnormal")
            samples = {}
            if sample_dir_norm.is_dir():
                for f in sorted(sample_dir_norm.glob("*.wav"))[:3]:
                    samples[f"Healthy Sample: {f.name}"] = f
            if sample_dir_anom.is_dir():
                for f in sorted(sample_dir_anom.glob("*.wav"))[:3]:
                    samples[f"Faulty Sample: {f.name}"] = f

            selected_sample = None
            if samples:
                choice = st.selectbox("Select a benchmark sample", ["(None - use uploaded file)"] + list(samples.keys()))
                if choice != "(None - use uploaded file)":
                    selected_sample = samples[choice]

            audio_to_play = uploaded_file or selected_sample
            if audio_to_play:
                st.audio(audio_to_play)

            with st.expander("⚙️ Inference Settings", expanded=False):
                def_sc = bundle.cfg.get("default_score", "cnn")
                score_opts = ["MobileNetV2 fault score", "EchoLab fused score", "PCA on spectrogram", "PCA on embedding"]
                score_map = {
                    "MobileNetV2 fault score": "cnn",
                    "EchoLab fused score": "fused",
                    "PCA on spectrogram": "spca",
                    "PCA on embedding": "epca"
                }
                score_label = st.selectbox("Score used for verdict", score_opts, index=0)
                score_key = score_map[score_label]

                op_rule = st.radio("Threshold rule", ["ROC-best threshold (tuned)", "Label-free threshold (unsupervised)"], index=0)
                op_key = "roc" if "ROC-best" in op_rule else "label_free"

                thr_offset = st.slider("Threshold offset (z-units)", -3.0, 3.0, 0.0, 0.1)
                hop_s = st.select_slider("Window hop (seconds)", options=[2.5, 5.0, 10.0], value=5.0)

            run_btn = st.button("🚀 Analyze Recording", type="primary", use_container_width=True)

        with col_right:
            st.subheader("2. Diagnostic Verdict & Plots")
            if run_btn:
                if not uploaded_file and not selected_sample:
                    st.warning("Please upload an audio file or select a sample recording above.")
                else:
                    with st.spinner("Analyzing audio with edge TFLite..."):
                        # Save temp audio file
                        if uploaded_file:
                            suffix = Path(uploaded_file.name).suffix
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(uploaded_file.read())
                                audio_path = tmp.name
                                file_name = uploaded_file.name
                        else:
                            audio_path = str(selected_sample)
                            file_name = selected_sample.name

                        thr = bundle.threshold(score_key, op_key) + float(thr_offset)
                        try:
                            res = core.analyze_audio(
                                bundle,
                                audio_path,
                                name=file_name,
                                score=score_key,
                                operating=op_key,
                                thr=thr,
                                hop_s=float(hop_s)
                            )
                        finally:
                            if uploaded_file and os.path.exists(audio_path):
                                os.unlink(audio_path)

                        # Verdict Banner
                        verdict = res["verdict"]
                        pz = "–" if res["peak_z"] is None else f"{res['peak_z']:.2f}"
                        if verdict == "HEALTHY":
                            v_class = "el-verdict-healthy"
                            v_icon = "✔"
                        elif verdict == "FAULTY":
                            v_class = "el-verdict-faulty"
                            v_icon = "✖"
                        else:
                            v_class = "el-verdict-possible"
                            v_icon = "⚠"

                        st.markdown(f"""
                        <div class="{v_class}">
                            <h2 style="margin:0; font-size: 2.2rem; font-weight:800;">{v_icon} {verdict}</h2>
                            <p style="margin: 6px 0 0 0; opacity: 0.95; font-size: 0.95rem;">
                                {res['n_flagged']} of {res['n_valid']} analysed 10-second windows flagged &nbsp;·&nbsp;
                                Peak z = {pz} &nbsp;·&nbsp; Threshold z = {res['threshold']:.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Scores Summary Table
                        score_rows = ""
                        for s_name in SCORES:
                            s = res["per_score"][s_name]
                            flag = s["flagged_fraction"]
                            col = "#ef4444" if flag >= 0.5 else "#f59e0b" if flag > 0 else "#22c55e"
                            mz = "–" if s["mean_z"] is None else f"{s['mean_z']:.2f}"
                            xz = "–" if s["max_z"] is None else f"{s['max_z']:.2f}"
                            bold = "font-weight:700;" if s_name == score_key else ""
                            score_rows += f"""
                            <tr style="{bold}">
                                <td>{SCORE_LABEL[s_name]}</td>
                                <td>{mz}</td>
                                <td>{xz}</td>
                                <td>{s['threshold']:.2f}</td>
                                <td style="color:{col}; font-weight:700;">{100*flag:.0f} %</td>
                            </tr>
                            """

                        st.markdown(f"""
                        <table class="el-table">
                            <thead>
                                <tr><th>Score</th><th>Mean z</th><th>Peak z</th><th>Threshold z</th><th>Windows Flagged</th></tr>
                            </thead>
                            <tbody>{score_rows}</tbody>
                        </table>
                        """, unsafe_allow_html=True)

                        # Plots
                        fig_t = core.fig_timeline(res)
                        fig_s = core.fig_spectrogram(res, cam=None)
                        fig_p = core.fig_pca_map(bundle, res["best"]["emb"] if res["best"] is not None else None)

                        st.pyplot(fig_t)
                        st.pyplot(fig_s)
                        st.pyplot(fig_p)

                        # Windows Table & Download
                        df_win = res["windows"].copy()
                        df_win["end_s"] = df_win["start_s"] + 10
                        show_df = df_win[["start_s", "end_s", "rms", "z_cnn", "z_spca", "z_epca", "z_fused", "state"]].round(3)
                        st.dataframe(show_df, use_container_width=True)

                        csv_bytes = show_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Window Scores (CSV)",
                            data=csv_bytes,
                            file_name=f"echolab_{Path(file_name).stem}.csv",
                            mime="text/csv"
                        )
            else:
                st.info("Upload or select an audio file on the left and click **Analyze Recording** to inspect.")

    with tab_results:
        st.subheader("Model Validation & Metrics")
        fig_dir = Path("models/echolab/figures")
        if fig_dir.is_dir():
            c1, c2 = st.columns(2)
            with c1:
                if (fig_dir / "roc.png").exists():
                    st.image(str(fig_dir / "roc.png"), caption="ROC Curves (Supervised 5-Fold Cross-Validation)")
                if (fig_dir / "confusion.png").exists():
                    st.image(str(fig_dir / "confusion.png"), caption="Confusion Matrix")
                if (fig_dir / "gradcam.png").exists():
                    st.image(str(fig_dir / "gradcam.png"), caption="Grad-CAM Acoustic Attention (Fault Region)")
            with c2:
                if (fig_dir / "baseline_compare.png").exists():
                    st.image(str(fig_dir / "baseline_compare.png"), caption="Measured AUC vs Baseline")
                if (fig_dir / "scores.png").exists():
                    st.image(str(fig_dir / "scores.png"), caption="Anomaly Score Distributions")
                if (fig_dir / "cv_curves.png").exists():
                    st.image(str(fig_dir / "cv_curves.png"), caption="Cross-Validation Learning Curves")

    with tab_pca:
        st.subheader("Principal Component Analysis (PCA) Lab")
        st.markdown(
            "PCA learns the primary variance manifolds of **healthy fan sound**. "
            "When sound containing an abnormal defect (rattle, clicking, imbalance) is passed, "
            "it produces higher reconstruction errors. EchoLab fuses both spectrogram PCA and network embedding PCA."
        )
        pca_fig = Path("models/echolab/figures/pca_lab.png")
        if pca_fig.exists():
            st.image(str(pca_fig), caption="PCA Scree Curve and 2D Latent Projection Map", use_container_width=True)

    with tab_about:
        st.subheader("How EchoLab Works")
        st.markdown("""
        ### Industrial Edge Anomaly Detection Architecture
        
        1. **Audio Preprocessing:**
           * 10-second continuous windows at 16 kHz sampling rate.
           * Log-Mel Spectrogram extraction (128 Mel bands, 313 time frames).
           
        2. **Dual-Path Feature Extraction:**
           * **Transfer Learning:** MobileNetV2 pretrained on ImageNet, adapted to acoustic spectrograms.
           * **Unsupervised PCA:** Spectrogram PCA (80 components) + Embedding PCA (18 components).
           
        3. **Edge Optimization:**
           * Quantized TensorFlow Lite model (**0.58 MB** total size).
           * Sub-second execution on edge CPU devices.
           
        4. **Scoring & Decision Rule:**
           * Z-score normalization against healthy baseline sound distribution.
           * Calibrated ROC threshold yielding **88.7% AUC** and **86.5% accuracy**.
        """)


if __name__ == "__main__":
    main()
