from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference import TrafficSignPredictor

st.set_page_config(
    page_title="VN Traffic Sign Benchmark Console",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_engine() -> TrafficSignPredictor:
    return TrafficSignPredictor()

engine = get_engine()

PALETTE = {
    "Deep Learning (EfficientNetV2-B0)": "#2563EB",
    "SVM Classifier (HOG + PCA)": "#059669",
    "Gradient Boosting (XGBoost/LightGBM)": "#D97706"
}

def plot_benchmark_grouped_bar(results_dict: dict[str, dict]) -> go.Figure:
    records = []
    for model_name, res in results_dict.items():
        for rank, (cls_name, prob) in enumerate(res["predictions"].items()):
            records.append({
                "Architecture": model_name,
                "Candidate": f"Top-{rank+1}: {cls_name.split(' - ')[0]}",
                "Confidence": prob
            })
    df = pd.DataFrame(records)

    fig = px.bar(
        df,
        x="Candidate",
        y="Confidence",
        color="Architecture",
        barmode="group",
        color_discrete_map=PALETTE,
        text_auto=".1%"
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", title="Confidence Score", range=[0, 1.15]),
        font=dict(family="Inter, sans-serif", size=12)
    )
    return fig

def plot_latency_comparison(latencies: dict[str, float]) -> go.Figure:
    models = list(latencies.keys())
    values = list(latencies.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=models,
        orientation="h",
        marker=dict(
            color=[PALETTE.get(m, "#64748B") for m in models],
            line=dict(width=1, color="rgba(255,255,255,0.4)")
        ),
        text=[f"{v:.1f} ms" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", title="Latency (ms) - Lower is better"),
        yaxis=dict(showgrid=False),
        font=dict(family="Inter, sans-serif", size=12)
    )
    return fig

# ==================== HEADER ====================
st.header("Vietnamese Traffic Sign Classification System", divider="gray", anchor=False)
st.caption("Phase: Model Evaluation & Multi-Pipeline Benchmark | Role E Demo")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.subheader("System Status", anchor=False)

    for p_name, p_key in [("Deep Learning", "dl"), ("SVM Classifier", "svm"), ("Gradient Boosting", "boosting")]:
        status_txt = "Active (Weights Loaded)" if engine.model_status[p_key] else "Simulated (Mock Ready)"
        st.write(f"• **{p_name}**: `{status_txt}`")

    st.divider()
    top_k = st.slider("Top-K Candidates per Model", min_value=1, max_value=5, value=3)
    st.caption("Artifacts Storage: `saved_models/`")

# ==================== WORKSPACE ====================
col_input, col_output = st.columns([1, 1.4], gap="medium")

with col_input:
    with st.container(border=True):
        st.subheader("Input Acquisition", anchor=False)
        st.info("Input requirement: Standard **RGB image cropped tightly to the traffic sign** (exclude broad backgrounds).")

        tab_upload, tab_presets, tab_camera = st.tabs(["Upload File", "Preset Gallery", "Live Camera"])
        input_image = None

        with tab_upload:
            uploaded_file = st.file_uploader("Select cropped traffic sign image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                input_image = Image.open(uploaded_file)

        with tab_presets:
            preset_samples = {
                "Sample 01 - No Entry (P.102)": (220, 38, 38),
                "Sample 02 - Speed Limit 60 (P.127)": (234, 88, 12),
                "Sample 03 - Pedestrian Crossing (W.224)": (37, 99, 235)
            }
            selected_preset = st.selectbox("Select Preset Image", list(preset_samples.keys()))
            sample_file = f"demo/sample_images/{selected_preset.split(' ')[0].lower()}.jpg"
            if os.path.exists(sample_file):
                input_image = Image.open(sample_file)
            else:
                input_image = Image.new("RGB", (256, 256), color=preset_samples[selected_preset])

        with tab_camera:
            camera_file = st.camera_input("Capture cropped sign via camera", label_visibility="collapsed")
            if camera_file:
                input_image = Image.open(camera_file)

        if input_image:
            st.image(input_image, caption="Current Input Tensor (RGB)", use_container_width=True)
            run_btn = st.button("Run Multi-Pipeline Inference", type="primary", use_container_width=True)
        else:
            run_btn = False

with col_output:
    with st.container(border=True):
        st.subheader("3-Pipeline Performance Comparison", anchor=False)

        if run_btn and input_image:
            pipelines_to_compare = [
                ("Deep Learning (EfficientNetV2-B0)", "dl"),
                ("SVM Classifier (HOG + PCA)", "svm"),
                ("Gradient Boosting (XGBoost/LightGBM)", "boosting")
            ]

            results = {}
            latencies = {}

            with st.spinner("Processing image and evaluating all 3 pipelines simultaneously..."):
                for display_name, key in pipelines_to_compare:
                    res = engine.predict(input_image, key, top_k=top_k)
                    results[display_name] = res
                    latencies[display_name] = res["latency_ms"]

            # Bố trí 3 vùng/cột riêng biệt cho 3 mô hình như yêu cầu cuộc họp
            cols = st.columns(3)
            for idx, (display_name, key) in enumerate(pipelines_to_compare):
                with cols[idx]:
                    with st.container(border=True):
                        st.markdown(f"**{display_name.split(' ')[0]} {display_name.split(' ')[1]}**")
                        status_tag = "⚡ Real Model" if not results[display_name].get("is_mock", True) else "🧪 Simulated"
                        st.caption(status_tag)
                        st.metric("Top-1 Class", results[display_name]["top_class"].split(" - ")[0])
                        st.metric("Confidence", f"{results[display_name]['confidence']*100:.1f}%")
                        st.metric("Latency", f"{results[display_name]['latency_ms']} ms")

            st.divider()

            # Biểu đồ so sánh chi tiết
            tab_chart1, tab_chart2 = st.tabs(["Top-K Confidence Comparison", "CPU Latency Benchmark"])
            with tab_chart1:
                st.caption("Confidence scores across top candidate classes:")
                st.plotly_chart(plot_benchmark_grouped_bar(results), use_container_width=True)

            with tab_chart2:
                st.caption("Inference speed profiling on CPU (lower is faster):")
                st.plotly_chart(plot_latency_comparison(latencies), use_container_width=True)

        else:
            st.info("Upload or select a cropped traffic sign image and click 'Run Multi-Pipeline Inference' to benchmark.")