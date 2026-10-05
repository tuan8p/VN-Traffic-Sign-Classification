from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure local module lookup
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference import TrafficSignPredictor

# Page Setup
st.set_page_config(
    page_title="VN Traffic Sign Analytics & Benchmark",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Engine Initialization
@st.cache_resource
def get_engine() -> TrafficSignPredictor:
    return TrafficSignPredictor()

engine = get_engine()

# Theme-consistent color palette
PALETTE = {
    "Deep Learning": "#2563EB",       # Royal Blue
    "SVM Classifier": "#059669",      # Emerald Green
    "Gradient Boosting": "#D97706"    # Amber
}

# ==================== CHART GENERATORS ====================

def plot_single_pipeline_probs(predictions: dict[str, float]) -> go.Figure:
    """Elegant horizontal bar chart for Single Pipeline Top-K."""
    df = pd.DataFrame(list(predictions.items()), columns=["Class", "Probability"])
    df = df.sort_values(by="Probability", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["Probability"],
        y=df["Class"],
        orientation="h",
        marker=dict(
            color=df["Probability"],
            colorscale="Blues",
            line=dict(color="rgba(37, 99, 235, 0.6)", width=1)
        ),
        text=[f"{val*100:.1f}%" for val in df["Probability"]],
        textposition="outside"
    ))
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=260,
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False, range=[0, 1.15]),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12)
    )
    return fig

def plot_benchmark_grouped_bar(results_dict: dict[str, dict]) -> go.Figure:
    """Grouped bar chart comparing confidence on top candidate across models."""
    records = []
    for model_name, res in results_dict.items():
        for rank, (cls_name, prob) in enumerate(res["predictions"].items()):
            records.append({
                "Model": model_name,
                "Candidate": f"Top-{rank+1}: {cls_name.split(' - ')[0]}",
                "Confidence": prob
            })
    df = pd.DataFrame(records)

    fig = px.bar(
        df,
        x="Candidate",
        y="Confidence",
        color="Model",
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
    """Horizontal lollipop/bar chart for inference speed comparison."""
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

def plot_radar_summary() -> go.Figure:
    """Spider radar chart evaluating comprehensive architectural strengths."""
    categories = ["Accuracy", "Inference Speed", "Noise Resilience", "Low Memory", "Training Speed"]
    fig = go.Figure()

    # Characteristic profiles for report insights
    fig.add_trace(go.Scatterpolar(
        r=[94, 70, 92, 60, 65],
        theta=categories,
        fill="toself",
        name="Deep Learning",
        line=dict(color=PALETTE["Deep Learning"])
    ))
    fig.add_trace(go.Scatterpolar(
        r=[82, 95, 74, 90, 85],
        theta=categories,
        fill="toself",
        name="SVM Classifier",
        line=dict(color=PALETTE["SVM Classifier"])
    ))
    fig.add_trace(go.Scatterpolar(
        r=[86, 88, 78, 85, 80],
        theta=categories,
        fill="toself",
        name="Gradient Boosting",
        line=dict(color=PALETTE["Gradient Boosting"])
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showline=False, gridcolor="rgba(128,128,128,0.2)"),
            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)")
        ),
        margin=dict(l=40, r=40, t=25, b=25),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        font=dict(family="Inter, sans-serif", size=11)
    )
    return fig


# ==================== HEADER ====================
st.header("Vietnamese Traffic Sign Classification System", divider="gray", anchor=False)
st.caption("Benchmark & Inference Hub: Deep Learning vs. Classical Machine Learning")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.subheader("Configuration", anchor=False)

    execution_mode = st.radio(
        "Execution Mode",
        ["Benchmark (All 3 Pipelines)", "Single Pipeline"],
        index=0
    )

    pipeline_map = {
        "Deep Learning (EfficientNet/ResNet)": "dl",
        "SVM Classifier (HOG + PCA)": "svm",
        "Gradient Boosting (XGBoost/LightGBM)": "boosting"
    }

    selected_pipeline_key = "dl"
    if execution_mode == "Single Pipeline":
        selected_display = st.selectbox("Target Architecture", list(pipeline_map.keys()))
        selected_pipeline_key = pipeline_map[selected_display]

    top_k = st.slider("Top-K Candidates", min_value=1, max_value=8, value=5)

    st.divider()
    st.subheader("Weights Status", anchor=False)
    for p_name, p_key in [("Deep Learning", "dl"), ("SVM Classifier", "svm"), ("Gradient Boosting", "boosting")]:
        status_txt = "Loaded (.onnx/.pth)" if engine.model_status[p_key] else "Simulated (Mock Ready)"
        st.caption(f"• **{p_name}**: `{status_txt}`")

    st.caption("Source: `saved_models/`")

# ==================== WORKSPACE ====================
col_input, col_output = st.columns([1, 1.3], gap="medium")

with col_input:
    with st.container(border=True):
        st.subheader("Data Acquisition", anchor=False)

        tab_upload, tab_presets, tab_camera = st.tabs(["Upload File", "Preset Gallery", "Live Camera"])
        input_image = None

        with tab_upload:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
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
            camera_file = st.camera_input("Capture sign via camera", label_visibility="collapsed")
            if camera_file:
                input_image = Image.open(camera_file)

        if input_image:
            st.image(input_image, caption="Input Tensor Preview", use_container_width=True)
            run_btn = st.button("Run Classification", type="primary", use_container_width=True)
        else:
            st.info("Please provide an image using one of the input sources above.")
            run_btn = False

with col_output:
    with st.container(border=True):
        st.subheader("Inference & Benchmark Analytics", anchor=False)

        if run_btn and input_image:
            if execution_mode == "Single Pipeline":
                # Single Pipeline Flow
                with st.spinner("Executing model pipeline..."):
                    res = engine.predict(input_image, selected_pipeline_key, top_k)

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Class", res["top_class"].split(" - ")[0])
                m2.metric("Confidence", f"{res['confidence']*100:.1f}%")
                m3.metric("Latency", f"{res['latency_ms']} ms")

                st.caption(f"Full Descriptor: **{res['top_class']}**")
                st.write("**Top-K Confidence Distribution:**")
                st.plotly_chart(plot_single_pipeline_probs(res["predictions"]), use_container_width=True)

            else:
                # 3-Pipeline Comparative Benchmark Flow
                pipelines_to_compare = [
                    ("Deep Learning", "dl"),
                    ("SVM Classifier", "svm"),
                    ("Gradient Boosting", "boosting")
                ]

                results = {}
                latencies = {}

                with st.spinner("Benchmarking all 3 pipelines simultaneously..."):
                    for display_name, key in pipelines_to_compare:
                        res = engine.predict(input_image, key, top_k=3)
                        results[display_name] = res
                        latencies[display_name] = res["latency_ms"]

                # Metric Cards Comparison
                m_cols = st.columns(3)
                for idx, (display_name, _) in enumerate(pipelines_to_compare):
                    with m_cols[idx]:
                        with st.container(border=True):
                            st.caption(f"**{display_name}**")
                            st.metric("Prediction", results[display_name]["top_class"].split(" - ")[0])
                            st.metric("Confidence", f"{results[display_name]['confidence']*100:.1f}%")
                            st.metric("Latency", f"{results[display_name]['latency_ms']} ms")

                # Interactive Analytical Visualizations
                tab_comp1, tab_comp2, tab_comp3 = st.tabs([
                    "Confidence Distribution",
                    "Latency Benchmark",
                    "Architectural Radar Profile"
                ])

                with tab_comp1:
                    st.caption("Side-by-side Top-3 prediction confidence comparison:")
                    st.plotly_chart(plot_benchmark_grouped_bar(results), use_container_width=True)

                with tab_comp2:
                    st.caption("Inference speed profiling on CPU (lower is faster):")
                    st.plotly_chart(plot_latency_comparison(latencies), use_container_width=True)

                with tab_comp3:
                    st.caption("Multi-dimensional architectural trade-offs:")
                    st.plotly_chart(plot_radar_summary(), use_container_width=True)

        else:
            st.info("Trigger 'Run Classification' on the left panel to generate comparative analysis.")