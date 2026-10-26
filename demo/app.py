from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# =========================================================
# CẤU HÌNH ỨNG DỤNG
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference import TrafficSignPredictor

st.set_page_config(
    page_title="Hệ thống nhận diện biển báo giao thông",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

SAMPLE_DIR = CURRENT_DIR / "sample_images"

# Danh sách ứng viên tìm class_table.csv (ưu tiên saved_models/)
CLASS_TABLE_CANDIDATES = [
    CURRENT_DIR / "saved_models" / "class_table.csv",
    CURRENT_DIR.parent / "saved_models" / "class_table.csv",
    CURRENT_DIR / "class_table.csv",
    CURRENT_DIR.parent / "class_table.csv",
]
CLASS_TABLE_PATH = next(
    (path for path in CLASS_TABLE_CANDIDATES if path.exists()),
    CLASS_TABLE_CANDIDATES[0],
)

# Cấu hình danh sách 3 mô hình (Tên chính, Kiến trúc phụ, Key)
PIPELINES = [
    ("Học sâu", "(EfficientNetV2-B0)", "dl"),
    ("Bộ phân loại SVM", "(HOG + LBP)", "svm"),
    ("Gradient Boosting", "(LightGBM)", "boosting"),
]

PALETTE = {
    "Học sâu (EfficientNetV2-B0)": "#2563EB",
    "Bộ phân loại SVM (HOG + LBP)": "#059669",
    "Gradient Boosting (LightGBM)": "#D97706",
}

# =========================================================
# CSS: CĂN CHỈNH GIAO DIỆN VÀ ĐỒNG NHẤT CÁC CỘT
# =========================================================

st.markdown(
    """
    <style>
    /* Ẩn sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    div[data-testid="collapsedControl"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 98% !important;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }

    /* Ghim Mục 1 (Cột nhập ảnh) bám theo màn hình khi cuộn */
    @media (min-width: 900px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            position: -webkit-sticky !important;
            position: sticky !important;
            top: 1.5rem !important;
            z-index: 10 !important;
            align-self: flex-start !important;
        }
    }

    /* Vùng tiêu đề mô hình căn giữa và đồng nhất chiều cao */
    .model-header-box {
        text-align: center;
        min-height: 4.8rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .model-main-title {
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.3;
    }

    .model-sub-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #4B5563;
        margin-top: 2px;
    }

    .model-status-tag {
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 4px;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.035em;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 0.25rem;
    }

    .sign-code {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
        min-height: 2rem;
    }

    /* Chiều cao cố định vùng tên tiếng Việt để gióng thẳng các mục bên dưới */
    .sign-name {
        min-height: 3.2rem;
        font-size: 0.95rem;
        line-height: 1.4;
        color: #4B5563;
        display: flex;
        align-items: flex-start;
        margin-top: 0.2rem;
    }

    .metric-container {
        min-height: 3.6rem;
    }

    .metric-number {
        font-size: 1.65rem;
        font-weight: 600;
        line-height: 1.2;
    }

    @media (max-width: 899px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
            position: static !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TẢI MÔ HÌNH VÀ BẢNG TÊN BIỂN BÁO
# =========================================================

@st.cache_resource
def get_engine() -> TrafficSignPredictor:
    return TrafficSignPredictor()


@st.cache_data
def load_class_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ("sign_code", "name_vi"):
        df[column] = df[column].fillna("").astype(str).str.strip()
    df["class_id"] = df["class_id"].astype(str).str.strip()
    return df


engine = get_engine()

class_table = pd.DataFrame(columns=["class_id", "sign_code", "name_vi"])
class_table_error = None

if CLASS_TABLE_PATH.exists():
    try:
        class_table = load_class_table(str(CLASS_TABLE_PATH))
    except Exception as exc:
        class_table_error = str(exc)
else:
    class_table_error = f"Không tìm thấy class_table.csv tại {CLASS_TABLE_PATH}."


# =========================================================
# CHUẨN HÓA TÊN BIỂN BÁO
# =========================================================

def get_sign_info(raw_label: object) -> dict[str, str]:
    label = str(raw_label).strip()
    if not label or class_table.empty:
        return {"code": label or "Chưa rõ", "name": "Chưa có thông tin", "display": label or "Chưa rõ"}

    candidates = [label]
    parts = re.split(r"\s+[-–—]\s+", label, maxsplit=1)
    if len(parts) == 2:
        candidates = [parts[0].strip(), parts[1].strip(), label]

    for candidate in candidates:
        code_match = class_table[class_table["sign_code"].str.casefold() == candidate.casefold()]
        if not code_match.empty:
            row = code_match.iloc[0]
            code = str(row["sign_code"]).strip()
            name = str(row["name_vi"]).strip()
            return {"code": code, "name": name, "display": f"{code} — {name}"}

        name_match = class_table[class_table["name_vi"].str.casefold() == candidate.casefold()]
        if not name_match.empty:
            row = name_match.iloc[0]
            code = str(row["sign_code"]).strip()
            name = str(row["name_vi"]).strip()
            return {"code": code, "name": name, "display": f"{code} — {name}"}

    numeric_match = re.fullmatch(r"(?:class[_ -]?)?(\d+)", label, re.I)
    if numeric_match:
        id_match = class_table[class_table["class_id"] == numeric_match.group(1)]
        if not id_match.empty:
            row = id_match.iloc[0]
            code = str(row["sign_code"]).strip()
            name = str(row["name_vi"]).strip()
            return {"code": code, "name": name, "display": f"{code} — {name}"}

    return {"code": label, "name": "Chưa tìm thấy tên trong từ điển", "display": label}


# =========================================================
# BIỂU ĐỒ
# =========================================================

def plot_confidence_comparison(results: dict[str, dict]) -> go.Figure:
    records = []
    for model_name, result in results.items():
        for rank, (label, confidence) in enumerate(result.get("predictions", {}).items(), start=1):
            sign = get_sign_info(label)
            records.append({
                "Mô hình": model_name,
                "Biển báo": f"Top-{rank}: {sign['display']}",
                "Độ tin cậy": float(confidence),
                "Thứ hạng": rank,
            })

    df = pd.DataFrame(records)
    fig = px.bar(
        df,
        x="Biển báo",
        y="Độ tin cậy",
        color="Mô hình",
        barmode="group",
        color_discrete_map=PALETTE,
        text=df["Độ tin cậy"].map(lambda val: f"{val:.1%}"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=45, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Biển báo dự đoán", showgrid=False, tickangle=-25),
        yaxis=dict(title="Độ tin cậy", range=[0, 1.18], tickformat=".0%", gridcolor="rgba(128,128,128,0.15)"),
        font=dict(family="Arial, sans-serif", size=11),
    )
    return fig


def plot_latency_comparison(latencies: dict[str, float]) -> go.Figure:
    models = list(latencies.keys())
    values = list(latencies.values())
    fig = go.Figure(
        go.Bar(
            x=values,
            y=models,
            orientation="h",
            marker_color=[PALETTE.get(m, "#64748B") for m in models],
            text=[f"{v:.2f} ms" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=65, t=15, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Thời gian suy luận (ms) — càng thấp càng nhanh", rangemode="tozero", gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(title="", showgrid=False),
        font=dict(family="Arial, sans-serif", size=11),
    )
    return fig


# =========================================================
# TIÊU ĐỀ CHÍNH
# =========================================================

st.title("Hệ thống nhận diện biển báo giao thông Việt Nam")
st.caption("So sánh hiệu năng và độ chính xác phân loại giữa Deep Learning (EfficientNetV2), SVM và Gradient Boosting (LightGBM)")
st.divider()


# =========================================================
# BỐ CỤC CHÍNH (HAI CỘT)
# =========================================================

col_input, col_output = st.columns([0.85, 1.65], gap="large")

with col_input:
    with st.container(border=True):
        st.subheader("1. Nhập ảnh biển báo")
        st.write("Tải tệp tin, kéo thả từ máy tính hoặc chọn ảnh mẫu để tiến hành nhận diện.")

        # Thêm nút chọn chế độ nguồn ảnh để không bị xung đột / ghi đè nhau
        source_mode = st.radio(
            "Nguồn ảnh đầu vào",
            ["Tải / Kéo thả ảnh", "Thư viện ảnh mẫu"],
            horizontal=True,
            label_visibility="collapsed"
        )
        input_image = None

        if source_mode == "Tải / Kéo thả ảnh":
            uploaded_file = st.file_uploader(
                "Kéo thả hoặc duyệt ảnh biển báo",
                type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif", "avif", "ico"],
                help="Hỗ trợ đầy đủ các định dạng ảnh. Kéo thả file ảnh trực tiếp vào đây.",
                key="uploaded_traffic_sign",
            )
            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file).convert("RGB")
                except Exception:
                    st.error("Không thể đọc định dạng ảnh này. Vui lòng chọn ảnh khác.")
        else:
            if SAMPLE_DIR.exists():
                valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
                sample_files = sorted(
                    f for f in os.listdir(SAMPLE_DIR)
                    if f.lower().endswith(valid_exts)
                )
            else:
                sample_files = []

            if sample_files:
                selected_sample = st.selectbox(
                    "Chọn ảnh mẫu có sẵn",
                    sample_files,
                    format_func=lambda name: Path(name).stem.replace("_", " ").title(),
                )
                if selected_sample:
                    try:
                        input_image = Image.open(SAMPLE_DIR / selected_sample).convert("RGB")
                    except Exception:
                        st.error("Không thể đọc ảnh mẫu đã chọn.")
            else:
                st.info("Chưa có ảnh mẫu trong thư mục sample_images.")

        st.write("")
        top_k = st.slider(
            "Số lượng dự đoán hàng đầu (Top-K)",
            min_value=1,
            max_value=5,
            value=3,
            help="Số lớp biển báo có độ tin cậy cao nhất của mỗi mô hình.",
        )

        if input_image is not None:
            st.image(input_image, caption="Ảnh biển báo đầu vào đã chọn", width="stretch")
            st.caption(f"Kích thước ảnh: {input_image.width} × {input_image.height} pixel")

        run_btn = st.button(
            "Chạy nhận diện và so sánh mô hình",
            type="primary",
            width="stretch",
            disabled=input_image is None,
        )

# =========================================================
# THỰC HIỆN SUY LUẬN
# =========================================================

if run_btn and input_image is not None:
    new_results = {}
    new_latencies = {}
    try:
        with st.spinner("Đang chạy suy luận trên cả ba mô hình..."):
            for main_title, sub_title, key in PIPELINES:
                full_display_name = f"{main_title} {sub_title}"
                result = engine.predict(input_image, key, top_k=top_k)
                new_results[full_display_name] = result
                new_latencies[full_display_name] = float(result["latency_ms"])

        st.session_state["benchmark_results"] = new_results
        st.session_state["benchmark_latencies"] = new_latencies
    except Exception as exc:
        st.error(f"Quá trình nhận diện thất bại: {exc}")


# =========================================================
# KẾT QUẢ NHẬN DIỆN
# =========================================================

with col_output:
    st.subheader("2. Kết quả nhận diện")
    st.caption("Kết quả dự đoán và các chỉ số hiệu năng tương ứng của từng mô hình.")

    if class_table_error:
        st.warning(class_table_error)

    results = st.session_state.get("benchmark_results")
    latencies = st.session_state.get("benchmark_latencies")

    if not results:
        st.info("Hãy tải ảnh lên hoặc chọn ảnh mẫu, sau đó nhấn “Chạy nhận diện và so sánh mô hình” để xem kết quả.")
    else:
        if any(result.get("confidence_calibrated") is False for result in results.values()):
            st.caption(
                "Điểm SVM là decision margin đã chuẩn hóa để xếp hạng, không phải "
                "xác suất đã hiệu chỉnh và không nên so sánh trực tiếp với xác suất DL/Boosting."
            )
        model_cols = st.columns(3, gap="medium")

        for index, (main_title, sub_title, key) in enumerate(PIPELINES):
            full_display_name = f"{main_title} {sub_title}"
            result = results[full_display_name]
            sign = get_sign_info(result.get("top_class", "Chưa xác định"))

            # Xác định trạng thái nạp mô hình
            is_real = not result.get("is_mock", True)
            status_text = "Mô hình thực thi" if is_real else "Mô phỏng dự phòng"

            with model_cols[index]:
                with st.container(border=True):
                    # 1. Tên Model căn giữa, xuống dòng kiến trúc và có dòng trạng thái
                    st.markdown(
                        f"""
                        <div class="model-header-box">
                            <div class="model-main-title">{main_title}</div>
                            <div class="model-sub-title">{sub_title}</div>
                            <div class="model-status-tag">Trạng thái: {status_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.divider()

                    # 2. Biển báo dự đoán (mã + tên tiếng Việt thẳng hàng)
                    st.markdown('<div class="section-label">BIỂN BÁO DỰ ĐOÁN</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="sign-code">{sign["code"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="sign-name">{sign["name"]}</div>', unsafe_allow_html=True)
                    st.divider()

                    # 3. Độ tin cậy
                    st.markdown('<div class="section-label">ĐỘ TIN CẬY</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-container"><div class="metric-number">{float(result["confidence"]):.1%}</div></div>',
                        unsafe_allow_html=True,
                    )

                    # 4. Thời gian suy luận
                    st.markdown('<div class="section-label">THỜI GIAN SUY LUẬN</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-container"><div class="metric-number">{float(result["latency_ms"]):.2f} ms</div></div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        st.subheader("3. So sánh độ tin cậy")
        st.caption("So sánh điểm tin cậy của các lớp biển báo được dự đoán.")

        tab_confidence, tab_latency = st.tabs(["Độ tin cậy dự đoán", "Thời gian suy luận"])

        with tab_confidence:
            st.plotly_chart(plot_confidence_comparison(results), width="stretch")

        with tab_latency:
            st.caption("Thời gian tính bằng mili giây (ms) — giá trị thấp hơn thể hiện tốc độ nhanh hơn.")
            st.plotly_chart(plot_latency_comparison(latencies), width="stretch")

        st.divider()
        st.subheader("4. Chi tiết các dự đoán")

        detail_rows = []
        for main_title, sub_title, _ in PIPELINES:
            full_display_name = f"{main_title} {sub_title}"
            result = results[full_display_name]
            for rank, (label, confidence) in enumerate(result.get("predictions", {}).items(), start=1):
                sign = get_sign_info(label)
                detail_rows.append({
                    "Mô hình": full_display_name,
                    "Thứ hạng": rank,
                    "Mã biển báo": sign["code"],
                    "Tên biển báo": sign["name"],
                    "Độ tin cậy (%)": float(confidence) * 100.0,
                })

        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(
            detail_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Độ tin cậy (%)": st.column_config.ProgressColumn(
                    "Độ tin cậy (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
            },
        )
