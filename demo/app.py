from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="VN Traffic Sign Demo", layout="centered")
st.title("VN Traffic Sign Classification — Demo")
st.info("TODO(team-demo): load checkpoint, preprocess one image, show top-k probs.")
pipeline = st.selectbox("Pipeline", ["svm", "boosting", "dl"])
ckpt = st.text_input("Checkpoint path", value="")
img = st.file_uploader("Traffic sign image", type=["jpg", "jpeg", "png"])
if st.button("Predict"):
    st.warning(f"Not implemented — fill demo/app.py ({pipeline}, {ckpt}, {bool(img)})")
