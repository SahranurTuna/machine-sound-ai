import json
import os
import tempfile

import joblib
import numpy as np
import streamlit as st

from features import extract_mel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def load_meta():
    if os.path.isfile(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


st.title("Makine Ses Anomali Tespit Sistemi")

try:
    bundle = load_bundle()
    clf = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
except Exception as e:
    st.error(f"Model yuklenemedi: {e}")
    st.info("Once egitim icin calistirin: python train.py")
    st.stop()

meta = load_meta()
if meta:
    st.caption(
        f"Test: dogruluk %{meta.get('test_accuracy', 0) * 100:.1f} | "
        f"anormal yakalama %{meta.get('abnormal_recall', 0) * 100:.1f} | "
        f"esik {meta.get('threshold', threshold):.2f}"
    )

uploaded_file = st.file_uploader("WAV dosyasi yukleyin", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        vec = extract_mel(temp_path).squeeze().flatten().reshape(1, -1)
        probs = clf.predict_proba(vec)[0]
        abnormal_prob = float(probs[1])
        normal_prob = float(probs[0])

        st.subheader("Sonuc")
        st.write(f"Normal: %{normal_prob * 100:.2f}")
        st.write(f"Anormal: %{abnormal_prob * 100:.2f}")

        if abnormal_prob >= threshold:
            st.error("ANORMAL")
        else:
            st.success("NORMAL")
    finally:
        os.unlink(temp_path)
