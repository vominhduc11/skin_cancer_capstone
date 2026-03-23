from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR, DX_FULLNAME
from src.data_utils import load_artifact

st.set_page_config(page_title="Skin Cancer Detection Demo", page_icon="🩺", layout="centered")

st.title("🩺 Skin Cancer Detection - Demo")
st.caption("Demo dự đoán theo metadata của HAM10000 (không dùng ảnh thô).")

binary_model_path = ARTIFACT_DIR / "binary_svm.joblib"
multiclass_model_path = ARTIFACT_DIR / "multiclass_svm.joblib"

if not binary_model_path.exists() or not multiclass_model_path.exists():
    st.warning(
        "Chưa có model đã huấn luyện. Hãy chạy lệnh: `python -m src.train --csv data/HAM10000_metadata.csv`"
    )
    st.stop()


@st.cache_resource
def get_artifact(path: Path):
    return load_artifact(path)


binary_payload = get_artifact(binary_model_path)
multiclass_payload = get_artifact(multiclass_model_path)

task = st.radio(
    "Chọn bài toán dự đoán",
    options=["Binary classification", "Multi-class classification"],
)

st.subheader("Nhập thông tin bệnh nhân")

age = st.number_input("Age", min_value=0, max_value=120, value=45)
sex = st.selectbox("Sex", options=["male", "female", "unknown"])
localization = st.selectbox(
    "Localization",
    options=[
        "scalp",
        "face",
        "ear",
        "back",
        "trunk",
        "chest",
        "upper extremity",
        "lower extremity",
        "abdomen",
        "neck",
        "hand",
        "foot",
        "genital",
        "acral",
        "unknown",
    ],
)

input_df = pd.DataFrame(
    [
        {
            "age": age,
            "sex": sex,
            "localization": localization,
        }
    ]
)

if st.button("Dự đoán", type="primary"):
    if task == "Binary classification":
        payload = binary_payload
        model = payload["model"]
        pred = int(model.predict(input_df)[0])

        if pred == 1:
            st.error("Kết quả dự đoán: Nhóm nguy cơ cao (mel/bcc/akiec/vasc)")
        else:
            st.success("Kết quả dự đoán: Nhóm lành tính (nv/df/bkl)")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame(
                {
                    "Nhóm": payload["class_names"],
                    "Xác suất": probs,
                }
            )
            st.dataframe(prob_df, use_container_width=True)

    else:
        payload = multiclass_payload
        model = payload["model"]
        label_encoder = payload["label_encoder"]

        pred_encoded = model.predict(input_df)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        full_name = DX_FULLNAME.get(pred_label, pred_label)

        st.success(f"Kết quả dự đoán: {pred_label} - {full_name}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            class_names = list(label_encoder.classes_)
            prob_df = pd.DataFrame(
                {
                    "Lớp": class_names,
                    "Mô tả": [DX_FULLNAME.get(label, label) for label in class_names],
                    "Xác suất": probs,
                }
            ).sort_values("Xác suất", ascending=False)
            st.dataframe(prob_df, use_container_width=True)

st.markdown("---")
st.markdown(
    """
**Lưu ý học thuật**
- App này bám theo đề bài dùng `HAM10000_metadata.csv`
- Vì vậy đầu vào là metadata chứ không phải upload ảnh
- Muốn đúng nghĩa “image classification” hơn, bạn cần dùng ảnh thô + CNN
"""
)
