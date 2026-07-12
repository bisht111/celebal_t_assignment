"""import streamlit as st
from PIL import Image

from model_loader import load_model, CLASSES
from inference import predict
from utils import similarity, detect
from utils import create_heatmap

st.set_page_config(
    page_title="Satellite Change Detection",
    layout="wide"
)

st.title("🛰 Satellite Land Use Classifier")

model = load_model()

left, right = st.columns(2)

with left:

    before = st.file_uploader(
        "Upload Before Image",
        type=["jpg","png","jpeg"],
        key="before"
    )

with right:

    after = st.file_uploader(
        "Upload After Image",
        type=["jpg","png","jpeg"],
        key="after"
    )


if before and after:

    img1 = Image.open(before).convert("RGB")
    img2 = Image.open(after).convert("RGB")

    p1, c1, e1 = predict(model, img1)
    p2, c2, e2 = predict(model, img2)

    sim = similarity(e1, e2)

    changed = detect(sim)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.image(img1, caption="Before")

        st.write(f"Prediction : **{CLASSES[p1]}**")

        st.write(f"Confidence : **{c1:.3f}**")

    with col2:

        st.image(img2, caption="After")

        st.write(f"Prediction : **{CLASSES[p2]}**")

        st.write(f"Confidence : **{c2:.3f}**")

    st.markdown("---")

    st.metric(
        "Cosine Similarity",
        f"{sim:.3f}"
    )

    if changed:

        st.error("🚨 CHANGE DETECTED")

    else:

        st.success("✅ NO SIGNIFICANT CHANGE")
    


    mode = st.sidebar.selectbox(
        "Detection Mode",
        [
            "High Recall",
            "Balanced",
            "High Precision"
        ]
    )

    if mode=="High Recall":
        threshold=0.75

    elif mode=="Balanced":
        threshold=0.64

    else:
        threshold=0.50

    changed = sim < threshold
        
    heatmap = create_heatmap(
        img1,
        img2
    )

    st.markdown("---")

    st.image(
        heatmap,
        caption="Change Heatmap"
)"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
import numpy as np
import cv2

from model_loader import load_model, CLASSES
from inference import predict
from utils import similarity

st.set_page_config(
    page_title="Satellite Land Use Classifier & Change Detector",
    layout="wide"
)

st.title("🛰 Satellite Image Land-Use Classifier & Temporal Change Detector")

model = load_model()

st.sidebar.header("Detection Threshold")

mode = st.sidebar.selectbox(
    "Operating Point",
    [
        "High Recall",
        "Balanced",
        "High Precision"
    ]
)

if mode == "High Recall":
    threshold = 0.75
elif mode == "Balanced":
    threshold = 0.638
else:
    threshold = 0.50

left, right = st.columns(2)

with left:
    before_file = st.file_uploader(
        "Upload BEFORE Image",
        type=["jpg", "jpeg", "png"],
        key="before"
    )

with right:
    after_file = st.file_uploader(
        "Upload AFTER Image",
        type=["jpg", "jpeg", "png"],
        key="after"
    )

if before_file is not None and after_file is not None:

    before = Image.open(before_file).convert("RGB")
    after = Image.open(after_file).convert("RGB")

    pred1, conf1, emb1 = predict(model, before)
    pred2, conf2, emb2 = predict(model, after)

    score = similarity(emb1, emb2)

    changed = score < threshold

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.image(before, caption="Before Image")
        st.success(f"Prediction : {CLASSES[pred1]}")
        st.write(f"Confidence : {conf1:.3f}")

    with c2:
        st.image(after, caption="After Image")
        st.success(f"Prediction : {CLASSES[pred2]}")
        st.write(f"Confidence : {conf2:.3f}")

    st.divider()

    st.metric(
        "Cosine Similarity",
        f"{score:.3f}"
    )

    if changed:
        st.error("🚨 CHANGE DETECTED")
    else:
        st.success("✅ NO CHANGE DETECTED")

    before_np = np.array(before.resize((224,224)))
    after_np = np.array(after.resize((224,224)))

    diff = cv2.absdiff(before_np, after_np)

    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    st.divider()

    st.subheader("Change Heatmap")

    st.image(
        heat,
        use_container_width=True
    )