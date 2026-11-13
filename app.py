"""
Paper Surface Defect Detection - interactive demo.

    streamlit run app.py

Upload a photo of the paper reel (or pick a test image) and the app
classifies it as clean paper / hole / pen mark, draws a bounding box around
the defect, and shows how the model "sees" the image: the illumination-
flattened map and every candidate region with its score.
"""
import os
import sys
import tempfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
Q3 = os.path.join(ROOT, "q3")
sys.path.insert(0, Q3)

st.set_page_config(page_title="Paper Defect Detection", layout="wide")
st.title("Paper Surface Defect Detection")
st.caption("CT-317 Machine Learning CEP | Sumayyah Mohammedy, Amna Khan, Mariha Mahmood, "
           "Haris Bin Nasir")

if not os.path.exists(os.path.join(Q3, "models", "image_classifier.joblib")):
    st.error("Models not found. Run `python q3/q3_defect_detection.py` first.")
    st.stop()

import defect_features as dfx                       # noqa: E402
from predict import load_models, predict_image      # noqa: E402

DATA_DIR = os.path.join(Q3, "data", "out")
SPLIT = os.path.join(Q3, "results", "split.csv")
COLORS = {"paper": "#1a9850", "hole": "#d73027", "penMark": "#4575b4"}
LABEL = {"paper": "No defect", "hole": "Hole", "penMark": "Pen mark"}


@st.cache_resource
def models():
    return load_models()


img_b, reg_b = models()


def save_upload(data, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Illumination flattening**: the uneven lighting is removed and every pixel "
        "becomes a z-score (how much darker than the local paper it is)\n"
        "2. **Candidate regions**: dark spots and strokes are found and ranked\n"
        "3. **Features**: colour histogram, spatial binning, HOG, LBP texture, GLCM and "
        "shape descriptors of each candidate\n"
        f"4. **{reg_b['model_name']}** scores every candidate; the most defect-like one "
        "gives the class and the bounding box\n\n"
        "**Test accuracy: 97.3%** on 225 held-out images (99.1% for the best single "
        "classifier).")

tab_one, tab_batch = st.tabs(["Single image", "Batch (many images)"])

# ------------------------------------------------------------ single image
with tab_one:
    have_data = os.path.isdir(DATA_DIR) and os.path.exists(SPLIT)
    modes = ["Upload an image"] + (["Pick a test image"] if have_data else [])
    mode = st.radio("Input", modes, horizontal=True)
    img_path, true = None, None
    if mode == modes[0]:
        up = st.file_uploader("Paper image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up is not None:
            img_path = save_upload(up.read(), os.path.splitext(up.name)[1])
    else:
        test = pd.read_csv(SPLIT).query("split == 'test'").file.tolist()
        c1, c2 = st.columns([3, 1])
        with c2:
            st.write("")
            if st.button("Random test image"):
                st.session_state.pick = test[np.random.randint(len(test))]
        with c1:
            choice = st.selectbox("Test image", test, key="pick")
        img_path = os.path.join(DATA_DIR, choice)
        true = choice.split("/")[0]

    if img_path:
        with st.spinner("Analysing..."):
            cls, probs, box, img = predict_image(img_path, img_b, reg_b)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(cv2.cvtColor(cv2.resize(img, (800, 800)), cv2.COLOR_BGR2RGB),
                     caption=f"prediction: {LABEL[cls]}")
        with c2:
            st.markdown(f"<div style='font-size:15px'>Prediction</div>"
                        f"<div style='font-size:40px;font-weight:700;color:{COLORS[cls]}'>"
                        f"{LABEL[cls]}</div><div>confidence {probs[cls]:.0%}</div>",
                        unsafe_allow_html=True)
            if true:
                st.write(f"True class: **{LABEL[true]}** "
                         f"{'(correct)' if true == cls else '(wrong)'}")
            st.bar_chart(pd.Series({LABEL[k]: v for k, v in probs.items()}), height=200)
            if box:
                x0, y0, x1, y1 = box
                st.write(f"Bounding box (x0, y0, x1, y1): {box} px")
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                h = max(150, int(0.9 * max(x1 - x0, y1 - y0)))
                crop = img[max(0, cy - h):cy + h, max(0, cx - h):cx + h]
                st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="zoom on the defect")

        with st.expander("How the model sees this image (pre-processing and candidates)"):
            Xp, cands, scale = dfx.proposal_features(img_path, reg_b.get("k", 5))
            P = reg_b["model"].predict_proba(Xp)
            s, z, _ = dfx.preprocess(dfx.load_gray(img_path))
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(s, cmap="gray")
            ax[0].set_title("grayscale input")
            ax[1].imshow(np.clip(z, -30, 10), cmap="magma")
            ax[1].set_title("illumination-flattened z-map + candidates")
            best = int(np.argmax(1 - P[:, 0]))
            for r, c in enumerate(cands):
                x, y, w, h = c["bbox"]
                ax[1].add_patch(plt.Rectangle((x - 4, y - 4), w + 8, h + 8, fill=False,
                                              color="lime" if r == best else "cyan", lw=1.5))
                ax[1].text(x, y - 8, f"#{r + 1}", color="white", fontsize=8)
            for a in ax:
                a.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Green = the candidate chosen as the defect. Faint specks that appear "
                       "in every image (dust on the lens) are recognised as background.")
            st.dataframe(pd.DataFrame({
                "candidate": [f"#{i + 1}" for i in range(len(cands))],
                "darkness score": [round(c["score"]) for c in cands],
                "darkest z": [round(c["min_z"], 1) for c in cands],
                "area (px)": [c["area"] for c in cands],
                "P(no defect)": P[:, 0].round(3), "P(hole)": P[:, 1].round(3),
                "P(pen mark)": P[:, 2].round(3)}), width="stretch")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Inspect many images at once, as on a production line.")
    ups = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if ups:
        rows, prog = [], st.progress(0.0)
        for i, u in enumerate(ups):
            p = save_upload(u.read(), os.path.splitext(u.name)[1])
            cls, probs, box, _ = predict_image(p, img_b, reg_b)
            rows.append({"image": u.name, "prediction": LABEL[cls],
                         "confidence": round(probs[cls], 3), "bounding box": str(box or "-")})
            prog.progress((i + 1) / len(ups))
        df = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", len(df))
        c2.metric("Defective", int((df.prediction != "No defect").sum()))
        c3.metric("Defect rate", f"{(df.prediction != 'No defect').mean():.0%}")
        st.dataframe(df, width="stretch")
        st.download_button("Download results (CSV)", df.to_csv(index=False),
                           "defect_results.csv", "text/csv")

with st.expander("How good is the model? (results on 225 images it never saw)"):
    res = os.path.join(Q3, "results")
    for f, cap in [("model_comparison.png", "All classifiers and feature sets"),
                   ("confusion_matrices.png", "Confusion matrices"),
                   ("localization_zoom.png", "Bounding boxes on test images")]:
        if os.path.exists(os.path.join(res, f)):
            st.image(os.path.join(res, f), caption=cap)
