"""
Interactive demo of all three CEP solutions - test the trained models with
your own input and see WHY they decide what they decide.

    streamlit run app.py

Q1  type headlines            -> real/fake + the k nearest training headlines
Q2  record / upload / pick     -> emotion + probabilities + what the filter removed
Q3  upload / pick an image     -> class + bounding box + every scored candidate region

Requires the trained models in q*/models/ (included in the repository).
"""
import os
import sys
import tempfile

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("q2", "q3"):
    sys.path.insert(0, os.path.join(ROOT, sub))

st.set_page_config(page_title="ML CEP demo", layout="wide")
st.title("ML Complex Engineering Problem - live demo")
st.caption("CT-317 Machine Learning | Sumayyah Mohammedy, Amna Khan, Mariha Mahmood, "
           "Haris Bin Nasir")
tab1, tab2, tab3 = st.tabs(["Q1 Fake-news headlines (KNN)",
                            "Q2 Emotion from audio",
                            "Q3 Paper defect detection"])


def missing(path, script):
    st.warning(f"Model `{os.path.relpath(path, ROOT)}` not found. Run `python {script}` first.")


def results_image(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")


def save_upload(data, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


# ------------------------------------------------------------------ Q1 ----
with tab1:
    path = os.path.join(ROOT, "q1", "models", "knn_cosine.joblib")
    if not os.path.exists(path):
        missing(path, "q1/q1_knn.py")
    else:
        b = load_bundle(path)
        model, vec = b["model"], b["vectorizer"]
        st.write(f"KNN with **cosine** distance, k = {b['best_k']}, bag-of-words "
                 f"(CountVectorizer, {len(vec.vocabulary_)} words). Test accuracy 75.7%.")
        text = st.text_area("Headline(s), one per line",
                            "trump says he will build a wall on the border\n"
                            "hillary clinton is going to jail say insiders")
        lines = [t.strip().lower() for t in text.splitlines() if t.strip()]
        if lines:
            X = vec.transform(lines)
            proba = model.predict_proba(X)
            known = np.asarray((X > 0).sum(axis=1)).ravel()
            st.dataframe(pd.DataFrame({
                "headline": lines,
                "prediction": [b["labels"][int(c)] for c in model.predict(X)],
                "P(real)": proba[:, list(model.classes_).index(1)].round(2),
                "words known to the model": known}), width="stretch")
            if "train_headlines" in b:
                pick = st.selectbox("Explain a prediction: show its nearest training headlines",
                                    lines)
                xi = vec.transform([pick])
                if xi.nnz == 0:
                    st.info("None of these words are in the vocabulary - the prediction is "
                            "just the majority vote of arbitrary neighbours.")
                dist, idx = model.kneighbors(xi, n_neighbors=b["best_k"])
                st.dataframe(pd.DataFrame({
                    "cosine distance": dist[0].round(3),
                    "training headline": [b["train_headlines"][i] for i in idx[0]],
                    "label": [b["labels"][int(b["train_labels"][i])] for i in idx[0]]}),
                    width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            results_image(os.path.join(ROOT, "q1", "results", "knn_euclidean.png"), "Euclidean")
        with c2:
            results_image(os.path.join(ROOT, "q1", "results", "knn_cosine.png"), "Cosine")

# ------------------------------------------------------------------ Q2 ----
EMO = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy",
       "NEU": "Neutral", "SAD": "Sad"}

with tab2:
    path = os.path.join(ROOT, "q2", "models", "best_model.joblib")
    if not os.path.exists(path):
        missing(path, "q2/q2_audio_sentiment.py")
    else:
        import audio_features as af
        import librosa
        import librosa.display
        b = load_bundle(path)
        st.write(f"Model: **{b['model_name']}**, low-pass {b['lowpass_hz']} Hz, "
                 f"features: {', '.join(b['feature_groups'])}. "
                 f"Trained and tested on *different speakers*.")
        sample_dir = os.path.join(ROOT, "q2", "dataset_repo", "NoiseAudioWAV")
        have_data = os.path.isdir(sample_dir)
        modes = ["Record with microphone", "Upload a WAV file"] + \
                (["Pick a clip from the test speakers"] if have_data else [])
        mode = st.radio("Input", modes, horizontal=True)
        wav, true = None, None
        if mode == modes[0]:
            rec = st.audio_input("Say a sentence with an emotion (e.g. angry, happy, sad)")
            if rec is not None:
                wav = save_upload(rec.getvalue(), ".wav")
        elif mode == modes[1]:
            up = st.file_uploader("WAV clip", type=["wav"], key="wav")
            if up is not None:
                wav = save_upload(up.read(), ".wav")
        else:
            split = pd.read_csv(os.path.join(ROOT, "q2", "results", "split.csv"))
            choice = st.selectbox("Test clip", split.query("split == 'test'").file.tolist()[::7])
            wav = os.path.join(sample_dir, choice)
            true = EMO[choice.split("_")[2]]
        if wav:
            if mode != modes[0]:
                st.audio(wav)
            x = af.features_from_file(wav, cutoff=b["lowpass_hz"])[None, :]
            model = b["model"]
            pred = b["labels"][int(model.predict(x)[0])]
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader(f"Predicted emotion: {pred}")
                if true:
                    st.write(f"True label: **{true}** "
                             f"{'(correct)' if true == pred else '(wrong)'}")
                p = model.predict_proba(x)[0]
                st.bar_chart(pd.Series(p, index=[b["labels"][int(c)] for c in model.classes_]))
            with c2:
                y, sr = librosa.load(wav, sr=af.SR)
                yf = af.lowpass_filter(y, sr, b["lowpass_hz"])
                fig, ax = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
                for a, sig, t in zip(ax, (y, yf), ("input", f"after {b['lowpass_hz']} Hz low-pass")):
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)),
                                                ref=np.max)
                    librosa.display.specshow(D, sr=sr, hop_length=160, x_axis="time",
                                             y_axis="hz", ax=a)
                    a.set_title(t, fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        with st.expander("Model comparison on unseen speakers"):
            results_image(os.path.join(ROOT, "q2", "results", "model_comparison.png"), "")

# ------------------------------------------------------------------ Q3 ----
with tab3:
    mp = os.path.join(ROOT, "q3", "models", "image_classifier.joblib")
    if not os.path.exists(mp):
        missing(mp, "q3/q3_defect_detection.py")
    else:
        import defect_features as dfx
        from predict import load_models, predict_image
        img_b, reg_b = load_models()
        st.write(f"Decision: **{img_b.get('decision', 'single-candidate')}** "
                 f"({reg_b['model_name']} region classifier scores every dark-region proposal).")
        data_dir = os.path.join(ROOT, "q3", "data", "out")
        have_data = os.path.isdir(data_dir)
        modes = ["Upload an image"] + (["Pick a test image"] if have_data else [])
        mode = st.radio("Input", modes, horizontal=True, key="q3mode")
        img_path, true = None, None
        if mode == modes[0]:
            up = st.file_uploader("Paper image", type=["jpg", "jpeg", "png"], key="img")
            if up is not None:
                img_path = save_upload(up.read(), os.path.splitext(up.name)[1])
        else:
            split = pd.read_csv(os.path.join(ROOT, "q3", "results", "split.csv"))
            choice = st.selectbox("Test image", split.query("split == 'test'").file.tolist())
            img_path = os.path.join(data_dir, choice)
            true = choice.split("/")[0]
        if img_path:
            cls, probs, box, img = predict_image(img_path, img_b, reg_b)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(cv2.cvtColor(cv2.resize(img, (800, 800)), cv2.COLOR_BGR2RGB),
                         caption=f"prediction: {cls}")
            with c2:
                st.subheader(f"Prediction: {cls}")
                if true:
                    st.write(f"True class: **{true}** {'(correct)' if true == cls else '(wrong)'}")
                st.bar_chart(pd.Series(probs))
                if box:
                    x0, y0, x1, y1 = box
                    st.write(f"Bounding box (x0, y0, x1, y1) in pixels: {box}")
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    h = max(150, int(0.9 * max(x1 - x0, y1 - y0)))
                    crop = img[max(0, cy - h):cy + h, max(0, cx - h):cx + h]
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="zoom on the defect")
            with st.expander("How the model sees this image (pre-processing and proposals)",
                             expanded=False):
                Xp, cands, scale = dfx.proposal_features(img_path, reg_b.get("k", 5))
                P = reg_b["model"].predict_proba(Xp)
                s, z, _ = dfx.preprocess(dfx.load_gray(img_path))
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(s, cmap="gray")
                ax[0].set_title("grayscale input (1000 px)")
                ax[1].imshow(np.clip(z, -30, 10), cmap="magma")
                ax[1].set_title("illumination-flattened z-map + proposals")
                for r, c in enumerate(cands):
                    x, y, w, h = c["bbox"]
                    ax[1].add_patch(plt.Rectangle((x - 4, y - 4), w + 8, h + 8, fill=False,
                                                  color="cyan" if r else "lime", lw=1.5))
                    ax[1].text(x, y - 8, f"#{r + 1}", color="white", fontsize=8)
                for a in ax:
                    a.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(pd.DataFrame({
                    "proposal": [f"#{i + 1}" for i in range(len(cands))],
                    "darkness score": [round(c["score"]) for c in cands],
                    "min z": [round(c["min_z"], 1) for c in cands],
                    "area (px)": [c["area"] for c in cands],
                    "P(paper)": P[:, 0].round(3), "P(hole)": P[:, 1].round(3),
                    "P(penMark)": P[:, 2].round(3)}), width="stretch")
