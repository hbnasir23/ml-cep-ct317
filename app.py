"""
Voice Sentiment & Emotion Recognition - interactive demo.

    streamlit run app.py

Record your own voice (or upload / pick a clip), and the app predicts the
SENTIMENT (positive / neutral / negative) and the EMOTION (anger, disgust,
fear, happy, neutral, sad), shows how confident it is, what the noise filter
removed, and warns when the recording itself is the problem.

Model: 4 kHz low-pass filter -> pretrained speech embeddings (the deployed
feature set: WavLM / emotion2vec+ / both) -> scikit-learn classifier trained on
CREMA-D + RAVDESS (see README.md / report).
"""
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
Q2 = os.path.join(ROOT, "q2")
sys.path.insert(0, Q2)

st.set_page_config(page_title="Voice Sentiment Recognition", layout="wide")
st.title("Voice Sentiment & Emotion Recognition")
st.caption("CT-317 Machine Learning CEP | Sumayyah Mohammedy, Amna Khan, Mariha Mahmood, "
           "Haris Bin Nasir")

import audio_features as af   # noqa: E402
import predict as pr          # noqa: E402

MODEL_PATH = pr.MODEL_PATH
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python q2/q2_final_model.py` first.")
    st.stop()

SENT_COLOR = {"Positive": "#1a9850", "Neutral": "#7f7f7f", "Negative": "#d73027"}
EMO_CODE = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy",
            "NEU": "Neutral", "SAD": "Sad"}
RAV_CODE = {"01": "Neutral", "03": "Happy", "04": "Sad", "05": "Anger", "06": "Fear",
            "07": "Disgust"}


@st.cache_resource(show_spinner="Loading the speech model (first start can take a minute)...")
def load():
    b = pr.load_model(MODEL_PATH)
    pr.warm_up(b)
    return b


bundle = load()
for key, default in (("history", []), ("examples", []), ("personal", None), ("last_x", None)):
    if key not in st.session_state:
        st.session_state[key] = default


def active_model():
    if st.session_state.personal is not None and st.session_state.get("use_personal", True):
        return st.session_state.personal
    return None

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Low-pass filter** at 4 kHz removes high-frequency noise\n"
        f"2. **{bundle['representation']}** (pretrained speech model) turns the voice "
        "into a numeric description\n"
        f"3. A **{bundle['model_name']}** classifier predicts 6 emotions; their "
        "probabilities are summed into 3 sentiments\n\n"
        "**Tips for your own recordings**: speak one full sentence (2-4 s), "
        "act the emotion clearly, keep the microphone at a normal distance.")
    st.markdown("**Sentiment mapping**  \nPositive = Happy  \nNeutral = Neutral  \n"
                "Negative = Anger, Disgust, Fear, Sad")

modes = ["Record with microphone", "Upload a WAV file"]
crema_dir = os.path.join(Q2, "dataset_repo", "NoiseAudioWAV")
rav_dir = os.path.join(Q2, "ravdess")
split_csv = os.path.join(Q2, "results", "split.csv")
if os.path.isdir(crema_dir) and os.path.exists(split_csv):
    modes.append("CREMA-D clip (unseen test speakers)")
if os.path.isdir(rav_dir):
    modes.append("RAVDESS clip (other corpus)")
mode = st.radio("Input", modes, horizontal=True)

wav, true, src = None, None, None
if mode == modes[0]:
    rec = st.audio_input("Press the microphone, say one sentence with an emotion, press stop")
    if rec is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(rec.getvalue())
        tmp.close()
        wav, src = tmp.name, "microphone"
elif mode == modes[1]:
    up = st.file_uploader("WAV file", type=["wav"])
    if up is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(up.read())
        tmp.close()
        wav, src = tmp.name, up.name
elif mode.startswith("CREMA"):
    files = pd.read_csv(split_csv).query("split == 'test'").file.tolist()
    choice = st.selectbox("Clip", files[::5])
    wav, src = os.path.join(crema_dir, choice), choice
    true = EMO_CODE[choice.split("_")[2]]
else:
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(rav_dir) for f in fs
                   if f.endswith(".wav") and f.split("-")[2] in RAV_CODE)
    choice = st.selectbox("Clip (actor = last number)", [os.path.basename(f) for f in files][::4])
    wav = next(f for f in files if f.endswith(choice))
    src, true = choice, RAV_CODE[choice.split("-")[2]]

if wav:
    if mode != modes[0]:
        st.audio(wav)
    with st.spinner("Analysing..."):
        raw, x = pr.embed(wav, bundle)
        emo, sent = pr.predict_vector(x, bundle, active_model())
    st.session_state.last_x = x
    if active_model() is not None:
        st.success(f"Using your personalised model ({len(st.session_state.examples)} of your "
                   "own examples).")
    e_pred = max(emo, key=emo.get)
    s_pred = max(sent, key=sent.get)
    issues = pr.check_quality(raw)

    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    with c1:
        st.markdown(f"<div style='font-size:15px'>Sentiment</div>"
                    f"<div style='font-size:40px;font-weight:700;color:{SENT_COLOR[s_pred]}'>"
                    f"{s_pred}</div><div style='font-size:15px'>confidence {sent[s_pred]:.0%}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:15px;margin-top:18px'>Emotion</div>"
                    f"<div style='font-size:32px;font-weight:700'>{e_pred}</div>"
                    f"<div style='font-size:15px'>confidence {emo[e_pred]:.0%}</div>",
                    unsafe_allow_html=True)
        if true:
            ok_e = true == e_pred
            ok_s = bundle["sentiments"][bundle["sentiment_of"][bundle["emotions"].index(true)]] == s_pred
            st.write(f"True emotion: **{true}** - emotion {'correct' if ok_e else 'wrong'}, "
                     f"sentiment {'correct' if ok_s else 'wrong'}")
        if max(emo.values()) < 0.4:
            st.info("Low confidence: the voice does not clearly match one emotion.")
        for i in issues:
            st.warning(i)
    with c2:
        st.caption("Sentiment probabilities")
        st.bar_chart(pd.Series(sent), height=180)
        st.caption("Emotion probabilities")
        st.bar_chart(pd.Series(emo), height=200)
    with c3:
        import librosa
        import librosa.display
        yf = af.lowpass_filter(raw, af.SR, bundle["lowpass_hz"])
        fig, ax = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True)
        for a, sig, t in zip(ax, (raw, yf), ("your input", "after 4 kHz low-pass filter")):
            D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)),
                                        ref=np.max)
            librosa.display.specshow(D, sr=af.SR, hop_length=160, x_axis="time", y_axis="hz", ax=a)
            a.set_title(t, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    entry = {"input": src, "sentiment": s_pred, "P(sent)": round(sent[s_pred], 2),
             "emotion": e_pred, "P(emotion)": round(emo[e_pred], 2), "true": true or "-",
             "model": "personalised" if active_model() is not None else "base"}
    if not st.session_state.history or st.session_state.history[-1] != entry:
        st.session_state.history.append(entry)

    # feedback loop: label this clip and keep it as a personal training example
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        correct = st.selectbox("What emotion did you actually express? (adds this clip as a "
                               "training example of your voice)", bundle["emotions"],
                               index=bundle["emotions"].index(true or e_pred))
    with fc2:
        st.write("")
        st.write("")
        already = any(np.array_equal(x, v) for v, _, _ in st.session_state.examples)
        if st.button("Add as my example", disabled=already,
                     help="This recording is already one of your examples" if already else None):
            st.session_state.examples.append((x, bundle["emotions"].index(correct), src))
            # keep a copy so models can be evaluated on real user recordings
            import shutil
            import time as _t
            rec_dir = os.path.join(Q2, "my_recordings")
            os.makedirs(rec_dir, exist_ok=True)
            shutil.copy(wav, os.path.join(rec_dir, f"{correct}_{int(_t.time() * 1000)}.wav"))
            st.toast(f"Added a '{correct}' example ({len(st.session_state.examples)} total)")

# ------------------------------------------------------- personalisation ---
st.subheader("Teach the model your voice (few-shot personalisation)")
st.markdown(
    "A model trained on studio actors is less accurate on a new voice and microphone. "
    "Record **2 examples of each emotion** (12 clips) with *Record with microphone* above and "
    "press **Add as my example** after each, choosing the emotion you meant. Then press "
    "**Personalise**. In our tests on 8 unseen speakers this raised emotion accuracy from "
    "71% to 82% and sentiment accuracy from 84% to 90%.")
ex = st.session_state.examples
if ex:
    counts = pd.Series([bundle["emotions"][e] for _, e, _ in ex]).value_counts()
    st.dataframe(counts.reindex(bundle["emotions"], fill_value=0).rename("your examples").to_frame().T,
                 width="stretch")
p1, p2, p3 = st.columns(3)
with p1:
    if st.button("Personalise", disabled=len(ex) < 3):
        with st.spinner("Retraining on the corpus + your examples..."):
            st.session_state.personal = pr.personalised_model(
                bundle, [v for v, _, _ in ex], [e for _, e, _ in ex])
        st.rerun()
with p2:
    st.toggle("Use personalised model", value=True, key="use_personal",
              disabled=st.session_state.personal is None)
with p3:
    if st.button("Reset to base model"):
        st.session_state.personal = None
        st.session_state.examples = []
        st.rerun()

if st.session_state.history:
    st.subheader("Your session")
    st.dataframe(pd.DataFrame(st.session_state.history[::-1]), width="stretch")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

with st.expander("How good is the model? (results on data it never saw)"):
    res = os.path.join(Q2, "results", "wavlm")
    for f, cap in [("model_comparison.png", "Classifier comparison"),
                   ("cm_sentiment.png", "Sentiment confusion matrix"),
                   ("cm_emotion.png", "Emotion confusion matrix")]:
        if os.path.exists(os.path.join(res, f)):
            st.image(os.path.join(res, f), caption=cap)
