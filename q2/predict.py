"""
Predict the emotion and sentiment of WAV files with the trained WavLM model.

Usage (from the ml_cep folder):
    python q2/predict.py my_recording.wav [more.wav ...]

Needs q2/models/wavlm_emotion.joblib (in the repository, or produced by
q2_wavlm_emotion.py). The WavLM network itself (~380 MB) is downloaded from
Hugging Face automatically on first use.
"""
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_features as af  # noqa: E402
import ssl_features as ssl   # noqa: E402

MODEL_PATH = os.path.join(HERE, "models", "wavlm_emotion.joblib")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        sys.exit("No trained model found - run  python q2/q2_wavlm_emotion.py  first.")
    return joblib.load(path)


def check_quality(y, sr=af.SR):
    """Simple input checks that explain bad predictions on own recordings."""
    issues = []
    dur = len(y) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if dur < 1.0:
        issues.append(f"very short recording ({dur:.1f} s) - speak a full sentence (2-4 s)")
    if dur > 8.0:
        issues.append(f"long recording ({dur:.1f} s) - the model was trained on 2-4 s sentences")
    if peak < 0.05 or rms < 0.005:
        issues.append("very quiet - move closer to the microphone")
    if peak > 0.99 and np.mean(np.abs(y) > 0.98) > 0.001:
        issues.append("clipping - the recording is too loud")
    return issues


def embed(path, bundle):
    """Raw file -> (clean signal, pooled WavLM feature vector)."""
    import librosa
    raw, _ = librosa.load(path, sr=af.SR, mono=True)
    y = af.load_and_clean(path, cutoff=bundle["lowpass_hz"])
    E = ssl.embed_signal(y)
    return raw, E[bundle["layers"]].mean(axis=0)


def load_corpus(path=os.path.join(HERE, "models", "train_embeddings.npz")):
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"]


def personalised_model(bundle, X_user, y_user):
    """Base model refit on corpus + the user's own weighted examples."""
    from personalization import personalise
    Xc, yc = load_corpus()
    return personalise(bundle["model"], Xc, yc, np.asarray(X_user), np.asarray(y_user))


def predict_vector(x, bundle, model=None):
    """Returns (emotion probabilities dict, sentiment probabilities dict)."""
    model = model if model is not None else bundle["model"]
    p = model.predict_proba(x[None, :])[0]
    emo = {bundle["emotions"][int(c)]: float(v) for c, v in zip(model.classes_, p)}
    sent = {s: 0.0 for s in bundle["sentiments"]}
    for c, v in zip(model.classes_, p):
        sent[bundle["sentiments"][bundle["sentiment_of"][int(c)]]] += float(v)
    return emo, sent


def main(paths):
    if not paths:
        print(__doc__)
        return
    b = load_model()
    print(f"Model: WavLM ({b['representation']}) + {b['model_name']}\n")
    for p in paths:
        raw, x = embed(p, b)
        emo, sent = predict_vector(x, b)
        e = max(emo, key=emo.get)
        s = max(sent, key=sent.get)
        top = sorted(emo.items(), key=lambda kv: -kv[1])[:3]
        print(f"{os.path.basename(p):35s} -> sentiment {s} ({sent[s]:.2f}), emotion {e} "
              f"(" + ", ".join(f"{k} {v:.2f}" for k, v in top) + ")")
        for issue in check_quality(raw):
            print(f"   warning: {issue}")


if __name__ == "__main__":
    main(sys.argv[1:])
