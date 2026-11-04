"""
Predict the emotion of one or more WAV files with the trained Q2 model.

Usage (from the ml_cep folder, after running q2_audio_sentiment.py once):
    python q2/predict.py path/to/clip.wav [more.wav ...]
    python q2/predict.py q2/dataset_repo/NoiseAudioWAV/1001_DFA_ANG_XX.wav
"""
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_features as af  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.joblib")


def main(paths):
    if not paths:
        print(__doc__)
        return
    if not os.path.exists(MODEL_PATH):
        sys.exit("No trained model found - run  python q2/q2_audio_sentiment.py  first.")
    bundle = joblib.load(MODEL_PATH)
    model, names = bundle["model"], bundle["labels"]
    print(f"Model: {bundle['model_name']} (low-pass {bundle['lowpass_hz']} Hz)\n")
    for p in paths:
        x = af.features_from_file(p, cutoff=bundle["lowpass_hz"])[None, :]
        pred = model.predict(x)[0]
        line = f"{os.path.basename(p):35s} -> {names[int(pred)]}"
        clf = model.named_steps["clf"]
        if hasattr(clf, "predict_proba") and getattr(clf, "probability", True):
            proba = model.predict_proba(x)[0]
            top = np.argsort(proba)[::-1][:3]
            line += "   (" + ", ".join(f"{names[int(model.classes_[i])]} {proba[i]:.2f}" for i in top) + ")"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
