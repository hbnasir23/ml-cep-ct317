"""
Evaluate the models on the user's own labelled recordings (q2/my_recordings/,
file names start with the emotion, e.g. Happy_123.wav). These recordings are
private and not part of the repository.

    python q2/eval_my_voice.py [wavlm|e2v]
"""
import glob
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_features as af  # noqa: E402

NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
SENT = np.array([0, 0, 0, 2, 1, 0])
SENT_NAMES = ["Negative", "Neutral", "Positive"]


def files():
    fs = sorted(glob.glob(os.path.join(HERE, "my_recordings", "*.wav")))
    return fs, np.array([NAMES.index(os.path.basename(f).split("_")[0]) for f in fs])


def main(kind):
    fs, y = files()
    print(f"{len(fs)} own recordings: " + ", ".join(f"{NAMES[e]} {np.sum(y == e)}" for e in np.unique(y)))
    if kind == "wavlm":
        import ssl_features as ssl
        b = joblib.load(os.path.join(HERE, "models", "wavlm_emotion.joblib"))
        X = np.stack([ssl.embed_signal(af.load_and_clean(f))[b["layers"]].mean(0) for f in fs])
        P = b["model"].predict_proba(X)
        np.save(os.path.join(HERE, "my_recordings", "_wavlm_X.npy"), X)
    else:
        import e2v_features as e2v
        labs = e2v.labels()
        idx = [next(i for i, l in enumerate(labs) if k in l)
               for k in ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]]
        out = [e2v.embed_signal(af.load_and_clean(f)) for f in fs]
        X = np.stack([o[0] for o in out])
        P = np.stack([o[1][idx] for o in out])
        P = P / P.sum(1, keepdims=True)
        np.save(os.path.join(HERE, "my_recordings", "_e2v_X.npy"), X)
    pred = P.argmax(1)
    for f, t, p, pr in zip(fs, y, pred, P):
        print(f"  {os.path.basename(f)[:22]:22s} true {NAMES[t]:8s} pred {NAMES[p]:8s} "
              f"({pr[p]:.2f})  {'OK' if t == p else '--'}")
    print(f"{kind}: emotion accuracy {np.mean(pred == y):.2f}, "
          f"sentiment accuracy {np.mean(SENT[pred] == SENT[y]):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "wavlm")
