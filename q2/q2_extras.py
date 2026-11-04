"""
Q2 extra experiments (run after q2_audio_sentiment.py, re-uses its cached
features and the hyper-parameters it selected).

    python q2/q2_extras.py

 1. Soft-voting ensemble of SVM + Logistic Regression + Neural Network.
 2. Per-speaker feature normalisation: every feature is z-scored using the
    mean/std of all clips of the same speaker. This removes speaker identity
    (voice pitch, timbre, recording level) and keeps the emotion-related
    variation. It needs several clips of a speaker but NO labels, so it is
    legitimate at test time (e.g. a call-centre agent or a user profile).
 3. Results by emotion intensity level (LO / MD / HI / XX) for the best model.
Outputs: q2/results/extras.csv, q2/results/accuracy_by_level.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q2_audio_sentiment import LABELS, OUT_DIR, CACHE_DIR, SEED  # noqa: E402


def parse(v):
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return v


def build_models(table):
    p = {m: {k: parse(v) for k, v in json.loads(table.loc[m, "best_params"]).items()}
         for m in table.index}
    svm = SVC(kernel="rbf", probability=True, random_state=SEED,
              C=float(p["SVM"]["C"]), gamma=p["SVM"]["gamma"])
    lr = LogisticRegression(max_iter=3000, C=float(p["Logistic Regression"]["C"]))
    hl = p["Neural Network"]["hidden_layer_sizes"]
    hl = tuple(int(x) for x in str(hl).strip("()[] ").split(",") if x.strip())
    mlp = MLPClassifier(hidden_layer_sizes=hl, alpha=float(p["Neural Network"]["alpha"]),
                        max_iter=600, early_stopping=True, n_iter_no_change=20, random_state=SEED)
    sc = lambda est: Pipeline([("scaler", StandardScaler()), ("clf", est)])  # noqa: E731
    return {"SVM": sc(svm), "Logistic Regression": sc(lr), "Neural Network": sc(mlp),
            "Ensemble (SVM+LR+NN)": sc(VotingClassifier(
                [("svm", svm), ("lr", lr), ("mlp", mlp)], voting="soft"))}


def speaker_normalise(X, actors):
    Xn = np.empty_like(X)
    for a in np.unique(actors):
        m = actors == a
        Xn[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return Xn


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, "split.csv"))
    d = np.load(os.path.join(CACHE_DIR, "features_lp4000.npz"), allow_pickle=True)
    assert list(d["files"]) == list(df.file), "run q2_audio_sentiment.py first"
    X = d["X"]
    y = np.array([LABELS.index(e) for e in df.emotion])
    trva = df.split.isin(["train", "val"]).values
    te = (df.split == "test").values
    table = pd.read_csv(os.path.join(OUT_DIR, "model_comparison.csv")).set_index("model")

    rows, best = [], None
    for norm_name, Xv in [("global scaling", X), ("per-speaker normalisation",
                                                  speaker_normalise(X, df.actor.values))]:
        for name, model in build_models(table).items():
            model.fit(Xv[trva], y[trva])
            yp = model.predict(Xv[te])
            acc, f1 = accuracy_score(y[te], yp), f1_score(y[te], yp, average="macro")
            rows.append({"normalisation": norm_name, "model": name,
                         "test_acc": round(acc, 4), "test_macro_f1": round(f1, 4)})
            print(rows[-1], flush=True)
            if best is None or acc > best[0]:
                best = (acc, name, norm_name, yp)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "extras.csv"), index=False)
    print("\n", res.to_string(index=False))

    acc, name, norm_name, yp = best
    lv = df[te].assign(correct=(yp == y[te]))
    by_level = lv.groupby("level").correct.agg(["mean", "size"]).rename(
        columns={"mean": "accuracy", "size": "n_clips"}).round(4)
    by_level.to_csv(os.path.join(OUT_DIR, "accuracy_by_level.csv"))
    print(f"\nAccuracy by intensity level ({name}, {norm_name}):\n", by_level)


if __name__ == "__main__":
    main()
