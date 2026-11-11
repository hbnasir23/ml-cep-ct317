"""
Q2 (improved) - speech emotion / sentiment recognition with pretrained WavLM
embeddings + the scikit-learn classifiers required by the handout.

Run order (from the ml_cep folder):
    python q2/q2_audio_sentiment.py      # split + classical baseline (once)
    python q2/q2_wavlm_emotion.py        # this script

What it does
 1. Loads the WavLM layer embeddings of all 7,442 CREMA-D clips (extracted by
    ssl_features.py, cached) and the same speaker-independent split.
 2. Chooses the representation on the VALIDATION speakers: every single
    layer, and averages of layer ranges (logistic-regression probe).
 3. Tunes KNN, Logistic Regression, Naive Bayes, SVM and MLP on validation,
    refits on train+val, evaluates once on the unseen test speakers.
 4. Reports 6-emotion accuracy and 3-class SENTIMENT accuracy
    (positive = Happy, neutral = Neutral, negative = Anger/Disgust/Fear/Sad),
    obtained by summing the emotion probabilities.
 5. Cross-corpus test on RAVDESS (different actors, studio and microphone,
    clean audio): the closest stand-in for "your own recordings".
 6. Multi-corpus model: retrain on CREMA-D + 16 RAVDESS actors and test on the
    8 held-out RAVDESS actors and the CREMA-D test speakers.
 7. Saves the deployed model (q2/models/wavlm_emotion.joblib) used by the app.
"""
import itertools
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ssl_features as ssl  # noqa: E402

OUT_DIR = os.path.join(HERE, "results", "wavlm")
MODEL_DIR = os.path.join(HERE, "models")
CACHE = os.path.join(HERE, "cache", "ssl")
RAV_DIR = os.path.join(HERE, "ravdess")
SEED = 42
LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
SENT_NAMES = ["Negative", "Neutral", "Positive"]
SENT_OF = np.array([0, 0, 0, 2, 1, 0])        # emotion index -> sentiment index
# RAVDESS emotion code (3rd field) -> our label; calm (02) and surprised (08) unused
RAV_MAP = {"01": "NEU", "03": "HAP", "04": "SAD", "05": "ANG", "06": "FEA", "07": "DIS"}


# ----------------------------------------------------------------- data ---
def load_crema():
    split = pd.read_csv(os.path.join(HERE, "results", "split.csv"))
    parts = [np.load(os.path.join(CACHE, f"emb_{k}.npy")) for k in range(4)]
    E = np.concatenate(parts).astype(np.float32)
    assert len(E) == len(split), "embedding shards do not match split.csv"
    y = np.array([LABELS.index(e) for e in split.emotion])
    return split, E, y


def ravdess_table():
    rows = []
    for root, _, files in os.walk(RAV_DIR):
        for f in files:
            if f.endswith(".wav"):
                parts = f[:-4].split("-")    # modality-channel-emotion-intensity-stmt-rep-actor
                if parts[2] in RAV_MAP:
                    rows.append({"path": os.path.join(root, f), "file": f,
                                 "emotion": RAV_MAP[parts[2]], "actor": int(parts[6]),
                                 "intensity": "normal" if parts[3] == "01" else "strong"})
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def load_ravdess():
    df = ravdess_table()
    cache = os.path.join(CACHE, "ravdess_emb.npy")
    if os.path.exists(cache) and len(np.load(cache, mmap_mode="r")) == len(df):
        E = np.load(cache).astype(np.float32)
    else:
        print(f"Embedding {len(df)} RAVDESS clips ...", flush=True)
        E = np.stack(Parallel(n_jobs=4)(delayed(_embed)(p) for p in df.path))
        np.save(cache, E.astype(np.float16))
    y = np.array([LABELS.index(e) for e in df.emotion])
    return df, E, y


def _embed(p):
    import torch
    torch.set_num_threads(2)
    return ssl.embed_file(p)


# ----------------------------------------------------- representation ----
def representations():
    reps = {f"layer {l}": [l] for l in range(ssl.N_LAYERS)}
    reps.update({"mean layers 0-12": list(range(13)), "mean layers 3-9": list(range(3, 10)),
                 "mean layers 4-8": list(range(4, 9)), "mean layers 6-12": list(range(6, 13))})
    return reps


def pool(E, layers):
    return E[:, layers, :].mean(axis=1)


def probe(Xtr, ytr, Xva, yva):
    m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.01, max_iter=3000))])
    return accuracy_score(yva, m.fit(Xtr, ytr).predict(Xva))


# --------------------------------------------------------------- models ---
def model_grid():
    return {
        "KNN": (KNeighborsClassifier(),
                {"n_neighbors": [5, 11, 21, 41], "weights": ["uniform", "distance"],
                 "metric": ["euclidean", "cosine"]}, False),
        "Logistic Regression": (LogisticRegression(max_iter=5000),
                                {"C": [0.001, 0.003, 0.01, 0.03, 0.1]}, False),
        "Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}, True),
        "SVM": (SVC(kernel="rbf", probability=True, random_state=SEED),
                {"C": [1, 3, 10, 30], "gamma": ["scale", 3e-4, 1e-3]}, False),
        "Neural Network": (MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=20,
                                         random_state=SEED),
                           {"hidden_layer_sizes": [(256,), (512,), (512, 128)],
                            "alpha": [1e-2, 1e-1, 1.0]}, False),
    }


def make(est, params, pca=False):
    steps = [("scaler", StandardScaler())]
    if pca:   # Gaussian NB assumes independent features -> decorrelate first
        steps.append(("pca", PCA(n_components=128, whiten=True, random_state=SEED)))
    steps.append(("clf", clone(est).set_params(**params)))
    return Pipeline(steps)


def sentiment_proba(P):
    S = np.zeros((len(P), 3))
    for e in range(6):
        S[:, SENT_OF[e]] += P[:, e]
    return S


def scores(model, X, y):
    P = model.predict_proba(X)
    yp = model.classes_[P.argmax(1)]
    sp = sentiment_proba(P).argmax(1)
    return {"emo_acc": accuracy_score(y, yp), "emo_f1": f1_score(y, yp, average="macro"),
            "sent_acc": accuracy_score(SENT_OF[y], sp),
            "sent_f1": f1_score(SENT_OF[y], sp, average="macro"),
            "sent_bal": balanced_accuracy_score(SENT_OF[y], sp)}, yp, sp


def cm_plot(y, yp, names, title, path):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cm = confusion_matrix(y, yp, labels=range(len(names)), normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, colorbar=False,
                                                         values_format=".2f", cmap="Blues")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- main ---
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    split, E, y = load_crema()
    tr, va, te = (split.split == s for s in ("train", "val", "test"))
    tr, va, te = tr.values, va.values, te.values
    trva = tr | va

    # 1) choose the representation on validation speakers
    t = time.time()
    reps = representations()
    probe_res = Parallel(n_jobs=6)(delayed(probe)(pool(E[tr], L), y[tr], pool(E[va], L), y[va])
                                   for L in reps.values())
    probe_df = pd.DataFrame({"representation": list(reps), "val_acc": probe_res})
    probe_df.to_csv(os.path.join(OUT_DIR, "layer_probe.csv"), index=False)
    best_rep = probe_df.sort_values("val_acc", ascending=False).iloc[0].representation
    layers = reps[best_rep]
    print(probe_df.to_string(index=False))
    print(f"-> chosen representation: {best_rep} ({time.time() - t:.0f}s)\n", flush=True)

    per_layer = probe_df[probe_df.representation.str.startswith("layer")]
    plt.figure(figsize=(7, 3.6))
    plt.plot(range(13), per_layer.val_acc, "o-")
    plt.xlabel("WavLM layer (0 = CNN features)")
    plt.ylabel("validation accuracy")
    plt.title("Emotion information per WavLM layer (linear probe)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "layer_probe.png"), dpi=130)
    plt.close()

    X = pool(E, layers)
    rav_df, Er, yr = load_ravdess()
    Xr = pool(Er, layers)

    maj = np.mean(SENT_OF[y[te]] == 0)
    print(f"Sentiment majority baseline (always 'Negative'): {maj:.3f}; "
          "balanced accuracy of that baseline = 0.333")
    with open(os.path.join(OUT_DIR, "baselines.txt"), "w") as fh:
        fh.write("6-emotion chance: 0.167\n"
                 f"sentiment majority baseline accuracy: {maj:.4f}\n"
                 "sentiment majority baseline balanced accuracy: 0.3333\n")

    # 2) the five classifiers
    rows, fitted = [], {}
    for name, (est, grid, use_pca) in model_grid().items():
        t = time.time()
        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        val = Parallel(n_jobs=-1)(delayed(lambda p: accuracy_score(
            y[va], make(est, p, use_pca).fit(X[tr], y[tr]).predict(X[va])))(p) for p in combos)
        params = combos[int(np.argmax(val))]
        model = make(est, params, use_pca).fit(X[trva], y[trva])
        s_te, yp, sp = scores(model, X[te], y[te])
        s_rv, _, _ = scores(model, Xr, yr)
        fitted[name] = model
        rows.append({"model": name, "params": json.dumps({k: str(v) for k, v in params.items()}),
                     "val_emo_acc": round(max(val), 4),
                     "test_emo_acc": round(s_te["emo_acc"], 4), "test_emo_f1": round(s_te["emo_f1"], 4),
                     "test_sent_acc": round(s_te["sent_acc"], 4), "test_sent_f1": round(s_te["sent_f1"], 4),
                     "test_sent_bal_acc": round(s_te["sent_bal"], 4),
                     "ravdess_emo_acc": round(s_rv["emo_acc"], 4),
                     "ravdess_sent_acc": round(s_rv["sent_acc"], 4),
                     "ravdess_sent_bal_acc": round(s_rv["sent_bal"], 4),
                     "time_s": round(time.time() - t)})
        print({k: v for k, v in rows[-1].items() if k != "params"}, params, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    ax = table.set_index("model")[["test_emo_acc", "test_sent_acc", "ravdess_emo_acc",
                                   "ravdess_sent_acc"]].rename(columns={
        "test_emo_acc": "6 emotions (CREMA-D test speakers)",
        "test_sent_acc": "sentiment (CREMA-D test speakers)",
        "ravdess_emo_acc": "6 emotions (RAVDESS, other corpus)",
        "ravdess_sent_acc": "sentiment (RAVDESS, other corpus)"}).plot.bar(
        figsize=(10, 4.8), rot=0, width=0.8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("")
    ax.set_title("WavLM embeddings + scikit-learn classifiers")
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=130)
    plt.close()

    best = table.sort_values("val_emo_acc", ascending=False).iloc[0].model
    model = fitted[best]
    s_te, yp, sp = scores(model, X[te], y[te])
    cm_plot(y[te], yp, NAMES, f"{best}: 6 emotions (test speakers)",
            os.path.join(OUT_DIR, "cm_emotion.png"))
    cm_plot(SENT_OF[y[te]], sp, SENT_NAMES, f"{best}: sentiment (test speakers)",
            os.path.join(OUT_DIR, "cm_sentiment.png"))
    with open(os.path.join(OUT_DIR, "report_best.txt"), "w") as fh:
        fh.write(f"{best} on {best_rep}\n\n6 emotions:\n")
        fh.write(classification_report(y[te], yp, target_names=NAMES, digits=3))
        fh.write("\nSentiment:\n")
        fh.write(classification_report(SENT_OF[y[te]], sp, target_names=SENT_NAMES, digits=3))

    # 3) multi-corpus training: CREMA-D (train+val) + RAVDESS actors 1-16
    rav_tr = rav_df.actor.values <= 16
    rav_te = ~rav_tr
    Xm = np.vstack([X[trva], Xr[rav_tr]])
    ym = np.concatenate([y[trva], yr[rav_tr]])
    multi = clone(model).fit(Xm, ym)
    single_rav = scores(model, Xr[rav_te], yr[rav_te])[0]
    multi_rav, myp, msp = scores(multi, Xr[rav_te], yr[rav_te])
    multi_crema = scores(multi, X[te], y[te])[0]
    mc = pd.DataFrame([
        {"training data": "CREMA-D only", "CREMA-D test emo": s_te["emo_acc"],
         "CREMA-D test sent": s_te["sent_acc"], "RAVDESS 17-24 emo": single_rav["emo_acc"],
         "RAVDESS 17-24 sent": single_rav["sent_acc"]},
        {"training data": "CREMA-D + RAVDESS 1-16", "CREMA-D test emo": multi_crema["emo_acc"],
         "CREMA-D test sent": multi_crema["sent_acc"], "RAVDESS 17-24 emo": multi_rav["emo_acc"],
         "RAVDESS 17-24 sent": multi_rav["sent_acc"]}]).round(4)
    mc.to_csv(os.path.join(OUT_DIR, "multi_corpus.csv"), index=False)
    print("\nMulti-corpus:\n", mc.to_string(index=False))
    cm_plot(yr[rav_te], myp, NAMES, "Multi-corpus model: RAVDESS held-out actors",
            os.path.join(OUT_DIR, "cm_ravdess_multi.png"))

    # 4) deployed model: multi-corpus, all available data
    final = clone(model).fit(np.vstack([X, Xr]), np.concatenate([y, yr]))
    os.makedirs(MODEL_DIR, exist_ok=True)
    # corpus features are shipped with the model so the app can personalise it
    np.savez_compressed(os.path.join(MODEL_DIR, "train_embeddings.npz"),
                        X=np.vstack([X, Xr]).astype(np.float16), y=np.concatenate([y, yr]))
    joblib.dump({"model": final, "model_name": best, "representation": best_rep,
                 "layers": layers, "emotions": NAMES, "sentiments": SENT_NAMES,
                 "sentiment_of": SENT_OF.tolist(), "ssl_model": ssl.MODEL_NAME,
                 "lowpass_hz": 4000},
                os.path.join(MODEL_DIR, "wavlm_emotion.joblib"))

    print("\n", table.drop(columns=["params"]).to_string(index=False))
    print(f"\nBest (validation): {best}. Deployed model saved to models/wavlm_emotion.joblib")


if __name__ == "__main__":
    main()
