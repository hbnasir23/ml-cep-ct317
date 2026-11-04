"""
Q2 - Sentiment (emotion) analysis from audio recordings (CREMA-D, 6 emotions).

Usage (from the ml_cep folder):
    python q2/q2_audio_sentiment.py                 # full pipeline
    python q2/q2_audio_sentiment.py --cutoff-study  # + compare low-pass cut-offs

Steps
 1. Index the 7,442 clips; label = emotion code in the file name
    (ActorID_Sentence_Emotion_Level.wav).
 2. Speaker-independent split: actors (not clips) are divided 70/15/15 into
    train/validation/test, so no voice appears in more than one set. A random
    clip split would let the model recognise the speaker instead of the
    emotion and give over-optimistic scores.
 3. Low-pass filter (Butterworth, 4 kHz) + silence trimming, then extract the
    242-dimensional feature vector (audio_features.py). Cached to disk.
 4. Tune KNN, Logistic Regression, Naive Bayes, SVM and Neural Network (MLP)
    on the validation set, refit the best setting on train+val, evaluate once
    on the untouched test set.
 5. Extra analysis: per-feature-group ablation and (optional) cut-off study.

Outputs go to q2/results/, the best model to q2/models/best_model.joblib.
"""
import argparse
import itertools
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, ConfusionMatrixDisplay)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import audio_features as af

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(HERE, "dataset_repo", "NoiseAudioWAV")
OUT_DIR = os.path.join(HERE, "results")
MODEL_DIR = os.path.join(HERE, "models")
CACHE_DIR = os.path.join(HERE, "cache")
SEED = 42
EMOTIONS = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear",
            "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
LABELS = list(EMOTIONS)


# ---------------------------------------------------------------- data ----
def index_dataset(audio_dir=AUDIO_DIR):
    rows = []
    for fn in sorted(os.listdir(audio_dir)):
        if not fn.lower().endswith(".wav"):
            continue
        actor, sentence, emo, level = fn[:-4].split("_")
        rows.append({"path": os.path.join(audio_dir, fn), "file": fn, "actor": actor,
                     "sentence": sentence, "emotion": emo, "level": level})
    return pd.DataFrame(rows)


def speaker_independent_split(df, seed=SEED):
    """70% / 15% / 15% of ACTORS -> train / val / test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, tmp_idx = next(gss.split(df, groups=df.actor))
    tmp = df.iloc[tmp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_rel, te_rel = next(gss2.split(tmp, groups=tmp.actor))
    split = np.empty(len(df), dtype=object)
    split[tr_idx] = "train"
    split[tmp_idx[va_rel]] = "val"
    split[tmp_idx[te_rel]] = "test"
    df = df.copy()
    df["split"] = split
    a = {s: set(df.actor[df.split == s]) for s in ("train", "val", "test")}
    assert not (a["train"] & a["val"] or a["train"] & a["test"] or a["val"] & a["test"])
    return df


def extract_all(df, cutoff=af.LOWPASS_HZ, n_jobs=-1):
    cache = os.path.join(CACHE_DIR, f"features_lp{cutoff}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if list(d["files"]) == list(df.file):
            print(f"Loaded cached features: {cache}")
            return d["X"]
    print(f"Extracting features for {len(df)} clips (low-pass {cutoff} Hz)...")
    t = time.time()
    feats = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(af.features_from_file)(p, cutoff) for p in df.path)
    X = np.vstack(feats)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, X=X, files=np.array(df.file))
    print(f"  done in {time.time() - t:.0f}s, X shape = {X.shape}")
    return X


# -------------------------------------------------------------- models ----
def model_grid():
    """name -> (estimator, parameter grid). Every model is preceded by a
    StandardScaler (distance/gradient based models need comparable scales)."""
    return {
        "KNN": (KNeighborsClassifier(),
                {"n_neighbors": [3, 5, 9, 15, 21, 31], "weights": ["uniform", "distance"],
                 "metric": ["euclidean", "cosine"]}),
        "Logistic Regression": (LogisticRegression(max_iter=3000),
                                {"C": [0.001, 0.01, 0.1, 1, 10]}),
        "Naive Bayes": (GaussianNB(),
                        {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]}),
        "SVM": (SVC(kernel="rbf"),
                {"C": [0.3, 1, 3, 10, 30], "gamma": ["scale", 0.001, 0.003, 0.01]}),
        "Neural Network": (MLPClassifier(max_iter=600, early_stopping=True,
                                         n_iter_no_change=20, random_state=SEED),
                           {"hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256)],
                            "alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
    }


def make_pipeline(est, params):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", clone(est).set_params(**params))])


def tune_on_validation(est, grid, Xtr, ytr, Xva, yva):
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    def run(p):
        m = make_pipeline(est, p).fit(Xtr, ytr)
        return accuracy_score(yva, m.predict(Xva))

    scores = Parallel(n_jobs=-1)(delayed(run)(p) for p in combos)
    best = int(np.argmax(scores))
    return combos[best], scores[best], list(zip(combos, scores))


# ---------------------------------------------------------------- plots ---
def plot_lowpass_demo(path, cutoff, out):
    import librosa
    import librosa.display
    y, sr = librosa.load(path, sr=af.SR)
    yf = af.lowpass_filter(y, sr, cutoff)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for col, (sig, title) in enumerate([(y, "Original (noisy)"), (yf, f"Low-pass {cutoff} Hz")]):
        t = np.arange(len(sig)) / sr
        ax[0, col].plot(t, sig, lw=0.5)
        ax[0, col].set_title(f"{title} - waveform")
        ax[0, col].set_xlabel("time (s)")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=160, x_axis="time", y_axis="hz", ax=ax[1, col])
        ax[1, col].set_title(f"{title} - spectrogram")
    fig.suptitle(os.path.basename(path))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusions(preds, y_true, out):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4.4))
    for ax, (name, yp) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, yp, labels=list(range(len(LABELS))), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
            ax=ax, colorbar=False, values_format=".2f", cmap="Blues")
        ax.set_title(f"{name}\nacc={accuracy_score(y_true, yp):.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_bars(table, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    w = 0.27
    ax.bar(x - w, table["val_acc"], w, label="Validation acc")
    ax.bar(x, table["test_acc"], w, label="Test acc")
    ax.bar(x + w, table["test_macro_f1"], w, label="Test macro-F1")
    ax.axhline(1 / 6, color="gray", ls="--", lw=1, label="Chance (1/6)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Emotion classification on unseen speakers")
    ax.legend()
    for i, v in enumerate(table["test_acc"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-study", action="store_true",
                    help="also compare low-pass cut-offs 2/3/4/6 kHz and no filter")
    args = ap.parse_args()
    for d in (OUT_DIR, MODEL_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)

    df = speaker_independent_split(index_dataset())
    df.to_csv(os.path.join(OUT_DIR, "split.csv"), index=False)
    summary = df.groupby(["split", "emotion"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary["actors"] = df.groupby("split").actor.nunique()
    print("Speaker-independent split:\n", summary, "\n")
    summary.to_csv(os.path.join(OUT_DIR, "split_summary.csv"))

    plot_lowpass_demo(df.path.iloc[0], af.LOWPASS_HZ, os.path.join(OUT_DIR, "lowpass_demo.png"))

    X = extract_all(df)
    # integer labels 0..5 in LABELS order (MLP early-stopping needs numeric y)
    y = np.array([LABELS.index(e) for e in df.emotion])
    tr, va, te = (df.split == "train").values, (df.split == "val").values, (df.split == "test").values
    trva = tr | va

    rows, preds, tuning_log = [], {}, {}
    best_overall = None
    for name, (est, grid) in model_grid().items():
        t = time.time()
        params, val_acc, log = tune_on_validation(est, grid, X[tr], y[tr], X[va], y[va])
        tuning_log[name] = [{"params": {k: str(v) for k, v in p.items()}, "val_acc": s}
                            for p, s in log]
        final = make_pipeline(est, params).fit(X[trva], y[trva])   # refit on train+val
        yp = final.predict(X[te])
        preds[name] = yp
        acc, f1 = accuracy_score(y[te], yp), f1_score(y[te], yp, average="macro")
        rows.append({"model": name, "best_params": json.dumps({k: str(v) for k, v in params.items()}),
                     "val_acc": round(val_acc, 4), "test_acc": round(acc, 4),
                     "test_macro_f1": round(f1, 4), "time_s": round(time.time() - t, 1)})
        print(f"{name:20s} val={val_acc:.4f} test={acc:.4f} macroF1={f1:.4f} {params}")
        with open(os.path.join(OUT_DIR, f"report_{name.replace(' ', '_')}.txt"), "w") as fh:
            fh.write(f"{name}  params={params}\n\n")
            fh.write(classification_report(y[te], yp, labels=list(range(len(LABELS))),
                                           target_names=[EMOTIONS[l] for l in LABELS], digits=3))
        if best_overall is None or val_acc > best_overall[1]:
            best_overall = (name, val_acc, final)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    with open(os.path.join(OUT_DIR, "tuning_log.json"), "w") as fh:
        json.dump(tuning_log, fh, indent=1)
    plot_bars(table, os.path.join(OUT_DIR, "model_comparison.png"))
    plot_confusions(preds, y[te], os.path.join(OUT_DIR, "confusion_matrices.png"))
    print("\n", table.to_string(index=False))

    name, _, model = best_overall
    joblib.dump({"model": model, "model_name": name,
                 "labels": [EMOTIONS[l] for l in LABELS],
                 "lowpass_hz": af.LOWPASS_HZ, "feature_groups": af.FEATURE_GROUPS},
                os.path.join(MODEL_DIR, "best_model.joblib"))
    print(f"\nBest model on validation: {name} -> saved to models/best_model.joblib")

    # ---- feature-group ablation with the SVM setting chosen above --------
    svm_params = json.loads(table.set_index("model").loc["SVM", "best_params"])
    svm_params = {k: (float(v) if k == "C" else (v if v == "scale" else float(v)))
                  for k, v in svm_params.items()}
    slices = af.feature_group_slices()
    abl = []
    for g, s in list(slices.items()) + [("ALL", slice(0, X.shape[1]))]:
        m = make_pipeline(SVC(kernel="rbf"), svm_params).fit(X[trva][:, s], y[trva])
        abl.append({"feature_group": g, "n_features": s.stop - s.start,
                    "test_acc": round(accuracy_score(y[te], m.predict(X[te][:, s])), 4)})
    for g in slices:  # leave-one-group-out
        keep = np.ones(X.shape[1], bool)
        keep[slices[g]] = False
        m = make_pipeline(SVC(kernel="rbf"), svm_params).fit(X[trva][:, keep], y[trva])
        abl.append({"feature_group": f"ALL minus {g}", "n_features": int(keep.sum()),
                    "test_acc": round(accuracy_score(y[te], m.predict(X[te][:, keep])), 4)})
    abl = pd.DataFrame(abl)
    abl.to_csv(os.path.join(OUT_DIR, "feature_ablation.csv"), index=False)
    print("\nFeature-group ablation (SVM):\n", abl.to_string(index=False))

    # ---- optional low-pass cut-off study --------------------------------
    if args.cutoff_study:
        res = []
        for c in (2000, 3000, 4000, 6000, 7900):
            Xc = extract_all(df, cutoff=c)
            m = make_pipeline(SVC(kernel="rbf"), svm_params).fit(Xc[trva], y[trva])
            res.append({"cutoff_hz": c if c != 7900 else "none (~Nyquist)",
                        "test_acc": round(accuracy_score(y[te], m.predict(Xc[te])), 4)})
            print(res[-1])
        pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, "cutoff_study.csv"), index=False)

    print(f"\nAll results in {OUT_DIR}")


if __name__ == "__main__":
    main()
