"""
Q3 - Surface defect detection & classification (paper reel: paper / hole / penMark).

Usage (from the ml_cep folder):
    python q3/q3_defect_detection.py            # full pipeline
    python q3/q3_defect_detection.py --no-aug   # without data augmentation

Steps
 1. Stratified 70/15/15 split of the 1500 images (train / val / test).
 2. Feature extraction (defect_features.py). Training (and validation, for
    the final refit) images are augmented x4 (original, rot90, rot180,
    horizontal flip) - the paper surface has no preferred orientation.
 3. Task (i) three-class classification: SVM, Naive Bayes, Logistic
    Regression and Neural Network (MLP), each tuned on the validation set
    for three feature sets (global / patch / all), refit on train+val and
    evaluated on the untouched test set.
 4. Task (ii) defect localisation: the trained region (patch-feature)
    classifier scores the top-5 dark-region proposals of an image; the
    proposal with the highest P(defect) is the defect and gets a bounding box.
    Box images are saved for every test image with a defect.

Outputs: q3/results/, models in q3/models/.
"""
import argparse
import itertools
import json
import os
import time

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import defect_features as dfx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "out")
OUT_DIR = os.path.join(HERE, "results")
MODEL_DIR = os.path.join(HERE, "models")
CACHE_DIR = os.path.join(HERE, "cache")
SEED = 42
CLASSES = ["paper", "hole", "penMark"]        # label 0 = no defect
AUGS = ("orig", "rot90", "rot180", "flip")


# ------------------------------------------------------------------ data --
def index_dataset():
    rows = []
    for ci, c in enumerate(CLASSES):
        for fn in sorted(os.listdir(os.path.join(DATA_DIR, c)), key=lambda s: int(s.split(".")[0])):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                rows.append({"path": os.path.join(DATA_DIR, c, fn), "file": f"{c}/{fn}",
                             "cls": c, "y": ci})
    df = pd.DataFrame(rows)
    tr, tmp = train_test_split(df.index, test_size=0.30, random_state=SEED, stratify=df.y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=df.y[tmp])
    df["split"] = ""
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    return df


def extract(df, use_aug=True):
    tag = "aug" if use_aug else "noaug"
    cache = os.path.join(CACHE_DIR, f"features_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if list(d["files"]) == list(df.file):
            print(f"Loaded cached features: {cache}")
            return list(d["feats"])
    print(f"Extracting features for {len(df)} images (augmentation={use_aug})...")
    t = time.time()
    jobs = [AUGS if (use_aug and s != "test") else ("orig",) for s in df.split]
    feats = Parallel(n_jobs=-1, batch_size=8)(
        delayed(dfx.image_features)(p, a) for p, a in zip(df.path, jobs))
    feats = [np.stack(f) for f in feats]
    os.makedirs(CACHE_DIR, exist_ok=True)
    arr = np.empty(len(feats), dtype=object)
    arr[:] = feats
    np.savez_compressed(cache, feats=arr, files=np.array(df.file))
    print(f"  done in {time.time() - t:.0f}s")
    return feats


def stack(feats, idx, cols, all_augs):
    """Rows for images idx; all augmented copies if all_augs else originals."""
    X, rows = [], []
    for i in idx:
        f = feats[i] if all_augs else feats[i][:1]
        X.append(f[:, cols])
        rows += [i] * len(f)
    return np.vstack(X), np.array(rows)


# ---------------------------------------------------------------- models --
def model_grid():
    return {
        "SVM": (SVC(kernel="rbf", random_state=SEED),
                {"C": [0.3, 1, 3, 10, 30], "gamma": ["scale", 1e-4, 3e-4]}),
        "Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]}),
        "Logistic Regression": (LogisticRegression(max_iter=4000),
                                {"C": [0.001, 0.01, 0.1, 1]}),
        "Neural Network": (MLPClassifier(max_iter=500, early_stopping=True, n_iter_no_change=15,
                                         random_state=SEED),
                           {"hidden_layer_sizes": [(128,), (256,), (256, 64)],
                            "alpha": [1e-3, 1e-2, 1e-1, 1.0]}),
    }


def make_pipe(est, params):
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(est).set_params(**params))])


def tune(est, grid, Xtr, ytr, Xva, yva):
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]

    def run(p):
        return accuracy_score(yva, make_pipe(est, p).fit(Xtr, ytr).predict(Xva))

    scores = Parallel(n_jobs=-1)(delayed(run)(p) for p in combos)
    b = int(np.argmax(scores))
    return combos[b], scores[b]


# ---------------------------------------------------------- localisation --
K_PROPOSALS = 5


def extract_regions(df, k=K_PROPOSALS):
    """Patch features of the top-k proposals of every image (cached).
    Region labels: the top proposal of a defect image gets the image class,
    every other proposal (all proposals of clean paper, and lower-ranked
    proposals of defect images - mostly static blemishes / texture) is
    'paper' (= background). Lower-ranked proposals act as hard negatives."""
    cache = os.path.join(CACHE_DIR, f"regions_k{k}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if list(d["files"]) == list(df.file):
            print(f"Loaded cached region features: {cache}")
            return d["X"], d["img"], d["rank"]
    print(f"Extracting top-{k} proposal features for {len(df)} images...")
    t = time.time()
    res = Parallel(n_jobs=-1, batch_size=8)(
        delayed(dfx.proposal_features)(p, k) for p in df.path)
    X = np.vstack([r[0] for r in res])
    img = np.concatenate([[i] * len(r[0]) for i, r in enumerate(res)])
    rank = np.concatenate([np.arange(len(r[0])) for r in res])
    np.savez_compressed(cache, X=X, img=img, rank=rank, files=np.array(df.file))
    print(f"  done in {time.time() - t:.0f}s, {len(X)} proposals")
    return X, img, rank


def region_labels(y_img, img, rank):
    return np.where(rank == 0, y_img[img], 0)


def multi_proposal_predict(P, img):
    """Image decision from per-proposal probabilities: the image is defective
    if any proposal is; its class/box come from the most defect-like one.
    Returns dict image_index -> (class, row index of chosen proposal)."""
    out = {}
    for i in np.unique(img):
        rows = np.flatnonzero(img == i)
        dp = 1 - P[rows, 0]
        b = rows[int(np.argmax(dp))]
        out[i] = (int(np.argmax(P[b])), b)
    return out


def localize(path, region_model, k=K_PROPOSALS):
    """Score the top-k dark-region proposals with the trained region
    classifier. Returns (best proposal, its class probabilities, scale,
    all proposals, P(defect) of each proposal)."""
    X, cands, scale = dfx.proposal_features(path, k)
    P = region_model.predict_proba(X)                      # columns = classes 0,1,2
    defect_p = 1 - P[:, 0]
    b = int(np.argmax(defect_p))
    return cands[b], P[b], scale, cands, defect_p


def box_in_original(cand, scale, pad=6):
    x, y, w, h = cand["bbox"]
    return (int(x * scale) - pad, int(y * scale) - pad,
            int((x + w) * scale) + pad, int((y + h) * scale) + pad)


def draw_box(path, cand, scale, text, color=(0, 0, 255)):
    img = cv2.imread(path)
    x0, y0, x1, y1 = box_in_original(cand, scale)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 6)
    cv2.putText(img, text, (max(5, x0), max(60, y0 - 15)), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, color, 5, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-aug", action="store_true", help="disable data augmentation")
    args = ap.parse_args()
    use_aug = not args.no_aug
    for d in (OUT_DIR, MODEL_DIR, CACHE_DIR, os.path.join(OUT_DIR, "localization")):
        os.makedirs(d, exist_ok=True)

    df = index_dataset()
    df.to_csv(os.path.join(OUT_DIR, "split.csv"), index=False)
    print(pd.crosstab(df.split, df.cls), "\n")
    feats = extract(df, use_aug)
    y_img = df.y.values
    tr = np.flatnonzero(df.split == "train")
    va = np.flatnonzero(df.split == "val")
    te = np.flatnonzero(df.split == "test")
    sl = dfx.group_slices()

    def cols_of(groups):
        return np.concatenate([np.arange(sl[g].start, sl[g].stop) for g in groups])

    feature_sets = {"global": cols_of(dfx.GLOBAL_GROUPS),
                    "patch": cols_of(dfx.PATCH_GROUPS),
                    "all": cols_of(dfx.ALL_GROUPS)}

    rows, preds, fitted = [], {}, {}
    for fs_name, cols in feature_sets.items():
        Xtr, rtr = stack(feats, tr, cols, use_aug)
        Xva, rva = stack(feats, va, cols, False)
        Xfit, rfit = stack(feats, np.concatenate([tr, va]), cols, use_aug)
        Xte, rte = stack(feats, te, cols, False)
        for name, (est, grid) in model_grid().items():
            t = time.time()
            params, vacc = tune(est, grid, Xtr, y_img[rtr], Xva, y_img[rva])
            final_params = dict(params)
            if name == "SVM":           # Platt probabilities only for the final model
                final_params["probability"] = True    # (needed for localisation scores)
            model = make_pipe(est, final_params).fit(Xfit, y_img[rfit])
            yp = model.predict(Xte)
            acc, f1 = accuracy_score(y_img[rte], yp), f1_score(y_img[rte], yp, average="macro")
            rows.append({"features": fs_name, "model": name, "n_features": len(cols),
                         "best_params": json.dumps({k: str(v) for k, v in params.items()}),
                         "val_acc": round(vacc, 4), "test_acc": round(acc, 4),
                         "test_macro_f1": round(f1, 4), "time_s": round(time.time() - t, 1)})
            preds[(fs_name, name)] = yp
            fitted[(fs_name, name)] = model
            print(f"[{fs_name:6s}] {name:20s} val={vacc:.4f} test={acc:.4f} F1={f1:.4f} "
                  f"{params} ({time.time() - t:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print("\n", table.to_string(index=False))

    # bar chart: test accuracy per classifier and feature set
    piv = table.pivot(index="model", columns="features", values="test_acc")[["global", "patch", "all"]]
    ax = piv.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Paper defect classification (3 classes) - test accuracy")
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=130)
    plt.close()

    # confusion matrices for the 'all' feature set
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, name in zip(axes, model_grid()):
        yp = preds[("all", name)]
        cm = confusion_matrix(y_img[te], yp, labels=[0, 1, 2])
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} (all features)\nacc={accuracy_score(y_img[te], yp):.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=130)
    plt.close(fig)
    for (fs_name, name), yp in preds.items():
        if fs_name == "all":
            with open(os.path.join(OUT_DIR, f"report_{name.replace(' ', '_')}.txt"), "w") as fh:
                fh.write(classification_report(y_img[te], yp, labels=[0, 1, 2],
                                               target_names=CLASSES, digits=4))

    # ---------------- region classifier (scores every proposal) -----------
    # Same 4 classifier families, trained on the patch features of the top-5
    # proposals of each image (hard negatives included). Hyper-parameters
    # are chosen by IMAGE-level validation accuracy of the multi-proposal
    # decision, which is what the region classifier is used for.
    XR, img_of, rank = extract_regions(df)
    yR = region_labels(y_img, img_of, rank)
    in_tr, in_va = np.isin(img_of, tr), np.isin(img_of, va)
    in_fit, in_te = in_tr | in_va, np.isin(img_of, te)
    print(f"Region set: {in_tr.sum()} train proposals "
          f"({(yR[in_tr] != 0).sum()} defects, {(yR[in_tr] == 0).sum()} background)")

    def mp_accuracy(model, mask):
        P = model.predict_proba(XR[mask])
        dec = multi_proposal_predict(P, img_of[mask])
        return np.mean([dec[i][0] == y_img[i] for i in dec]), dec

    region_rows, region_fitted = [], {}
    for name, (est, grid) in model_grid().items():
        t = time.time()
        if name == "SVM":
            est = clone(est).set_params(probability=True)
        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]

        def run(p, est=est):
            m = make_pipe(est, p).fit(XR[in_tr], yR[in_tr])
            return mp_accuracy(m, in_va)[0]

        scores = Parallel(n_jobs=-1)(delayed(run)(p) for p in combos)
        params, vacc = combos[int(np.argmax(scores))], max(scores)
        model = make_pipe(est, params).fit(XR[in_fit], yR[in_fit])
        tacc, dec = mp_accuracy(model, in_te)
        yp = np.array([dec[i][0] for i in te])
        region_rows.append({"features": "multi-proposal (patch)", "model": name,
                            "best_params": json.dumps({k: str(v) for k, v in params.items()}),
                            "val_acc": round(vacc, 4), "test_acc": round(tacc, 4),
                            "test_macro_f1": round(f1_score(y_img[te], yp, average="macro"), 4),
                            "time_s": round(time.time() - t, 1)})
        region_fitted[name] = model
        preds[("multi", name)] = yp
        print(f"[region] {name:20s} val={vacc:.4f} test={tacc:.4f} {params} "
              f"({time.time() - t:.0f}s)", flush=True)
    rtable = pd.DataFrame(region_rows)
    rtable.to_csv(os.path.join(OUT_DIR, "region_classifier_comparison.csv"), index=False)
    print("\n", rtable.to_string(index=False))

    # ---------------- choose final models (validation only) ----------------
    best = table.sort_values(["val_acc", "test_macro_f1"], ascending=False).iloc[0]
    best_key = (best.features, best.model)
    rbest = rtable.sort_values("val_acc", ascending=False).iloc[0]
    region_model = region_fitted[rbest.model]
    use_multi = rbest.val_acc > best.val_acc
    joblib.dump({"model": fitted[best_key], "cols": feature_sets[best_key[0]],
                 "features": best_key[0], "model_name": best_key[1], "classes": CLASSES,
                 "decision": "multi-proposal" if use_multi else "single-candidate"},
                os.path.join(MODEL_DIR, "image_classifier.joblib"))
    joblib.dump({"model": region_model, "model_name": rbest.model, "classes": CLASSES,
                 "k": K_PROPOSALS},
                os.path.join(MODEL_DIR, "region_classifier.joblib"))
    print(f"\nBest single-candidate image classifier: {best_key} (val {best.val_acc}) | "
          f"best region classifier: {rbest.model} (val {rbest.val_acc}) -> final image "
          f"decision = {'multi-proposal' if use_multi else 'single-candidate'}")

    final_pred = preds[("multi", rbest.model)] if use_multi else preds[best_key]
    with open(os.path.join(OUT_DIR, "final_classifier_report.txt"), "w") as fh:
        fh.write(f"Final decision: {'multi-proposal ' + rbest.model if use_multi else best_key}\n\n")
        fh.write(classification_report(y_img[te], final_pred, labels=[0, 1, 2],
                                       target_names=CLASSES, digits=4))
        fh.write("\nconfusion matrix (rows = true paper/hole/penMark):\n")
        fh.write(str(confusion_matrix(y_img[te], final_pred, labels=[0, 1, 2])))

    # ---------------- task (ii): localisation on every test image ----------
    loc_rows = []
    for j, i in enumerate(te):
        path = df.path[i]
        cand, prob, scale, cands, dp = localize(path, region_model)
        img_pred = int(final_pred[j])
        x0, y0, x1, y1 = box_in_original(cand, scale)
        loc_rows.append({"file": df.file[i], "true": df.cls[i], "pred": CLASSES[img_pred],
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "p_defect": round(float(1 - prob[0]), 4),
                         "proposal_rank": int(np.argmax(dp)) + 1})
        if img_pred != 0:
            txt = f"{CLASSES[img_pred]} {1 - prob[0]:.2f}"
            out = draw_box(path, cand, scale, txt)
            cv2.imwrite(os.path.join(OUT_DIR, "localization",
                                     df.file[i].replace("/", "_")), cv2.resize(out, (800, 800)))
    loc = pd.DataFrame(loc_rows)
    loc.to_csv(os.path.join(OUT_DIR, "localization_test.csv"), index=False)
    dfct = loc[loc.true != "paper"]
    print(f"Localisation: boxes drawn for {(loc.pred != 'paper').sum()} test images -> "
          f"results/localization/. On defect images the chosen proposal was the darkest "
          f"region in {(dfct.proposal_rank == 1).sum()}/{len(dfct)} cases.")

    # contact sheet of 24 random defect test images with boxes
    sample = loc[loc.true != "paper"].sample(24, random_state=SEED)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12.5))
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, r.file)), cv2.COLOR_BGR2RGB)
        cv2.rectangle(img, (r.x0, r.y0), (r.x1, r.y1), (255, 0, 0), 10)
        ax.imshow(img)
        ax.set_title(f"{r.file}\npred={r.pred}", fontsize=9,
                     color="black" if r.pred == r.true else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "localization_examples.png"), dpi=90)
    plt.close(fig)

    # zoomed crops (the defects are tiny at full-image scale)
    fig, axes = plt.subplots(4, 6, figsize=(15, 10.5))
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, r.file)), cv2.COLOR_BGR2RGB)
        cx, cy = (r.x0 + r.x1) // 2, (r.y0 + r.y1) // 2
        half = max(120, int(0.9 * max(r.x1 - r.x0, r.y1 - r.y0)))
        cv2.rectangle(img, (r.x0, r.y0), (r.x1, r.y1), (255, 0, 0), 3)
        crop = img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        ax.imshow(crop)
        ax.set_title(f"{r.file} -> {r.pred}", fontsize=9,
                     color="black" if r.pred == r.true else "red")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "localization_zoom.png"), dpi=90)
    plt.close(fig)

    # preprocessing illustration
    ex = df[df.split == "test"].groupby("cls").head(1)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, (_, r) in zip(axes, ex.iterrows()):
        s, z, _ = dfx.preprocess(dfx.load_gray(r.path))
        c = dfx.find_candidates(z, k=1)[0]
        row[0].imshow(s, cmap="gray"); row[0].set_title(f"{r.cls}: grayscale (1000 px)")
        row[1].imshow(np.clip(z, -30, 10), cmap="magma"); row[1].set_title("illumination-flattened z map")
        patch, _ = dfx.crop_patch(z, c)
        row[2].imshow(np.clip(patch, -30, 10), cmap="magma"); row[2].set_title("top candidate patch")
        for a in row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "preprocessing_pipeline.png"), dpi=90)
    plt.close(fig)
    print(f"\nAll results in {OUT_DIR}")


if __name__ == "__main__":
    main()
