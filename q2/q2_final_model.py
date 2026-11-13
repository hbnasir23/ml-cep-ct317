"""
Q2 final model: compare WavLM, emotion2vec+ and their combination as features
for the five scikit-learn classifiers, then deploy the best one.

    python q2/q2_final_model.py

Needs: cache/ssl/emb_*.npy + cache/ssl/ravdess_emb.npy (WavLM, from
q2_wavlm_emotion.py) and cache/e2v/emb_0.npz, emb_1.npz (emotion2vec+, from
e2v_features.py; CREMA-D clips in split order followed by RAVDESS clips).

Protocol (unchanged): feature set and hyper-parameters chosen on the CREMA-D
VALIDATION speakers only; CREMA-D test speakers and RAVDESS used for testing.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q2_wavlm_emotion import (load_crema, load_ravdess, pool, model_grid, make, scores,  # noqa: E402
                              cm_plot, sentiment_proba, SENT_OF, NAMES, SENT_NAMES, LABELS)
from personalization import _run  # noqa: E402

OUT = os.path.join(HERE, "results", "final")
MODEL_DIR = os.path.join(HERE, "models")
E2V_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]


def load_e2v(n_crema):
    parts = [np.load(os.path.join(HERE, "cache", "e2v", f"emb_{k}.npz")) for k in (0, 1)]
    E = np.concatenate([p["E"] for p in parts])
    S = np.concatenate([p["S"] for p in parts])
    labels = [str(l) for l in parts[0]["labels"]]
    head = [next(i for i, l in enumerate(labels) if k in l) for k in E2V_LABELS]
    return E[:n_crema], E[n_crema:], S[:n_crema][:, head], S[n_crema:][:, head]


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def prior_corrected_sent(P, prior):
    S = sentiment_proba(P) / prior
    return S.argmax(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    split, Ew, y = load_crema()
    rav, Ewr, yr = load_ravdess()
    tr, va, te = [(split.split == s).values for s in ("train", "val", "test")]
    trva = tr | va
    Xe, Xer, Se, Ser = load_e2v(len(split))
    Xw, Xwr = pool(Ew, [7]), pool(Ewr, [7])

    # zero-shot: emotion2vec+'s own classification head, no training on our data
    zs = {"CREMA-D test": accuracy_score(y[te], Se[te].argmax(1)),
          "RAVDESS (all)": accuracy_score(yr, Ser.argmax(1))}
    print("emotion2vec+ zero-shot (own head):", {k: round(v, 4) for k, v in zs.items()})

    feats = {"WavLM layer 7": (Xw, Xwr),
             "emotion2vec+": (Xe, Xer),
             "WavLM + emotion2vec+": (np.hstack([l2(Xw), l2(Xe)]), np.hstack([l2(Xwr), l2(Xer)]))}
    prior = np.bincount(SENT_OF[y[tr]], minlength=3) / tr.sum()

    rows, fitted = [], {}
    for fname, (X, Xr) in feats.items():
        for name, (est, grid, use_pca) in model_grid().items():
            t = time.time()
            combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
            val = Parallel(n_jobs=6)(delayed(lambda p: accuracy_score(
                y[va], make(est, p, use_pca).fit(X[tr], y[tr]).predict(X[va])))(p) for p in combos)
            params = combos[int(np.argmax(val))]
            m = make(est, params, use_pca).fit(X[trva], y[trva])
            s_te, _, _ = scores(m, X[te], y[te])
            s_rv, _, _ = scores(m, Xr, yr)
            sp_bal = balanced_accuracy_score(SENT_OF[y[te]], prior_corrected_sent(m.predict_proba(X[te]), prior))
            rows.append({"features": fname, "model": name,
                         "params": json.dumps({k: str(v) for k, v in params.items()}),
                         "val_emo": round(max(val), 4), "test_emo": round(s_te["emo_acc"], 4),
                         "test_emo_f1": round(s_te["emo_f1"], 4), "test_sent": round(s_te["sent_acc"], 4),
                         "test_sent_bal_corrected": round(sp_bal, 4),
                         "ravdess_emo": round(s_rv["emo_acc"], 4), "ravdess_sent": round(s_rv["sent_acc"], 4),
                         "time_s": round(time.time() - t)})
            fitted[(fname, name)] = (m, params, est, use_pca)
            print({k: v for k, v in rows[-1].items() if k != "params"}, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "feature_model_comparison.csv"), index=False)

    best = table.sort_values("val_emo", ascending=False).iloc[0]
    key = (best.features, best.model)
    model, params, est, use_pca = fitted[key]
    X, Xr = feats[best.features]
    print(f"\nBEST on validation: {key} (val {best.val_emo})")

    s_te, yp, sp = scores(model, X[te], y[te])
    cm_plot(y[te], yp, NAMES, f"{best.model} on {best.features}\n6 emotions, CREMA-D test speakers",
            os.path.join(OUT, "cm_emotion.png"))
    spc = prior_corrected_sent(model.predict_proba(X[te]), prior)
    cm_plot(SENT_OF[y[te]], spc, SENT_NAMES, "Sentiment (prior-corrected), CREMA-D test",
            os.path.join(OUT, "cm_sentiment.png"))

    # multi-corpus
    rtr = rav.actor.values <= 16
    multi = clone(model).fit(np.vstack([X[trva], Xr[rtr]]), np.concatenate([y[trva], yr[rtr]]))
    m_rv, myp, _ = scores(multi, Xr[~rtr], yr[~rtr])
    m_cr, _, _ = scores(multi, X[te], y[te])
    single_rv, _, _ = scores(model, Xr[~rtr], yr[~rtr])
    cm_plot(yr[~rtr], myp, NAMES, "Multi-corpus model, unseen RAVDESS actors 17-24",
            os.path.join(OUT, "cm_ravdess_multi.png"))

    # personalisation on the 8 unseen RAVDESS actors
    Xc = np.vstack([X, Xr[rtr]])
    yc = np.concatenate([y, yr[rtr]])
    base = clone(model).fit(Xc, yc)
    jobs = [(a, k, r) for a in range(17, 25) for k in (0, 1, 2, 3) for r in range(3 if k else 1)]
    res = Parallel(n_jobs=6)(delayed(_run)(base, Xc, yc, Xr[rav.actor.values == a],
                                           yr[rav.actor.values == a], k, r, SENT_OF)
                             for a, k, r in jobs)
    pers = pd.DataFrame([{"actor": a, "k": k, "emo": o[0], "sent": o[1]} for (a, k, r), o in zip(jobs, res)])
    pers_m = pers.groupby("k")[["emo", "sent"]].agg(["mean", "std"]).round(4)
    pers_m.to_csv(os.path.join(OUT, "personalization.csv"))

    summary = {
        "best": {"features": best.features, "model": best.model, "params": params},
        "crema_test": {k: round(v, 4) for k, v in s_te.items()},
        "crema_test_sent_bal_prior_corrected": round(balanced_accuracy_score(SENT_OF[y[te]], spc), 4),
        "ravdess_unseen_corpus_emo": round(scores(model, Xr, yr)[0]["emo_acc"], 4),
        "ravdess_17_24_single_corpus": {k: round(v, 4) for k, v in single_rv.items()},
        "ravdess_17_24_multi_corpus": {k: round(v, 4) for k, v in m_rv.items()},
        "crema_test_multi_corpus": {k: round(v, 4) for k, v in m_cr.items()},
        "personalization_mean": pers.groupby("k")[["emo", "sent"]].mean().round(4).to_dict(),
        "emotion2vec_zero_shot": {k: round(v, 4) for k, v in zs.items()},
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))

    # plots
    piv = table.pivot(index="model", columns="features", values="test_emo")[list(feats)]
    ax = piv.plot.bar(figsize=(9.5, 4.6), rot=0, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("6-emotion accuracy (CREMA-D test speakers)")
    ax.set_xlabel("")
    ax.set_title("Feature sets x classifiers")
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_model_comparison.png"), dpi=130)
    plt.close()
    pm = pers.groupby("k")[["emo", "sent"]].mean()
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(pm.index * 6, pm.emo, "o-", label="6 emotions")
    plt.plot(pm.index * 6, pm.sent, "s-", label="sentiment")
    plt.xticks(pm.index * 6)
    plt.ylim(0.4, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xlabel("own labelled clips of the new speaker")
    plt.ylabel("accuracy on the speaker's other clips")
    plt.title(f"Few-shot personalisation ({best.features})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "personalization.png"), dpi=130)
    plt.close()

    # deploy: best setup trained on everything
    Xall, yall = np.vstack([X, Xr]), np.concatenate([y, yr])
    final = clone(model).fit(Xall, yall)
    extractor = {"WavLM layer 7": "wavlm", "emotion2vec+": "e2v",
                 "WavLM + emotion2vec+": "fusion"}[best.features]
    joblib.dump({"model": final, "model_name": best.model, "representation": best.features,
                 "extractor": extractor, "layers": [7], "emotions": NAMES, "sentiments": SENT_NAMES,
                 "sentiment_of": SENT_OF.tolist(), "lowpass_hz": 4000},
                os.path.join(MODEL_DIR, "final_emotion.joblib"))
    np.savez_compressed(os.path.join(MODEL_DIR, "final_train_embeddings.npz"),
                        X=Xall.astype(np.float16), y=yall)
    print(f"\nDeployed: {best.features} + {best.model} -> models/final_emotion.joblib")


if __name__ == "__main__":
    main()
