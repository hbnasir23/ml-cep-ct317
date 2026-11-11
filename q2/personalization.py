"""
Few-shot personalisation: adapt the emotion model to a NEW speaker / microphone
with a handful of that person's own labelled recordings.

Why: a model trained on acted studio corpora loses accuracy on a different
voice, microphone and room (domain shift). Adding even 1-3 example clips per
emotion from the target user, with a higher sample weight, moves the decision
boundaries towards that user's voice. Logistic Regression retrains in seconds,
so this can run inside the app.

    python q2/personalization.py       # offline experiment -> results/wavlm/personalization.csv

The experiment simulates new users with the 8 RAVDESS actors that the base
model never saw (actors 17-24): for each actor, k clips per emotion are used
for adaptation and the actor's REMAINING clips are the test set.
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

USER_WEIGHT = 20.0     # each user clip counts like 20 corpus clips
SEED = 42


def personalise(base_pipeline, X_corpus, y_corpus, X_user, y_user, weight=USER_WEIGHT):
    """Refit a copy of the pipeline on corpus + weighted user examples."""
    X = np.vstack([X_corpus, X_user])
    y = np.concatenate([y_corpus, y_user])
    w = np.concatenate([np.ones(len(y_corpus)), np.full(len(y_user), weight)])
    return clone(base_pipeline).fit(X, y, clf__sample_weight=w)


def _run(base, Xc, yc, Xa, ya, k, rep, sent_of):
    rng = np.random.RandomState(SEED + rep)
    adapt = []
    for e in np.unique(ya):
        idx = np.flatnonzero(ya == e)
        adapt += list(rng.choice(idx, size=min(k, len(idx) - 1), replace=False)) if k else []
    test = np.setdiff1d(np.arange(len(ya)), adapt)
    m = personalise(base, Xc, yc, Xa[adapt], ya[adapt]) if k else base
    yp = m.predict(Xa[test])
    return accuracy_score(ya[test], yp), accuracy_score(sent_of[ya[test]], sent_of[yp])


def main():
    from q2_wavlm_emotion import load_crema, load_ravdess, pool, SENT_OF, OUT_DIR
    bundle = joblib.load(os.path.join(HERE, "models", "wavlm_emotion.joblib"))
    split, E, y = load_crema()
    rav, Er, yr = load_ravdess()
    L = bundle["layers"]
    X, Xr = pool(E, L), pool(Er, L)
    tr_r = rav.actor.values <= 16
    Xc = np.vstack([X, Xr[tr_r]])           # all CREMA-D + RAVDESS actors 1-16
    yc = np.concatenate([y, yr[tr_r]])
    base = clone(bundle["model"]).fit(Xc, yc)

    jobs = []
    for actor in range(17, 25):
        m = rav.actor.values == actor
        for k in (0, 1, 2, 3):
            for rep in range(3 if k else 1):
                jobs.append((actor, k, rep, Xr[m], yr[m]))
    out = Parallel(n_jobs=6)(delayed(_run)(base, Xc, yc, Xa, ya, k, rep, SENT_OF)
                             for actor, k, rep, Xa, ya in jobs)
    df = pd.DataFrame([{"actor": a, "k_per_emotion": k, "rep": r, "emo_acc": o[0], "sent_acc": o[1]}
                       for (a, k, r, _, _), o in zip(jobs, out)])
    summ = df.groupby("k_per_emotion")[["emo_acc", "sent_acc"]].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(OUT_DIR, "personalization.csv"))
    df.to_csv(os.path.join(OUT_DIR, "personalization_runs.csv"), index=False)
    print(summ)

    means = df.groupby("k_per_emotion")[["emo_acc", "sent_acc"]].mean()
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(means.index * 6, means.emo_acc, "o-", label="6 emotions")
    plt.plot(means.index * 6, means.sent_acc, "s-", label="sentiment")
    plt.xticks(means.index * 6)
    plt.xlabel("own labelled clips provided by the new speaker (k per emotion x 6)")
    plt.ylabel("accuracy on the speaker's other clips")
    plt.ylim(0.4, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Few-shot personalisation (8 unseen RAVDESS actors)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "personalization.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
