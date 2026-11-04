"""
Q1 - Classification with Nearest Neighbours: real vs. fake news headlines.

Usage (from the ml_cep folder):
    python q1/q1_knn.py

Outputs (written to q1/results/):
    knn_euclidean.png   train/val accuracy vs k (default Minkowski p=2 = Euclidean)
    knn_cosine.png      train/val accuracy vs k (metric='cosine')
    knn_comparison.png  both validation curves on one plot
    results.txt         per-k accuracies, best k and test accuracy for both metrics
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "results")
SEED = 42
K_RANGE = range(1, 21)


def load_data(data_dir=DATA_DIR, seed=SEED):
    """(a) Load headlines, vectorize with CountVectorizer, split 70/15/15.

    Labels: 1 = real, 0 = fake.
    The split is stratified so each subset keeps the real/fake ratio.
    The CountVectorizer vocabulary is fitted on the training split only, so no
    information from validation/test headlines leaks into the features
    (words never seen in training are simply ignored at validation/test time).

    Returns X_train, X_val, X_test, y_train, y_val, y_test, vectorizer
    """
    with open(os.path.join(data_dir, "clean_real.txt"), encoding="utf-8") as f:
        real = [line.strip() for line in f if line.strip()]
    with open(os.path.join(data_dir, "clean_fake.txt"), encoding="utf-8") as f:
        fake = [line.strip() for line in f if line.strip()]

    headlines = np.array(real + fake)
    labels = np.array([1] * len(real) + [0] * len(fake))

    # 70% train, 30% temp -> temp split in half = 15% val, 15% test
    h_train, h_tmp, y_train, y_tmp = train_test_split(
        headlines, labels, test_size=0.30, random_state=seed, stratify=labels)
    h_val, h_test, y_val, y_test = train_test_split(
        h_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(h_train)
    X_val = vectorizer.transform(h_val)
    X_test = vectorizer.transform(h_test)
    # kept so the demo app can show the nearest training headlines
    vectorizer.train_headlines_ = list(h_train)
    vectorizer.train_labels_ = list(y_train)

    print(f"Loaded {len(real)} real + {len(fake)} fake = {len(headlines)} headlines")
    print(f"Split: train={X_train.shape[0]}, val={X_val.shape[0]}, "
          f"test={X_test.shape[0]}, vocabulary={len(vectorizer.vocabulary_)} words")
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


def select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test,
                     metric=None, plot_path=None):
    """(b)/(c) Train KNN for k=1..20, plot train/val accuracy, pick best k on
    validation, report its test accuracy.

    metric=None leaves KNeighborsClassifier at its default (Minkowski, p=2,
    i.e. Euclidean). Pass metric='cosine' for part (c).
    Ties in validation accuracy are broken by choosing the larger k
    (smoother, less over-fitted decision boundary).
    """
    kwargs = {} if metric is None else {"metric": metric}
    name = metric or "euclidean (default)"
    train_acc, val_acc = [], []

    for k in K_RANGE:
        clf = KNeighborsClassifier(n_neighbors=k, **kwargs)
        clf.fit(X_train, y_train)
        train_acc.append(clf.score(X_train, y_train))
        val_acc.append(clf.score(X_val, y_val))

    val_arr = np.array(val_acc)
    best_idx = np.flatnonzero(val_arr == val_arr.max())[-1]
    best_k = list(K_RANGE)[best_idx]

    best = KNeighborsClassifier(n_neighbors=best_k, **kwargs).fit(X_train, y_train)
    test_acc = best.score(X_test, y_test)

    print(f"\n=== KNN, metric = {name} ===")
    print(f"{'k':>3} {'train acc':>10} {'val acc':>9} {'train err':>10} {'val err':>8}")
    for k, tr, va in zip(K_RANGE, train_acc, val_acc):
        print(f"{k:>3} {tr:>10.4f} {va:>9.4f} {1 - tr:>10.4f} {1 - va:>8.4f}")
    print(f"Best k = {best_k} | val acc = {val_arr[best_idx]:.4f} | TEST acc = {test_acc:.4f}")

    if plot_path:
        plt.figure(figsize=(8, 5))
        plt.plot(list(K_RANGE), train_acc, "o-", label="Training accuracy")
        plt.plot(list(K_RANGE), val_acc, "s-", label="Validation accuracy")
        plt.axvline(best_k, color="gray", ls="--", lw=1,
                    label=f"Best k = {best_k} (test acc = {test_acc:.3f})")
        plt.xticks(list(K_RANGE))
        plt.xlabel("k (number of neighbours)")
        plt.ylabel("Accuracy")
        plt.title(f"KNN real vs. fake headlines - metric: {name}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return {"metric": name, "train_acc": train_acc, "val_acc": val_acc,
            "best_k": best_k, "best_val_acc": float(val_arr[best_idx]),
            "test_acc": test_acc, "model": best}


def cosine_vs_euclidean_demo():
    """(c) The hint dataset ['cat', 'bulldozer', 'cat cat cat']."""
    docs = ["cat", "bulldozer", "cat cat cat"]
    X = CountVectorizer().fit_transform(docs)
    lines = ["Hint dataset: " + str(docs),
             "Count vectors (bulldozer, cat): " + str(X.toarray().tolist()),
             "Euclidean distances:\n" + np.array2string(euclidean_distances(X), precision=3),
             "Cosine distances:\n" + np.array2string(cosine_distances(X), precision=3)]
    text = "\n".join(lines)
    print("\n" + text)
    return text


EXPLANATION = """
How metric='cosine' works:
  cosine_distance(u, v) = 1 - (u . v) / (||u|| * ||v||)
  It only depends on the ANGLE between the two word-count vectors, not on their
  length. Two headlines that use the same words in the same proportions have
  distance 0 no matter how long they are; headlines with no words in common have
  distance 1.

Why it beats Euclidean here:
  With Euclidean distance, 'cat' is closer to 'bulldozer' (sqrt(2) = 1.414) than
  to 'cat cat cat' (2.0), even though 'cat' and 'cat cat cat' are about the same
  thing. Euclidean distance is dominated by vector LENGTH (headline length /
  repeated words), so short headlines all look similar to each other regardless
  of topic, and long ones look far from everything. Cosine distance gives
  'cat' vs 'cat cat cat' = 0 and 'cat' vs 'bulldozer' = 1, i.e. it measures
  which words are used (topic / style), which is what separates real from fake
  headlines. Bag-of-words vectors are also very sparse and high dimensional,
  where Euclidean distances concentrate and become less informative, while the
  normalised overlap measured by cosine stays meaningful.
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()

    euc = select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test,
                           metric=None, plot_path=os.path.join(OUT_DIR, "knn_euclidean.png"))
    cos = select_knn_model(X_train, X_val, X_test, y_train, y_val, y_test,
                           metric="cosine", plot_path=os.path.join(OUT_DIR, "knn_cosine.png"))

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_RANGE), euc["val_acc"], "o-", label="Validation - Euclidean")
    plt.plot(list(K_RANGE), cos["val_acc"], "s-", label="Validation - Cosine")
    plt.xticks(list(K_RANGE))
    plt.xlabel("k")
    plt.ylabel("Validation accuracy")
    plt.title("Euclidean vs. cosine KNN")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "knn_comparison.png"), dpi=150)
    plt.close()

    # save the best (cosine) model + vectorizer for predict / demo app
    import joblib
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": cos["model"], "best_k": cos["best_k"],
                 "train_headlines": vectorizer.train_headlines_,
                 "train_labels": vectorizer.train_labels_,
                 "labels": {0: "fake", 1: "real"}},
                os.path.join(HERE, "models", "knn_cosine.joblib"))

    demo = cosine_vs_euclidean_demo()
    print(EXPLANATION)

    with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
        for r in (euc, cos):
            f.write(f"=== metric: {r['metric']} ===\n")
            f.write("k,train_acc,val_acc\n")
            for k, tr, va in zip(K_RANGE, r["train_acc"], r["val_acc"]):
                f.write(f"{k},{tr:.4f},{va:.4f}\n")
            f.write(f"best_k={r['best_k']} val_acc={r['best_val_acc']:.4f} "
                    f"test_acc={r['test_acc']:.4f}\n\n")
        f.write(demo + "\n" + EXPLANATION)
    print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
