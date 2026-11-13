"""
Independent re-check of the deployed models on the held-out TEST images,
straight from the image files (no cached features).

    python q3/verify_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _one(path):
    from predict import load_models, predict_image
    img_b, reg_b = load_models()
    cls, probs, box, _ = predict_image(path, img_b, reg_b)
    return cls


def main():
    split = pd.read_csv(os.path.join(HERE, "results", "split.csv"))
    test = split[split.split == "test"]
    paths = [os.path.join(HERE, "data", "out", f) for f in test.file]
    pred = Parallel(n_jobs=6)(delayed(_one)(p) for p in paths)
    classes = ["paper", "hole", "penMark"]
    print(f"Test images: {len(test)}  accuracy: {accuracy_score(test.cls, pred):.4f}\n")
    print(classification_report(test.cls, pred, labels=classes, digits=4))
    print("confusion matrix (rows = true paper/hole/penMark):")
    print(confusion_matrix(test.cls, pred, labels=classes))


if __name__ == "__main__":
    main()
