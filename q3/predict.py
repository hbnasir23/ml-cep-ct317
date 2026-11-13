"""
Classify paper images (paper / hole / penMark) and draw a bounding box
around the defect.

Usage (from the ml_cep folder, after running q3_defect_detection.py once):
    python q3/predict.py image.jpg [more.jpg ...] [--out folder]
Annotated images are written to q3/predictions/ by default.
"""
import argparse
import os
import sys

import cv2
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import defect_features as dfx                      # noqa: E402
from q3_defect_detection import localize, draw_box, box_in_original  # noqa: E402


def load_models():
    img = os.path.join(HERE, "models", "image_classifier.joblib")
    reg = os.path.join(HERE, "models", "region_classifier.joblib")
    if not (os.path.exists(img) and os.path.exists(reg)):
        sys.exit("Models not found - run  python q3/q3_defect_detection.py  first.")
    return joblib.load(img), joblib.load(reg)


def predict_image(path, img_bundle, reg_bundle):
    """Returns (class name, class probabilities dict, box or None, annotated BGR image)."""
    classes = img_bundle["classes"]
    reg = reg_bundle["model"]
    cand, p_best, scale, _, _ = localize(path, reg, k=reg_bundle.get("k", 5))
    if img_bundle.get("decision") == "multi-proposal":
        # class = class of the most defect-like proposal
        proba, model_classes = p_best, reg.classes_
    else:
        model = img_bundle["model"]
        x = dfx.image_features(path)[0][img_bundle["cols"]][None, :]
        proba, model_classes = model.predict_proba(x)[0], model.classes_
    cls = classes[int(model_classes[int(np.argmax(proba))])]
    probs = {classes[int(c)]: float(p) for c, p in zip(model_classes, proba)}
    if cls == "paper":
        img = cv2.imread(path)
        cv2.putText(img, f"no defect {probs['paper']:.2f}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 160, 0), 6, cv2.LINE_AA)
        return cls, probs, None, img
    img = draw_box(path, cand, scale, f"{cls} {probs[cls]:.2f}")
    return cls, probs, box_in_original(cand, scale), img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "predictions"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img_b, reg_b = load_models()
    print(f"Image classifier: {img_b['model_name']} ({img_b['features']} features, decision={img_b.get('decision')}); "
          f"region classifier: {reg_b['model_name']}\n")
    for p in a.images:
        cls, probs, box, img = predict_image(p, img_b, reg_b)
        out = os.path.join(a.out, os.path.basename(os.path.dirname(p)) + "_" + os.path.basename(p))
        cv2.imwrite(out, img)
        pstr = ", ".join(f"{k} {v:.2f}" for k, v in probs.items())
        print(f"{p}\n   -> {cls:8s} ({pstr})  box={box}  saved {out}")


if __name__ == "__main__":
    main()
