"""
Q3 - visualise the misclassified test images (zoomed on the chosen proposal).

    python q3/error_analysis.py      (after q3_defect_detection.py)
Writes q3/results/errors.png
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
loc = pd.read_csv(os.path.join(HERE, "results", "localization_test.csv"))
err = loc[loc.true != loc.pred]
n = max(1, len(err))
fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
axes = [axes] if n == 1 else axes
for ax, (_, r) in zip(axes, err.iterrows()):
    img = cv2.cvtColor(cv2.imread(os.path.join(HERE, "data", "out", r.file)), cv2.COLOR_BGR2RGB)
    cx, cy = (r.x0 + r.x1) // 2, (r.y0 + r.y1) // 2
    h = 150
    cv2.rectangle(img, (r.x0, r.y0), (r.x1, r.y1), (255, 0, 0), 3)
    ax.imshow(img[max(0, cy - h):cy + h, max(0, cx - h):cx + h])
    ax.set_title(f"{r.file}\ntrue={r.true} pred={r.pred}\nP(defect)={r.p_defect:.2f}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "errors.png"), dpi=110)
print(err.to_string(index=False))
