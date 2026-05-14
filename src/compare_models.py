"""
compare_models.py
=================

Reads ``artifacts/model_comparison.json`` and renders:

* a side-by-side metrics bar chart (``figures/model_comparison.png``)
* per-class F1 heatmap for the two models (``figures/per_class_f1.png``)

Run from project root::

    python -m src.compare_models
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report as _cr

try:
    from .data import load_dataset, split   # package mode: python -m src.compare_models
except ImportError:
    from data import load_dataset, split    # script mode: python src/compare_models.py

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def main() -> None:
    cmp = json.loads((ARTIFACTS / "model_comparison.json").read_text())
    models = cmp["models"]

    names = [m["name"] for m in models.values()]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    values = np.array([[m.get(k, 0.0) for k in metrics] for m in models.values()])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(names), 1)

    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, (name, row) in enumerate(zip(names, values)):
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, row, width, label=name, color=palette[i % len(palette)])
        for b, v in zip(bars, row):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — Symptom2Disease (24 classes)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    out = FIGURES / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # Per-class F1 — recomputed from the saved pipelines so this script works
    # regardless of which component last wrote the *_meta.json files
    # (streamlit_web-app/inference.py writes a minimal meta without "report").
    df = load_dataset()
    _, X_test, _, y_test = split(df)

    svm_pipe = joblib.load(ARTIFACTS / "calibrated_svm_pipeline.joblib")
    lr_pipe  = joblib.load(ARTIFACTS / "logreg_pipeline.joblib")

    svm_report = _cr(y_test, svm_pipe.predict(X_test), output_dict=True, zero_division=0)
    lr_report  = _cr(y_test, lr_pipe.predict(X_test),  output_dict=True, zero_division=0)

    classes = sorted([k for k in svm_report.keys()
                      if k not in ("accuracy", "macro avg", "weighted avg")])
    svm_f1 = [svm_report[c]["f1-score"] for c in classes]
    lr_f1 = [lr_report[c]["f1-score"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, max(6, 0.28 * len(classes))))
    yp = np.arange(len(classes))
    ax.barh(yp - 0.2, svm_f1, height=0.4, label="Calibrated SVM", color="#4C78A8")
    ax.barh(yp + 0.2, lr_f1, height=0.4, label="Word+Char LogReg", color="#F58518")
    ax.set_yticks(yp)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-class F1 — SVM vs Logistic Regression")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    out = FIGURES / "per_class_f1.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
