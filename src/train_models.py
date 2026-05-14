"""
train_models.py
===============

Trains the **two AI techniques** required by the coursework brief on the
Symptom2Disease dataset and saves all artifacts under ``artifacts/``.

Technique 1 — TF-IDF (word 1-2 grams) + calibrated LinearSVC
    Strong linear baseline. Calibrated so we get genuine predict_proba
    for top-k differential diagnosis.

Technique 2 — TF-IDF (word + character 3-5 grams) + Logistic Regression
    Different feature space (character n-grams catch morphological cues like
    *itchy*, *itching*, *itch*) and a different optimiser, so the comparison
    is meaningful — not just two flavours of the same model.

Run from the project root::

    python -m src.train_models

Outputs (under ``artifacts/``)::

    calibrated_svm_pipeline.joblib
    calibrated_svm_meta.json
    logreg_pipeline.joblib
    logreg_meta.json
    model_comparison.json     # combined report consumed by the Streamlit
                              # app and the IEEE paper
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from .data import RANDOM_SEED, load_dataset, split   # package mode: python -m src.train_models
except ImportError:
    from data import RANDOM_SEED, load_dataset, split    # script mode: python src/train_models.py

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def build_calibrated_svm() -> Pipeline:
    """Technique 1: word 1-2 grams + Platt-scaled LinearSVC."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_SEED), cv=3, method="sigmoid")),
        ]
    )


def build_logreg() -> Pipeline:
    """Technique 2: word + character n-gram union + Logistic Regression."""
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
    )
    union = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    return Pipeline(
        [
            ("features", union),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    C=4.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "labels": sorted(np.unique(np.concatenate([y_test.values, y_pred]))),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=sorted(np.unique(np.concatenate([y_test.values, y_pred]))),
        ).tolist(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> dict:
    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)

    print(f"Loaded {len(df)} rows · {df['label_norm'].nunique()} classes")
    print(f"Train: {len(X_train)} · Test: {len(X_test)}")

    results = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_classes": int(df["label_norm"].nunique()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": {},
    }

    # ---- Technique 1: Calibrated TF-IDF + LinearSVC ----------------------
    print("\nTraining Technique 1 — Calibrated TF-IDF + LinearSVC …")
    svm = build_calibrated_svm()
    svm.fit(X_train, y_train)
    svm_eval = _evaluate(svm, X_test, y_test)
    joblib.dump(svm, ARTIFACT_DIR / "calibrated_svm_pipeline.joblib")
    (ARTIFACT_DIR / "calibrated_svm_meta.json").write_text(json.dumps(svm_eval, indent=2))
    results["models"]["svm_tfidf"] = {
        "name": "TF-IDF (word 1-2g) + Calibrated LinearSVC",
        **{k: v for k, v in svm_eval.items() if k not in ("report", "confusion_matrix", "labels")},
    }
    print(f"  acc={svm_eval['accuracy']:.4f}  f1={svm_eval['f1_macro']:.4f}")

    # ---- Technique 2: Word+Char n-gram union + Logistic Regression -------
    print("\nTraining Technique 2 — Word+Char TF-IDF + Logistic Regression …")
    lr = build_logreg()
    lr.fit(X_train, y_train)
    lr_eval = _evaluate(lr, X_test, y_test)
    joblib.dump(lr, ARTIFACT_DIR / "logreg_pipeline.joblib")
    (ARTIFACT_DIR / "logreg_meta.json").write_text(json.dumps(lr_eval, indent=2))
    results["models"]["logreg_charword"] = {
        "name": "TF-IDF (word + char 3-5g) + Logistic Regression",
        **{k: v for k, v in lr_eval.items() if k not in ("report", "confusion_matrix", "labels")},
    }
    print(f"  acc={lr_eval['accuracy']:.4f}  f1={lr_eval['f1_macro']:.4f}")

    # ---- Optional Technique 3: BERT (read external metrics if present) ---
    bert_meta_path = ARTIFACT_DIR / "bert_meta.json"
    if bert_meta_path.exists():
        try:
            bert_meta = json.loads(bert_meta_path.read_text())
            results["models"]["bert"] = {
                "name": "Fine-tuned BERT (bert-base-uncased)",
                "accuracy": float(bert_meta.get("accuracy", 0.0)),
                "precision_macro": float(bert_meta.get("precision_macro", 0.0)),
                "recall_macro": float(bert_meta.get("recall_macro", 0.0)),
                "f1_macro": float(bert_meta.get("f1_macro", 0.0)),
            }
            print(f"\nLoaded external BERT metrics: f1={bert_meta.get('f1_macro')}")
        except Exception as e:
            print(f"\n[BERT metrics found but could not be parsed: {e}]")
    else:
        print("\n[BERT metrics not found — run notebook/bert_finetune_colab.ipynb on Colab to add them]")

    # ---- Save combined comparison report ---------------------------------
    (ARTIFACT_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote artifacts to {ARTIFACT_DIR}/")
    return results


if __name__ == "__main__":
    main()
