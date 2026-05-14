"""
inference.py
============
Model lifecycle (train / cache / load) and the inference pipeline.

The model is a Platt-scaled :class:`LinearSVC` over TF-IDF features
(``CalibratedClassifierCV(LinearSVC, method="sigmoid", cv=3)``). Calibration is
the reason this module exists — it converts SVM decision scores into genuine
probabilities so the top-3 differential is meaningful.

The cache key combines the dataset signature and the pipeline configuration
hash, so the model is automatically retrained whenever either changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from constants import DEFAULT_SEVERITY, RED_FLAGS, SEVERITY_MAP
from schemas import Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & training config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # coursework/project/
DATA_PATH = PROJECT_ROOT / "dataset" / "Symptom2Disease.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "calibrated_svm_pipeline.joblib"
META_PATH = ARTIFACT_DIR / "calibrated_svm_meta.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters for the calibrated TF-IDF + LinearSVC pipeline."""

    test_size: float = 0.30
    random_state: int = 42
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    sublinear_tf: bool = True
    svm_c: float = 1.0
    calibration_cv: int = 3
    calibration_method: str = "sigmoid"

    def signature(self) -> str:
        """Stable hash so the cache invalidates when hyper-parameters change."""
        payload = json.dumps(self.__dict__, sort_keys=True, default=str).encode()
        return hashlib.md5(payload).hexdigest()[:12]


DEFAULT_CONFIG = TrainingConfig()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse whitespace."""
    text = str(text).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def detect_red_flags(text: str) -> List[str]:
    """Return the subset of :data:`RED_FLAGS` that appear in *text*."""
    text_l = text.lower()
    return [flag for flag in RED_FLAGS if flag in text_l]


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _dataset_signature() -> str:
    """Cheap hash over the CSV's size + mtime — enough to detect changes."""
    stat = DATA_PATH.stat()
    h = hashlib.md5()
    h.update(str(stat.st_mtime).encode())
    h.update(str(stat.st_size).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(config: TrainingConfig = DEFAULT_CONFIG) -> Pipeline:
    """Construct a fresh, untrained pipeline from *config*."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.ngram_range,
                    min_df=config.min_df,
                    sublinear_tf=config.sublinear_tf,
                ),
            ),
            (
                "svm",
                CalibratedClassifierCV(
                    LinearSVC(C=config.svm_c, random_state=config.random_state),
                    cv=config.calibration_cv,
                    method=config.calibration_method,
                ),
            ),
        ]
    )


def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df[["label", "text"]].dropna().reset_index(drop=True)
    df["label_norm"] = df["label"].str.lower().str.strip()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# Train / cache / load
# ---------------------------------------------------------------------------

def train_and_save(config: TrainingConfig = DEFAULT_CONFIG) -> Dict:
    """Train the pipeline on the full dataset and persist artefacts."""
    logger.info("Training calibrated TF-IDF + LinearSVC pipeline…")
    df = _load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label_norm"],
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["label_norm"],
    )

    pipe = build_pipeline(config)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_classes": int(df["label_norm"].nunique()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset_sig": _dataset_signature(),
        "config_sig": config.signature(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved model to %s (acc=%.4f f1=%.4f)",
                MODEL_PATH, metrics["accuracy"], metrics["f1_macro"])
    return metrics


def _cache_is_fresh(meta: Dict, config: TrainingConfig) -> bool:
    return (
        meta.get("dataset_sig") == _dataset_signature()
        and meta.get("config_sig", config.signature()) == config.signature()
    )


def load_or_train(config: TrainingConfig = DEFAULT_CONFIG) -> Tuple[Pipeline, Dict]:
    """Return ``(pipeline, metrics)``, training only if the cache is stale."""
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            if _cache_is_fresh(meta, config):
                logger.debug("Using cached pipeline at %s", MODEL_PATH)
                return joblib.load(MODEL_PATH), meta
            logger.info("Cache stale — retraining (dataset or config changed)")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read cached metadata (%s) — retraining", exc)

    metrics = train_and_save(config)
    return joblib.load(MODEL_PATH), metrics


# ---------------------------------------------------------------------------
# Explainability — top driver tokens
# ---------------------------------------------------------------------------

def _top_drivers(pipe: Pipeline, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Top-*k* TF-IDF features in *text* that push the model toward its prediction.

    Computes ``tfidf(text) * mean(coef_[pred_class])`` over the LinearSVC
    estimators that ``CalibratedClassifierCV`` wraps internally, and returns the
    highest-scoring nonzero features that actually appear in the input.
    """
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    cal: CalibratedClassifierCV = pipe.named_steps["svm"]

    feature_names = vec.get_feature_names_out()
    x_vec = vec.transform([text])

    pred_class = cal.predict(x_vec)[0]
    class_idx = list(cal.classes_).index(pred_class)

    coefs: List[np.ndarray] = []
    for cc in cal.calibrated_classifiers_:
        # sklearn >=1.4 exposes `.estimator`; older versions used `.base_estimator`
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is None or not hasattr(est, "coef_"):
            continue
        coefs.append(est.coef_[class_idx])
    if not coefs:
        return []
    avg_coef = np.mean(np.vstack(coefs), axis=0)

    x_arr = x_vec.toarray()[0]
    scores = x_arr * avg_coef
    nonzero = np.where(x_arr > 0)[0]
    ranked = sorted(nonzero, key=lambda i: -scores[i])[:top_k]
    return [(feature_names[i], float(scores[i])) for i in ranked if scores[i] > 0]


# ---------------------------------------------------------------------------
# Inference entry-point
# ---------------------------------------------------------------------------

def predict(
    text: str,
    pipe: Optional[Pipeline] = None,
    *,
    top_k_differential: int = 3,
    top_k_drivers: int = 5,
) -> Prediction:
    """
    Run end-to-end triage on *text*.

    Returns a fully-populated :class:`Prediction` containing the top-1 disease,
    top-*k* differential, driver tokens, severity, and detected red flags.

    The red-flag detector overrides severity to "Urgent" regardless of the
    model output, mirroring real triage protocols.
    """
    if not text or not text.strip():
        raise ValueError("predict() requires a non-empty text input")

    if pipe is None:
        pipe, _ = load_or_train()

    cleaned = clean_text(text)
    proba = pipe.predict_proba([cleaned])[0]
    classes = list(pipe.classes_)
    order = np.argsort(-proba)
    differential = [(classes[i], float(proba[i])) for i in order[:top_k_differential]]
    top_disease, top_conf = differential[0]

    red_flags_hit = detect_red_flags(text)
    if red_flags_hit:
        severity, severity_source = "Urgent", "red_flag_override"
    else:
        severity = SEVERITY_MAP.get(top_disease, DEFAULT_SEVERITY)
        severity_source = "model_map"

    drivers = _top_drivers(pipe, cleaned, top_k=top_k_drivers)

    return Prediction(
        disease=top_disease,
        confidence=top_conf,
        severity=severity,
        severity_source=severity_source,
        differential=differential,
        drivers=drivers,
        red_flags_hit=red_flags_hit,
    )


__all__ = [
    "TrainingConfig",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "DATA_PATH",
    "ARTIFACT_DIR",
    "MODEL_PATH",
    "META_PATH",
    "build_pipeline",
    "clean_text",
    "detect_red_flags",
    "train_and_save",
    "load_or_train",
    "predict",
]
