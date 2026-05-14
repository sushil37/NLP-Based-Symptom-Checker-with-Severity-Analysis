"""
triage_engine.py
================
Public facade for the Symptom Triage engine.

The implementation has been split into focused modules for clarity and
testability:

* :mod:`constants`   — clinical knowledge tables (severity map, red flags)
* :mod:`schemas`     — framework-free dataclasses (``Prediction`` …)
* :mod:`inference`   — model lifecycle + ``predict()``
* :mod:`escalation`  — longitudinal escalation detector + Markdown export

This module re-exports the canonical public API so existing code paths
(notebooks, examples, Streamlit pages) continue to work without changes::

    from triage_engine import predict, detect_escalation, JournalEntry

New code should prefer importing directly from the submodules above.
"""

from __future__ import annotations

from constants import (
    DEFAULT_SEVERITY,
    RED_FLAGS,
    SEVERITY_COLOR,
    SEVERITY_DESCRIPTION,
    SEVERITY_LEVELS,
    SEVERITY_MAP,
    SEVERITY_RANK,
)
from escalation import detect_escalation, export_session_markdown
from inference import (
    ARTIFACT_DIR,
    DATA_PATH,
    DEFAULT_CONFIG,
    META_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    TrainingConfig,
    build_pipeline,
    clean_text,
    detect_red_flags,
    load_or_train,
    predict,
    train_and_save,
)
from schemas import EscalationVerdict, JournalEntry, Prediction

__all__ = [
    # Constants
    "SEVERITY_LEVELS",
    "SEVERITY_MAP",
    "SEVERITY_RANK",
    "SEVERITY_COLOR",
    "SEVERITY_DESCRIPTION",
    "DEFAULT_SEVERITY",
    "RED_FLAGS",
    # Schemas
    "Prediction",
    "JournalEntry",
    "EscalationVerdict",
    # Inference
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
    "load_or_train",
    "train_and_save",
    "predict",
    # Escalation
    "detect_escalation",
    "export_session_markdown",
]
