"""
constants.py
============
Single source of truth for clinical knowledge tables used by the triage engine:

* ``SEVERITY_MAP``   — deterministic disease → severity mapping
* ``SEVERITY_RANK``  — ordinal ranking used by the escalation detector
* ``SEVERITY_COLOR`` — palette shared by Streamlit and the IEEE figures
* ``RED_FLAGS``      — clinical phrases that force severity to "Urgent"

Keeping these here (instead of inside the inference module) means the values
can be imported by tests, notebooks, or downstream tools without dragging in
scikit-learn as a transitive dependency.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Severity taxonomy
# ---------------------------------------------------------------------------

SEVERITY_LEVELS: Tuple[str, str, str] = ("Mild", "Moderate", "Urgent")

SEVERITY_RANK: Dict[str, int] = {level: i for i, level in enumerate(SEVERITY_LEVELS)}

SEVERITY_COLOR: Dict[str, str] = {
    "Mild": "#7fbf7b",
    "Moderate": "#fdae61",
    "Urgent": "#d73027",
}

SEVERITY_DESCRIPTION: Dict[str, str] = {
    "Mild": "Self-care, monitor at home.",
    "Moderate": "See a GP within a few days.",
    "Urgent": "Seek care immediately.",
}

# Disease labels are lowercase to match the normalised dataset labels.
SEVERITY_MAP: Dict[str, str] = {
    # --- Mild ---------------------------------------------------------
    "common cold": "Mild",
    "allergy": "Mild",
    "acne": "Mild",
    "fungal infection": "Mild",
    "psoriasis": "Mild",
    "migraine": "Mild",
    # --- Moderate -----------------------------------------------------
    "bronchial asthma": "Moderate",
    "diabetes": "Moderate",
    "hypertension": "Moderate",
    "cervical spondylosis": "Moderate",
    "arthritis": "Moderate",
    "gastroesophageal reflux disease": "Moderate",
    "peptic ulcer disease": "Moderate",
    "urinary tract infection": "Moderate",
    "varicose veins": "Moderate",
    "chicken pox": "Moderate",
    "drug reaction": "Moderate",
    "dimorphic hemorrhoids": "Moderate",
    # --- Urgent -------------------------------------------------------
    "pneumonia": "Urgent",
    "dengue": "Urgent",
    "malaria": "Urgent",
    "typhoid": "Urgent",
    "jaundice": "Urgent",
    "impetigo": "Urgent",
}

DEFAULT_SEVERITY: str = "Moderate"  # fallback when a disease is missing from SEVERITY_MAP

# ---------------------------------------------------------------------------
# Red-flag phrases (curated from WHO emergency triage + UK NHS 111 guidance)
# ---------------------------------------------------------------------------

RED_FLAGS: List[str] = [
    "chest pain",
    "shortness of breath",
    "cannot breathe",
    "can't breathe",
    "cant breathe",
    "breathless",
    "unconscious",
    "unresponsive",
    "fainting",
    "fainted",
    "seizure",
    "stroke",
    "blood in stool",
    "blood in urine",
    "vomiting blood",
    "coughing blood",
    "severe bleeding",
    "severe pain",
    "high fever",
    "stiff neck",
    "confusion",
    "slurred speech",
    "numbness on one side",
    "suicidal",
]

__all__ = [
    "SEVERITY_LEVELS",
    "SEVERITY_RANK",
    "SEVERITY_COLOR",
    "SEVERITY_DESCRIPTION",
    "SEVERITY_MAP",
    "DEFAULT_SEVERITY",
    "RED_FLAGS",
]
