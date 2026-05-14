"""
schemas.py
==========
Plain dataclasses used as the engine's public API surface.

These are deliberately framework-free (no Streamlit, no sklearn imports) so
they can be serialised, unit-tested, or consumed by other front-ends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = [
    "Prediction",
    "JournalEntry",
    "EscalationVerdict",
]


@dataclass(frozen=True)
class Prediction:
    """Single inference result with full triage context."""

    disease: str
    confidence: float                       # top-1 probability
    severity: str                           # Mild / Moderate / Urgent
    severity_source: str                    # "model_map" or "red_flag_override"
    differential: List[Tuple[str, float]]   # top-k (disease, prob), sorted desc
    drivers: List[Tuple[str, float]]        # top-k (token, weight), sorted desc
    red_flags_hit: List[str]                # red-flag phrases found in the text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease": self.disease,
            "confidence": float(self.confidence),
            "severity": self.severity,
            "severity_source": self.severity_source,
            "differential": [(d, float(p)) for d, p in self.differential],
            "drivers": [(t, float(w)) for t, w in self.drivers],
            "red_flags_hit": list(self.red_flags_hit),
        }


@dataclass
class JournalEntry:
    """One logged user turn — the raw text plus the model's interpretation."""

    timestamp: str    # ISO-8601 (UTC)
    user_text: str
    prediction: Prediction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_text": self.user_text,
            "prediction": self.prediction.to_dict(),
        }


@dataclass(frozen=True)
class EscalationVerdict:
    """Output of :func:`detect_escalation` — what to do with the recent window."""

    level: str            # "Stable" / "Watch" / "Escalate"
    reason: str           # human-readable explanation
    last_n: int           # number of entries inspected
    delta_severity: int   # change in severity rank across the window

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
