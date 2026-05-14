"""
ui.py
=====
Reusable Streamlit UI helpers shared by every page.

Centralising these here keeps page modules thin and prevents the same colours,
chip markup, and bootstrap logic from drifting across the three pages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple

import streamlit as st

from constants import SEVERITY_COLOR
from escalation import detect_escalation
from inference import load_or_train
from schemas import EscalationVerdict, JournalEntry

# ---------------------------------------------------------------------------
# Module import bootstrap
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent


def ensure_app_on_path() -> None:
    """Make ``streamlit_web-app/`` importable when Streamlit launches a page directly.

    Streamlit runs each page as a top-level script, so the ``streamlit_web-app``
    package is not on ``sys.path`` by default. Pages call this at import time.
    """
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

JOURNAL_KEY = "journal"
PIPE_KEY = "pipe"
META_KEY = "meta"
INPUT_KEY = "symptom_input"


def bootstrap_session() -> None:
    """Initialise session keys and lazily load the model on first run."""
    st.session_state.setdefault(JOURNAL_KEY, [])
    st.session_state.setdefault(INPUT_KEY, "")

    if PIPE_KEY not in st.session_state:
        with st.spinner(
            "Loading / training NLP model on Symptom2Disease (1,200 rows, 24 classes)…"
        ):
            pipe, meta = load_or_train()
        st.session_state[PIPE_KEY] = pipe
        st.session_state[META_KEY] = meta


# ---------------------------------------------------------------------------
# Cosmetic helpers
# ---------------------------------------------------------------------------

# Streamlit method name + emoji for each escalation level
_ESCALATION_STYLE: dict[str, Tuple[str, str]] = {
    "Stable":   ("success", "✅"),
    "Watch":    ("warning", "⚠️"),
    "Escalate": ("error",   "🚨"),
}


def render_escalation_banner(verdict: EscalationVerdict, *, show_window: bool = True) -> None:
    """Render the Stable / Watch / Escalate banner consistently."""
    method, icon = _ESCALATION_STYLE.get(verdict.level, ("info", "ℹ️"))
    getattr(st, method)(f"{icon}  **{verdict.level}** — {verdict.reason}")
    if show_window and verdict.last_n:
        st.caption(
            f"Window analysed: last {verdict.last_n} entry/entries · "
            f"severity delta: {verdict.delta_severity:+d} ranks"
        )


def severity_chip(level: str, *, large: bool = False) -> str:
    """Return a small inline HTML pill for a severity level."""
    color = SEVERITY_COLOR.get(level, "#888")
    size = "1.0rem" if large else "0.82rem"
    padding = "6px 16px" if large else "3px 12px"
    return (
        f'<span style="background:{color};color:#fff;padding:{padding};'
        f'border-radius:10px;font-size:{size};font-weight:600;">{level}</span>'
    )


def severity_counter(level: str, count: int) -> str:
    """A vertical chip with a count on top and the severity label below."""
    color = SEVERITY_COLOR.get(level, "#888")
    return (
        f'<div style="background:{color};color:#fff;padding:8px 16px;'
        f'border-radius:10px;text-align:center;">'
        f'<div style="font-size:1.4rem;font-weight:700;">{count}</div>'
        f'<div style="font-size:0.8rem;">{level}</div></div>'
    )


def info_card(label: str, value: str, *, accent: str = "#888",
              sublabel: str = "", background: str = "#f6f8fa") -> str:
    """A bordered card used by the Live Triage page result strip."""
    sub_html = (
        f'<div style="font-size:0.9rem;color:#444;">{sublabel}</div>' if sublabel else ""
    )
    return (
        f'<div style="padding:18px;border-radius:12px;background:{background};'
        f'border-left:6px solid {accent};">'
        f'<div style="font-size:0.85rem;color:#666;">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;">{value}</div>'
        f"{sub_html}"
        "</div>"
    )


def solid_card(label: str, value: str, *, color: str, sublabel: str = "") -> str:
    """A coloured-fill card (used for the severity badge)."""
    sub_html = (
        f'<div style="font-size:0.75rem;opacity:0.85;">{sublabel}</div>'
        if sublabel else ""
    )
    return (
        f'<div style="padding:18px;border-radius:12px;background:{color};'
        f'color:#fff;text-align:center;">'
        f'<div style="font-size:0.85rem;opacity:0.9;">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;">{value}</div>'
        f"{sub_html}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Empty-state guards
# ---------------------------------------------------------------------------

def require_journal(
    *,
    empty_message: str = (
        "No entries logged yet. Head to the **Live Triage** page, describe your "
        "symptoms, and tick *Log this entry to my journal* to start tracking."
    ),
) -> Iterable[JournalEntry]:
    """Return the journal or stop the page with an info message if it's empty."""
    bootstrap_session()
    entries = st.session_state.get(JOURNAL_KEY) or []
    if not entries:
        st.info(empty_message)
        st.stop()
    return entries


def current_escalation() -> EscalationVerdict:
    """Convenience: detect escalation over the live session journal."""
    return detect_escalation(st.session_state.get(JOURNAL_KEY) or [])
