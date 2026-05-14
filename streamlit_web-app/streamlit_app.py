"""
streamlit_app.py
================
Home — "Live Triage" page.

Renders the symptom input box, runs inference via :mod:`triage_engine`, and
displays the disease / severity / red-flag result strip plus the differential,
driver tokens, and escalation banner.

Run from the project root::

    streamlit run streamlit_web-app/streamlit_app.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import streamlit as st

import ui
from constants import SEVERITY_COLOR, SEVERITY_DESCRIPTION
from inference import predict
from schemas import JournalEntry

ui.ensure_app_on_path()

# ---------------------------------------------------------------------------
# Page config & session bootstrap
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Symptom Triage Journal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.bootstrap_session()

# ---------------------------------------------------------------------------
# Constants for this page
# ---------------------------------------------------------------------------

EXAMPLES: Dict[str, str] = {
    "💚 Mild example": (
        "I have a runny nose, mild sore throat, and sneezing since yesterday. No fever."
    ),
    "🟠 Moderate example": (
        "Burning sensation when urinating and frequent urge to go to the bathroom "
        "for the past two days, with mild lower back pain."
    ),
    "🔴 Urgent example": (
        "Severe chest pain radiating to my left arm, shortness of breath, "
        "and cold sweats for the past hour."
    ),
}

PAGE_INTRO = (
    "Type symptoms in plain English — the model returns a **top-3 differential**, "
    "highlights the words that drove the prediction, detects clinical red-flag phrases, "
    "and tracks how your symptoms evolve so it can warn when things are escalating."
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🩺 Triage Journal")
        st.caption("MSc AI coursework prototype — Sushil Shrestha")
        st.divider()

        meta = st.session_state[ui.META_KEY]
        st.markdown("### Model")
        st.metric("Test accuracy", f"{meta['accuracy'] * 100:.2f}%")
        st.metric("Macro F1", f"{meta['f1_macro']:.3f}")
        st.caption(
            f"{meta['n_classes']} disease classes · "
            f"{meta['n_train']} train / {meta['n_test']} test · "
            f"calibrated TF-IDF + LinearSVC"
        )
        st.divider()

        st.markdown("### Session journal")
        st.metric("Entries logged", len(st.session_state[ui.JOURNAL_KEY]))
        if st.button("🧹 Clear journal", use_container_width=True):
            st.session_state[ui.JOURNAL_KEY] = []
            st.session_state[ui.INPUT_KEY] = ""
            st.rerun()

        st.divider()
        st.markdown("### Severity levels")
        for level in SEVERITY_COLOR:
            st.markdown(ui.severity_chip(level), unsafe_allow_html=True)
            st.caption(SEVERITY_DESCRIPTION[level])

        st.divider()
        st.caption(
            "⚠️ Academic prototype only. Does not provide medical advice. "
            "If you are unwell, contact a qualified clinician."
        )


# ---------------------------------------------------------------------------
# Header + escalation banner
# ---------------------------------------------------------------------------

def _render_header() -> None:
    st.title("🩺 Symptom Triage Journal")
    st.markdown(PAGE_INTRO)

    if st.session_state[ui.JOURNAL_KEY]:
        verdict = ui.current_escalation()
        if verdict.level != "Stable":
            ui.render_escalation_banner(verdict, show_window=False)


# ---------------------------------------------------------------------------
# Examples row
# ---------------------------------------------------------------------------

def _render_examples_row() -> None:
    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state[ui.INPUT_KEY] = text
                st.rerun()


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

def _render_result_strip(pred) -> None:
    sev_color = SEVERITY_COLOR[pred.severity]
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.markdown(
            ui.info_card(
                label="Most likely disease",
                value=pred.disease.title(),
                accent=sev_color,
                sublabel=f"Confidence: {pred.confidence * 100:.1f}%",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            ui.solid_card(
                label="Severity",
                value=pred.severity,
                color=sev_color,
                sublabel=pred.severity_source.replace("_", " "),
            ),
            unsafe_allow_html=True,
        )
    with c3:
        rf_count = len(pred.red_flags_hit)
        rf_color = "#d73027" if rf_count else "#7fbf7b"
        st.markdown(
            ui.info_card(
                label="Red-flag phrases",
                value=str(rf_count),
                accent=rf_color,
                sublabel="detected in your text",
            ),
            unsafe_allow_html=True,
        )


def _render_differential(pred) -> None:
    st.markdown("#### Differential diagnosis (top 3)")
    medals = ("🥇", "🥈", "🥉")
    cols = st.columns(len(pred.differential))
    for medal, col, (disease, prob) in zip(medals, cols, pred.differential):
        with col:
            st.markdown(f"**{medal} {disease.title()}**")
            st.progress(float(prob))
            st.caption(f"{prob * 100:.1f}%")


def _render_explainability(pred) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Why this prediction?")
        if pred.drivers:
            st.caption(
                "Top tokens / bigrams from your text that pushed the model toward this disease."
            )
            for token, weight in pred.drivers:
                st.markdown(f"- `{token}`  ·  weight **{weight:.3f}**")
        else:
            st.caption("No strong driver tokens recovered for this input.")
    with right:
        st.markdown("#### Red-flag detection")
        if pred.red_flags_hit:
            st.error("Clinical red-flag phrases detected — severity overridden to **Urgent**.")
            for flag in pred.red_flags_hit:
                st.markdown(f"- 🚩 **{flag}**")
        else:
            st.success("No clinical red-flag phrases detected in your text.")


def _render_session_escalation() -> None:
    if not st.session_state[ui.JOURNAL_KEY]:
        return
    st.markdown("---")
    st.markdown("### Session escalation status")
    ui.render_escalation_banner(ui.current_escalation())
    st.markdown(
        "👉 Open **Journal Timeline** in the sidebar to see your symptom history, "
        "or **Clinician Report** to export a structured summary."
    )


# ---------------------------------------------------------------------------
# Main body
# ---------------------------------------------------------------------------

def _render_triage_form() -> None:
    st.markdown("### How are you feeling right now?")
    text = st.text_area(
        "Describe your symptoms",
        key=ui.INPUT_KEY,
        height=130,
        placeholder=(
            "e.g. I've had a throbbing headache on the right side since yesterday, "
            "with nausea and sensitivity to light…"
        ),
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 5])
    with col_a:
        submitted = st.button("Run triage ▶", type="primary", use_container_width=True)
    with col_b:
        log_to_journal = st.checkbox("Log this entry to my journal", value=True)

    if not submitted:
        return

    if not text.strip():
        st.warning("Please describe your symptoms before running triage.")
        return

    pred = predict(text, st.session_state[ui.PIPE_KEY])

    if log_to_journal:
        st.session_state[ui.JOURNAL_KEY].append(
            JournalEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_text=text,
                prediction=pred,
            )
        )

    st.markdown("---")
    st.markdown("### Triage result")
    _render_result_strip(pred)
    _render_differential(pred)
    _render_explainability(pred)
    _render_session_escalation()


def _render_about() -> None:
    st.markdown("---")
    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **What's novel here?** Most public symptom-checker prototypes return a single
            disease prediction from one text box and stop there. This app adds three layers:

            1. **Severity-aware triage** — deterministic disease→severity mapping plus an
               overridable red-flag phrase detector (chest pain, shortness of breath, blood, …).
            2. **Longitudinal escalation detection** — the app remembers past entries within
               a session and flags worsening trends (Mild → Moderate → Urgent) over time.
            3. **Transparent explainability** — top driver tokens, top-3 differential
               probabilities, and a clinician-shareable Markdown export of the whole session.

            The disease classifier is a calibrated TF-IDF + LinearSVC pipeline trained on the
            Kaggle *Symptom2Disease* dataset (1,200 rows, 24 classes). Calibration (Platt
            scaling) makes the differential probabilities statistically meaningful rather
            than raw SVM decision scores.
            """
        )


def main() -> None:
    _render_sidebar()
    _render_header()
    _render_examples_row()
    _render_triage_form()
    _render_about()


main()
