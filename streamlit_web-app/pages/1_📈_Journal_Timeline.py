"""
Journal Timeline page — longitudinal view of the user's session journal.

Renders three blocks:

* a summary metrics strip,
* an Altair severity-over-time line chart,
* a disease-frequency bar chart (when at least two distinct diseases appear),
* a tabular view + per-entry expandable cards.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

# Make the parent app/ directory importable when Streamlit launches the page.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import altair as alt
import pandas as pd
import streamlit as st

import ui
from constants import SEVERITY_COLOR, SEVERITY_RANK
from schemas import JournalEntry

st.set_page_config(page_title="Journal Timeline", page_icon="📈", layout="wide")

st.title("📈 Journal Timeline")
st.caption("How your symptoms have evolved across this session.")

entries: Sequence[JournalEntry] = ui.require_journal()


# ---------------------------------------------------------------------------
# Build tidy DataFrame
# ---------------------------------------------------------------------------

def _entries_to_dataframe(items: Sequence[JournalEntry]) -> pd.DataFrame:
    rows: List[dict] = []
    for i, entry in enumerate(items, start=1):
        pred = entry.prediction
        rows.append(
            {
                "#": i,
                "Time": entry.timestamp,
                "Disease": pred.disease.title(),
                "Confidence": round(pred.confidence, 3),
                "Severity": pred.severity,
                "Severity rank": SEVERITY_RANK[pred.severity],
                "Red flags": ", ".join(pred.red_flags_hit) if pred.red_flags_hit else "—",
                "Symptoms": entry.user_text,
            }
        )
    return pd.DataFrame(rows)


df = _entries_to_dataframe(entries)


# ---------------------------------------------------------------------------
# Summary metrics + escalation verdict
# ---------------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total entries", len(df))
m2.metric("Unique diseases", df["Disease"].nunique())
m3.metric("Urgent entries", int(df["Severity"].value_counts().get("Urgent", 0)))
m4.metric("Peak severity", df.loc[df["Severity rank"].idxmax(), "Severity"])

ui.render_escalation_banner(ui.current_escalation())

st.divider()


# ---------------------------------------------------------------------------
# Severity timeline
# ---------------------------------------------------------------------------

def _severity_timeline(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=120))
        .encode(
            x=alt.X("#:O", title="Entry #", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Severity rank:Q",
                scale=alt.Scale(domain=[-0.3, 2.3]),
                axis=alt.Axis(
                    values=[0, 1, 2],
                    labelExpr=(
                        "datum.value === 0 ? 'Mild' : "
                        "datum.value === 1 ? 'Moderate' : 'Urgent'"
                    ),
                ),
                title="Severity",
            ),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(
                    domain=list(SEVERITY_COLOR.keys()),
                    range=list(SEVERITY_COLOR.values()),
                ),
                legend=alt.Legend(title="Severity"),
            ),
            tooltip=[
                alt.Tooltip("#:O", title="Entry"),
                alt.Tooltip("Disease:N"),
                alt.Tooltip("Severity:N"),
                alt.Tooltip("Confidence:Q", format=".1%"),
                alt.Tooltip("Symptoms:N", title="Symptoms"),
            ],
        )
        .properties(height=300)
    )


st.markdown("### Severity over time")
st.altair_chart(_severity_timeline(df), use_container_width=True)


# ---------------------------------------------------------------------------
# Disease frequency (only shown when there is something to compare)
# ---------------------------------------------------------------------------

def _disease_frequency_chart(data: pd.DataFrame) -> alt.Chart:
    freq = data["Disease"].value_counts().reset_index()
    freq.columns = ["Disease", "Count"]
    return (
        alt.Chart(freq)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("Count:Q", title="Occurrences", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Disease:N", sort="-x", title=""),
            color=alt.value("#4C78A8"),
            tooltip=["Disease:N", "Count:Q"],
        )
        .properties(height=max(80, freq.shape[0] * 38))
    )


if df["Disease"].nunique() > 1:
    st.markdown("### Disease frequency this session")
    st.altair_chart(_disease_frequency_chart(df), use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Tabular view
# ---------------------------------------------------------------------------

st.markdown("### All entries")
st.dataframe(
    df.drop(columns=["Severity rank"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Confidence": st.column_config.ProgressColumn(
            "Confidence", format="%.1%%", min_value=0, max_value=1
        ),
        "Symptoms": st.column_config.TextColumn("Symptoms", width="large"),
    },
)


# ---------------------------------------------------------------------------
# Per-entry expandable cards
# ---------------------------------------------------------------------------

def _render_entry_card(index: int, entry: JournalEntry) -> None:
    pred = entry.prediction
    sev_color = SEVERITY_COLOR[pred.severity]
    title = (
        f"#{index} · {pred.disease.title()} · "
        f"[{pred.severity}] · {entry.timestamp[:19].replace('T', ' ')} UTC"
    )
    with st.expander(title):
        st.markdown(
            f'<div style="border-left:4px solid {sev_color};padding:8px 14px;'
            f'background:#f9f9f9;border-radius:4px;font-style:italic;">'
            f"{entry.user_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Differential**")
            for disease, conf in pred.differential:
                st.markdown(f"- {disease.title()} — {conf * 100:.1f}%")
        with c2:
            st.markdown("**Top driver tokens**")
            if pred.drivers:
                for token, weight in pred.drivers:
                    st.markdown(f"- `{token}` ({weight:.3f})")
            else:
                st.caption("No drivers recovered.")
        with c3:
            st.markdown("**Red-flag phrases**")
            if pred.red_flags_hit:
                for flag in pred.red_flags_hit:
                    st.markdown(f"- 🚩 {flag}")
            else:
                st.caption("None detected.")
        st.caption(
            f"Severity source: `{pred.severity_source}` · "
            f"Confidence: {pred.confidence * 100:.1f}%"
        )


st.markdown("### Per-entry details")
for i, entry in enumerate(entries, start=1):
    _render_entry_card(i, entry)
