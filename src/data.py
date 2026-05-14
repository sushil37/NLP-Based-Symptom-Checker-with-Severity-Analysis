"""
data.py
=======

Loading + preprocessing for the Symptom2Disease dataset.
Shared by every training script and the Streamlit app so we have one
canonical pipeline (no silent drift between notebook and app).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent  # coursework/project/
DATA_PATH = ROOT / "dataset" / "Symptom2Disease.csv"
print('DATA_PATH', DATA_PATH)

RANDOM_SEED = 42


def clean_text(t: str) -> str:
    """Lowercase, strip non-alphanumerics, collapse whitespace."""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Symptom2Disease CSV and normalise labels to lowercase."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df[["label", "text"]].dropna().reset_index(drop=True)
    df["label_norm"] = df["label"].str.lower().str.strip()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.30,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified train / test split on the normalised label."""
    return train_test_split(
        df["clean_text"],
        df["label_norm"],
        test_size=test_size,
        random_state=seed,
        stratify=df["label_norm"],
    )
