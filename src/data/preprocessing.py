"""Data preprocessing utilities for clinical text and image features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = ["patient_id", "clinical_text", "image_path", "label", "anomaly_score", "report"]


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_manifest(df)
    return df


def split_summary(df: pd.DataFrame) -> Dict[str, int]:
    return df["label"].value_counts().to_dict()
