from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def build_feature_frame(rows: Iterable[Mapping[str, float]]) -> pd.DataFrame:
    """
    Build a dense feature frame from heterogeneous feed rows.
    Missing values are median-imputed per column.
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        raise ValueError("no feature rows provided")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("no numeric features in rows")
    df = df[numeric_cols].astype(float)
    medians = df.median(numeric_only=True)
    df = df.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians)
    return df


def split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise ValueError(f"missing target column: {target_col}")
    y = df[target_col].astype(int).to_numpy()
    X = df.drop(columns=[target_col]).astype(float)
    if X.empty:
        raise ValueError("no predictor columns after dropping target")
    return X, y
