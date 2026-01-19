from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import BenchmarkConfig


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _looks_integer(series: pd.Series) -> bool:
    """True if numeric values are effectively integers (even if stored as float)."""
    s = series.dropna()
    if s.empty:
        return True
    if not _is_numeric(s):
        return False
    arr = s.to_numpy(dtype=float, copy=False)
    return np.all(np.isclose(arr, np.round(arr)))


def get_feature_type(series: pd.Series, include_binary: bool = True) -> str:
    """
    Classify a feature as: continuous / binary / categorical.

    This mirrors the improved heuristic from the original script:
    - if non-numeric: categorical
    - if not integer-like: continuous
    - if many unique ints: continuous
    - else: binary or categorical
    """
    s = series.dropna()
    if s.empty:
        return "categorical"

    if not _is_numeric(s):
        return "categorical"

    if not _looks_integer(s):
        return "continuous"

    nunique = s.nunique(dropna=True)
    if nunique > 10:
        return "continuous"
    if include_binary and nunique == 2:
        return "binary"
    return "categorical"


def one_hot_if_reasonable(
    df: pd.DataFrame,
    target_col: str,
    cfg: BenchmarkConfig,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    One-hot encode categorical columns if it doesn't explode dimensionality.
    Returns: (processed_df, continuous_cols, binary_cols, categorical_cols)

    If expansion is too large, returns an empty df.
    """
    feat_cols = [c for c in df.columns if c != target_col]

    feature_types: Dict[str, str] = {c: get_feature_type(df[c], include_binary=True) for c in feat_cols}
    cont_cols = [c for c, t in feature_types.items() if t == "continuous"]
    bin_cols = [c for c, t in feature_types.items() if t == "binary"]
    cat_cols = [c for c, t in feature_types.items() if t == "categorical"]

    if not cat_cols:
        out = df.copy()
        out.columns = out.columns.astype(str).str.replace(".", "_", regex=False)
        return out, cont_cols, bin_cols, cat_cols

    dummy = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    added_cols = dummy.shape[1] - df.shape[1]

    ok_ratio = added_cols < df.shape[0] * cfg.max_dummy_ratio
    ok_cols_rows = (dummy.shape[1] < df.shape[0]) if cfg.require_cols_lt_rows else True

    if ok_ratio and ok_cols_rows:
        dummy = dummy.copy()
        dummy.columns = dummy.columns.astype(str).str.replace(".", "_", regex=False)
        return dummy, cont_cols, bin_cols, cat_cols

    return pd.DataFrame(), cont_cols, bin_cols, cat_cols


def dataset_is_too_big(df: pd.DataFrame, cfg: BenchmarkConfig) -> bool:
    return (df.shape[0] > cfg.max_rows) or (df.shape[1] > cfg.max_cols)


def make_factor_grid(n_features: int, multipliers: Iterable[float]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for m in multipliers:
        k = int(round(n_features * m))
        k = max(1, k)
        out.append((k, m))
    return out

