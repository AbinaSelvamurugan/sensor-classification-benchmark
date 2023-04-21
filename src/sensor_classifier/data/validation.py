"""Validate data without suppressing missingness, infinities, or label issues."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def check_features(X: pd.DataFrame, y: pd.Series, labels: list[str]) -> dict:
    if not isinstance(X, pd.DataFrame) or len(X) == 0 or X.shape[1] == 0:
        raise ValueError("Features must be a nonempty DataFrame")
    if not X.columns.is_unique or any(not isinstance(c, str) for c in X.columns):
        raise ValueError("Feature names must be unique strings")
    if len(X) != len(y) or y.isna().any():
        raise ValueError("Target length differs or contains missing labels")
    if y.nunique() < 2 or not set(y.astype(str).unique()).issubset(set(labels)):
        raise ValueError("Invalid class labels")
    if any(str(col).lower() in {"activity", "target", "label", "subject_id", "subject"}
           for col in X.columns):
        raise ValueError("Target/subject identifier appears among features")
    bad_types = [c for c in X if pd.api.types.is_datetime64_any_dtype(X[c]) or
                 pd.api.types.is_complex_dtype(X[c])]
    if bad_types:
        raise ValueError(f"Datetime or complex features unsupported: {bad_types}")
    numeric = X.select_dtypes(include="number").columns.tolist()
    categoricals = X.columns.difference(numeric, sort=False).tolist()
    infinite = int(np.isinf(X[numeric].to_numpy(dtype=float)).sum()) if numeric else 0
    missing = int(X.isna().sum().sum())
    duplicates = int(X.duplicated().sum())
    class_counts = {str(k): int(v) for k, v in y.value_counts().sort_index().items()}
    concerns = []
    if missing:
        concerns.append(f"{missing} missing predictor cells; train-fold imputation required")
    if infinite:
        concerns.append(f"{infinite} infinite numeric cells; treated as missing explicitly")
    if duplicates:
        concerns.append(f"{duplicates} duplicated predictor rows; retained pending domain review")
    if min(class_counts.values()) < 5:
        concerns.append("Fewer than five samples in at least one class")
    if max(class_counts.values()) / min(class_counts.values()) > 3:
        concerns.append("Class-frequency ratio exceeds 3:1")
    if len(X) < len(X.columns) * 2:
        concerns.append("Few samples relative to feature dimensionality")
    for concern in concerns:
        warnings.warn(concern, stacklevel=2)
    sanitized = X[numeric].replace([np.inf, -np.inf], np.nan) if numeric else pd.DataFrame()
    desc = sanitized.describe().T if numeric else pd.DataFrame()
    corr = sanitized[numeric[:60]].corr() if numeric else pd.DataFrame()
    highly_corr = int((np.triu((corr.abs() > 0.95).to_numpy(), 1)).sum()) if not corr.empty else 0
    return {"rows": len(X), "features": X.shape[1], "classes": len(labels),
            "missing_cells": missing, "missing_by_feature": {str(k): int(v) for k, v in X.isna().sum().items() if v},
            "infinite_cells": infinite, "duplicate_predictor_rows": duplicates,
            "numeric_features": numeric, "categorical_features": categoricals,
            "dtypes": {str(k): str(v) for k, v in X.dtypes.items()},
            "label_counts": class_counts, "memory_bytes": int(X.memory_usage(deep=True).sum()),
            "correlation_sample_features": list(corr.columns),
            "strong_correlation_pairs_in_first_60_numeric": highly_corr,
            "descriptive_statistics": desc.replace({np.nan: None}).to_dict(orient="index"),
            "concerns": concerns}


def replace_inf_with_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Explicit conversion; caller must record infinite count from check_features."""
    out = X.copy()
    numeric = out.select_dtypes(include="number").columns
    if len(numeric):
        out.loc[:, numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out
