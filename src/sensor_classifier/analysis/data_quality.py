"""Collect serializable data-quality information; never infer quality from test labels early."""
from __future__ import annotations

import pandas as pd

from sensor_classifier.data.validation import check_features


def quality_report(X: pd.DataFrame, y: pd.Series, labels: list[str]) -> dict:
    report = check_features(X, y, labels)
    report["notes"] = [
        "Descriptive statistics refer to training only; final test labels not inspected for model selection.",
        "Identical predictor rows can be legitimate overlapping windows, and are retained.",
        "Feature correlations are sampled from the first 60 numeric columns, not a feature ranking.",
        "Outliers are not deleted; feature transforms are fit independently in training folds.",
    ]
    return report
