"""Predeclared tolerance-based selection favors simpler models within tolerance."""
from __future__ import annotations

import pandas as pd

from sensor_classifier.models.registry import SIMPLICITY


def select_model(summaries: pd.DataFrame, metric: str, tolerance: float) -> dict:
    column = metric + "_mean"
    if summaries.empty or column not in summaries:
        raise ValueError("Missing CV summary for model selection")
    best = float(summaries[column].max())
    candidates = summaries[summaries[column] >= best - tolerance].copy()
    candidates["complexity_order"] = candidates["model"].map(SIMPLICITY)
    candidates = candidates.sort_values(["complexity_order", column], ascending=[True, False])
    selected = str(candidates.iloc[0]["model"])
    return {"selected_model": selected, "metric": metric, "best_observed_cv_score": best,
            "selected_cv_score": float(candidates.iloc[0][column]),
            "selection_tolerance": tolerance, "within_tolerance": candidates["model"].tolist(),
            "rationale": "Select simplest predefined estimator within tolerance of top training-only CV mean"}
