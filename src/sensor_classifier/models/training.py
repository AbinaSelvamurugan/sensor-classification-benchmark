"""Train model chosen using CV, record training-side diagnostics and feature evidence."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from sensor_classifier.evaluation.metrics import classification_metrics


def fit_selected(model: Pipeline, X: pd.DataFrame, y: pd.Series, labels: list[str]) -> tuple[Pipeline, dict]:
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    train_predictions = model.predict(X)
    return model, {"fit_seconds": fit_seconds,
                   "train_metrics": classification_metrics(y, train_predictions, labels)}


def explain_on_cv_validation(model: Pipeline, X: pd.DataFrame, y: pd.Series,
                             train_ids: np.ndarray, validation_ids: np.ndarray,
                             seed: int, metric: str) -> pd.DataFrame:
    """Never use held-out test labels to compute model-selection-time importance."""
    from sklearn.base import clone

    candidate = clone(model).fit(X.iloc[train_ids], y.iloc[train_ids])
    score = {"macro_f1": "f1_macro", "accuracy": "accuracy",
             "balanced_accuracy": "balanced_accuracy"}[metric]
    result = permutation_importance(candidate, X.iloc[validation_ids], y.iloc[validation_ids],
                                    scoring=score, n_repeats=2, random_state=seed, n_jobs=1)
    return pd.DataFrame({"feature": list(X.columns), "importance": result.importances_mean,
                         "std": result.importances_std, "method": "permutation_cv_validation"}).sort_values(
                             "importance", ascending=False).reset_index(drop=True)


def model_native_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["classifier"]
    names = model.named_steps["preprocess"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        method = "impurity_mdi"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).mean(axis=0)
        method = "mean_abs_linear_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "method"])
    if len(values) != len(names):
        raise AssertionError("Transformed feature importance/name mismatch")
    return pd.DataFrame({"feature": names, "importance": values, "method": method}).sort_values(
        "importance", ascending=False).reset_index(drop=True)
