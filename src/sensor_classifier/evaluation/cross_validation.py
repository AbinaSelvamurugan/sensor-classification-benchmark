"""Leakage-safe fold construction, diagnostics, and model comparison."""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from sensor_classifier.evaluation.metrics import classification_metrics

LOG = logging.getLogger(__name__)
SCORE_COLUMNS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                 "weighted_precision", "weighted_recall", "weighted_f1"]


def make_splits(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None,
                strategy: str, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if strategy in {"official_subject_holdout__stratified_group_cv", "group"}:
        if groups is None or groups.nunique() < folds:
            raise ValueError("Too few distinct groups for group CV")
        try:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y, groups))
        except ValueError as exc:
            LOG.warning("StratifiedGroupKFold unavailable for this data: %s; falling back to GroupKFold", exc)
            splits = list(GroupKFold(n_splits=folds).split(X, y, groups))
    elif strategy == "stratified":
        if y.value_counts().min() < folds:
            raise ValueError("Fewer class instances than requested CV folds")
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True,
                                     random_state=seed).split(X, y))
    elif strategy == "chronological":
        splits = list(TimeSeriesSplit(n_splits=folds).split(X, y))
    else:
        raise ValueError(f"Unsupported CV method: {strategy}")
    classes = set(y.unique())
    for i, (tr, va) in enumerate(splits):
        if len(set(tr) & set(va)):
            raise AssertionError(f"CV fold {i} index leakage")
        if groups is not None and set(groups.iloc[tr]) & set(groups.iloc[va]):
            raise ValueError(f"Group overlap in fold {i}")
        if set(y.iloc[tr].unique()) != classes:
            raise ValueError(f"CV fold {i} training subset missing classes")
        if set(y.iloc[va].unique()) != classes:
            warnings.warn(f"CV fold {i} validation missing classes; fixed-label macro scores include zeros", stacklevel=2)
    return splits


def evaluate_cv(model, X: pd.DataFrame, y: pd.Series,
                splits: list[tuple[np.ndarray, np.ndarray]], labels: list[str],
                name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    for number, (tr, va) in enumerate(splits, start=1):
        instance = clone(model)
        start = time.perf_counter()
        instance.fit(X.iloc[tr], y.iloc[tr])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted = instance.predict(X.iloc[va])
        predict_seconds = time.perf_counter() - start
        validation = classification_metrics(y.iloc[va], predicted, labels)
        train_prediction = instance.predict(X.iloc[tr])
        training = classification_metrics(y.iloc[tr], train_prediction, labels)
        rows.append({"model": name, "fold": number, "train_rows": len(tr), "validation_rows": len(va),
                     "fit_seconds": fit_seconds, "prediction_seconds": predict_seconds,
                     **{k: validation[k] for k in SCORE_COLUMNS},
                     "train_macro_f1": training["macro_f1"], "train_accuracy": training["accuracy"],
                     "metric_warnings": "; ".join(validation["warnings"])})
        for label in labels:
            diagnostics.append({"model": name, "fold": number, "class": label,
                                "train_count": int((y.iloc[tr] == label).sum()),
                                "validation_count": int((y.iloc[va] == label).sum()),
                                "recall": validation["per_class"][label]["recall"],
                                "precision": validation["per_class"][label]["precision"]})
        LOG.info("%s fold %d: macro F1 %.4f, accuracy %.4f", name, number,
                 validation["macro_f1"], validation["accuracy"])
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def summarize_folds(fold_table: pd.DataFrame) -> pd.DataFrame:
    metric_cols = SCORE_COLUMNS + ["train_macro_f1", "train_accuracy", "fit_seconds", "prediction_seconds"]
    grouped = fold_table.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index().fillna(0)
