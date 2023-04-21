"""Multiclass metrics with explicit zero-division handling."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    if len(true) != len(predicted) or len(true) == 0:
        raise ValueError("Empty arrays or differing labels/predictions")
    if set(true) - set(labels) or set(predicted) - set(labels):
        raise ValueError("Observed labels missing from explicit label ordering")
    micro = precision_recall_fscore_support(true, predicted, labels=labels,
                                            average="micro", zero_division=0)
    macro = precision_recall_fscore_support(true, predicted, labels=labels,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(true, predicted, labels=labels,
                                               average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(true, predicted, labels=labels,
                                                average=None, zero_division=0)
    zero_predictions = sorted(set(labels) - set(predicted))
    missing_true = sorted(set(labels) - set(true))
    report = classification_report(true, predicted, labels=labels, output_dict=True,
                                   zero_division=0)
    return {"accuracy": float(accuracy_score(true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
            "micro_precision": float(micro[0]), "micro_recall": float(micro[1]),
            "micro_f1": float(micro[2]), "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]), "weighted_f1": float(weighted[2]),
            "per_class": {label: {"precision": float(per_class[0][i]),
                                  "recall": float(per_class[1][i]),
                                  "f1": float(per_class[2][i]),
                                  "support": int(per_class[3][i])}
                          for i, label in enumerate(labels)},
            "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
            "classification_report": report,
            "warnings": ([f"No predictions for: {zero_predictions}"] if zero_predictions else [])
            + ([f"No ground truth in this split for: {missing_true}"] if missing_true else []),
            "undefined_policy": "zero_division=0; explicit labels including absent classes"}


def cluster_bootstrap(y_true, y_pred, groups, labels: list[str],
                      repeats: int, seed: int, score: str = "macro_f1") -> dict:
    """Resample entire test subjects, not correlated overlapping rows."""
    if groups is None or len(np.unique(groups)) < 5 or repeats <= 0:
        return {"available": False, "reason": "Requires >=5 test groups and repeats>0"}
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    scores = []
    for _ in range(repeats):
        selected = rng.choice(keys, size=len(keys), replace=True)
        ids = np.concatenate([np.flatnonzero(groups == key) for key in selected])
        scores.append(classification_metrics(y_true[ids], y_pred[ids], labels)[score])
    return {"available": True, "method": "percentile cluster bootstrap (subject resampling)",
            "units": int(len(keys)), "replicates": repeats,
            "metric": score, "interval_95": np.quantile(scores, [0.025, 0.975]).tolist(),
            "caution": "Small subject count; descriptive uncertainty, not a significance test"}
