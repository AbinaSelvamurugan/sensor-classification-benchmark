import json

import pandas as pd

from sensor_classifier.evaluation.error_analysis import confusion_patterns
from sensor_classifier.reporting.plots import comparison_figure, confusion_figures, eda_figures
from sensor_classifier.reporting.report import create_report
from sensor_classifier.utils.reproducibility import write_json


def test_plotting_and_errors(tmp_path, synthetic):
    X, y, _ = synthetic
    eda_figures(X, y, tmp_path)
    confusion_figures(__import__("numpy").array([[2, 1], [0, 3]]), ["a", "b"], tmp_path)
    comparison_figure(pd.DataFrame({"model": ["dummy"], "macro_f1_mean": [0.2],
                                    "macro_f1_std": [0.01]}), "macro_f1", tmp_path)
    assert (tmp_path / "confusion_normalized.png").exists()
    _, pairs = confusion_patterns([[2, 1], [0, 3]], ["a", "b"])
    assert pairs.total_errors.iloc[0] == 1


def test_report_uses_stored_values(tmp_path):
    write_json(tmp_path / "configuration.json", {"evaluation": {"primary_metric": "macro_f1", "folds": 3,
                        "seed": 42}, "models": {"models": ["dummy"]}})
    write_json(tmp_path / "dataset_manifest.json", {"dataset": "synthetic test fixture"})
    write_json(tmp_path / "data_quality.json", {"rows": 10, "features": 1, "classes": 2,
               "missing_cells": 0, "infinite_cells": 0, "duplicate_predictor_rows": 0,
               "memory_bytes": 160, "label_counts": {"a": 5, "b": 5}, "concerns": []})
    write_json(tmp_path / "selection.json", {"selected_model": "dummy", "metric": "macro_f1",
              "selection_tolerance": 0.005, "within_tolerance": ["dummy"], "rationale": "test"})
    per_class = {k: {"precision": .5, "recall": .5, "f1": .5, "support": 5} for k in ["a", "b"]}
    metrics = {k: 0.5 for k in ("accuracy", "balanced_accuracy", "macro_precision",
                "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "micro_f1")}
    metrics.update({"per_class": per_class, "warnings": [], "confusion_matrix": [[4, 1], [4, 1]]})
    write_json(tmp_path / "final_test.json", {"metrics": metrics, "feature_names": ["f"],
              "labels": ["a", "b"], "split_strategy": "group", "test_rows": 10,
              "train_groups": 10, "test_groups": 5, "test_class_counts": {"a": 5, "b": 5},
              "bootstrap": {"available": False, "reason": "fixture"},
              "fit_seconds": 0.1, "predict_seconds": 0.01,
              "training_metrics": {"macro_f1": 0.5}})
    pd.DataFrame([{"model": "dummy", "macro_f1_mean": .5, "macro_f1_std": 0.0,
                   "accuracy_mean": .5, "balanced_accuracy_mean": .5,
                   "fit_seconds_mean": .1}]).to_csv(tmp_path / "cv_summary.csv", index=False)
    pd.DataFrame([{"model": "dummy", "fold": 1, "macro_f1": .5}]).to_csv(
        tmp_path / "cv_folds.csv", index=False)
    pd.DataFrame([{"class_a": "a", "class_b": "b", "total_errors": 5,
                   "a_to_b": 1, "b_to_a": 4}]).to_csv(tmp_path / "confusion_pairs.csv", index=False)
    path = create_report(tmp_path)
    assert "0.5000" in path.read_text()
    assert "synthetic test fixture" in path.read_text()
