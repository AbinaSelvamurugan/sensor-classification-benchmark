import pandas as pd

from sensor_classifier.evaluation.cross_validation import evaluate_cv, make_splits, summarize_folds
from sensor_classifier.models.registry import make_pipeline


def test_group_cv_runs_and_reports_fold_stats(synthetic):
    X, y, groups = synthetic
    splits = make_splits(X, y, groups, "group", 3, 5)
    model = make_pipeline("logistic_regression", X, 5)
    fold_table, diag = evaluate_cv(model, X, y, splits, sorted(y.unique()), "logistic_regression")
    summary = summarize_folds(fold_table)
    assert len(fold_table) == 3 and len(diag) == 9
    assert "macro_f1_mean" in summary and fold_table.macro_f1.between(0, 1).all()
