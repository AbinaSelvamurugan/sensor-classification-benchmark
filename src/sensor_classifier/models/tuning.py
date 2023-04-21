"""Small, explicitly bounded training-only randomized parameter searches."""
from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

SPACES = {
    "logistic_regression": {"classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0]},
    "decision_tree": {"classifier__max_depth": [5, 8, 12, 18],
                      "classifier__min_samples_leaf": [2, 5, 10]},
    "random_forest": {"classifier__max_depth": [None, 15, 25],
                      "classifier__min_samples_leaf": [1, 2, 4],
                      "classifier__max_features": ["sqrt", 0.5]},
    "rbf_svm": {"classifier__C": [0.5, 1.0, 2.0, 4.0, 8.0],
                "classifier__gamma": ["scale", 0.001, 0.01]},
}

SCORERS = {"macro_f1": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"}


def tune_pipeline(pipeline, name: str, X: pd.DataFrame, y: pd.Series,
                  splits: list, metric: str, seed: int, iterations: int, n_jobs: int = 1) -> tuple[object, dict, pd.DataFrame]:
    if name not in SPACES:
        raise ValueError(f"Tuning not supported for {name}")
    search = RandomizedSearchCV(pipeline, SPACES[name], n_iter=iterations,
                                scoring=SCORERS[metric], cv=splits,
                                refit=True, random_state=seed, n_jobs=n_jobs,
                                error_score="raise", return_train_score=True)
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start
    table = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, {"best_params": search.best_params_,
                                    "best_cv_score": float(search.best_score_),
                                    "duration_seconds": elapsed, "iterations": len(table),
                                    "warning": "Best search CV score is selection-optimistic; no nested CV performed"}, table
