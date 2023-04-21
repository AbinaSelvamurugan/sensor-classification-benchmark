"""Common estimator factory. No training or preprocessing is done outside Pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sensor_classifier.data.preprocessing import build_preprocessing

MODEL_NAMES = ("dummy", "logistic_regression", "decision_tree", "random_forest", "rbf_svm")
SIMPLICITY = {"dummy": 0, "decision_tree": 1, "logistic_regression": 2,
              "random_forest": 3, "rbf_svm": 4}


def make_pipeline(name: str, X: pd.DataFrame, seed: int, cfg: dict | None = None) -> Pipeline:
    cfg = cfg or {}
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown estimator: {name}")
    classifiers = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1800, C=1.0, random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                 random_state=seed, n_jobs=int(cfg.get("n_jobs", 1))),
        "rbf_svm": SVC(C=2.0, kernel="rbf", gamma="scale", decision_function_shape="ovr"),
    }
    # Dummy and trees don't need scaling; linear/SVM models generally do.
    scale = name in {"logistic_regression", "rbf_svm"} and cfg.get("scale_linear", True)
    return Pipeline([("preprocess", build_preprocessing(
        X, scale=scale, numeric_imputer=cfg.get("numeric_imputer", "median"),
        categorical_imputer=cfg.get("categorical_imputer", "most_frequent"))),
        ("classifier", classifiers[name])])
