"""Fold-local imputation, encoding, and optional scaling."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing(X: pd.DataFrame, scale: bool,
                        numeric_imputer: str = "median",
                        categorical_imputer: str = "most_frequent") -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.columns.difference(numeric, sort=False).tolist()
    if numeric_imputer not in {"median", "mean"} or categorical_imputer != "most_frequent":
        raise ValueError("Unsupported imputer configuration")
    transformers = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy=numeric_imputer, add_indicator=False))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy=categorical_imputer)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise ValueError("No supported feature columns")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
