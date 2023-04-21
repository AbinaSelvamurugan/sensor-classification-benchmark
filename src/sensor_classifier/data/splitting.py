"""Subject-independent, stratified, and chronological holdout strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_indices(y: pd.Series, strategy: str, fraction: float, seed: int,
                  groups: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.arange(len(y))
    if strategy == "stratified":
        train, test = train_test_split(indexes, test_size=fraction, random_state=seed,
                                      stratify=y)
    elif strategy == "group":
        if groups is None:
            raise ValueError("Group strategy requires groups")
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        train, test = next(splitter.split(indexes, y, groups=groups))
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise AssertionError("Group leakage across holdout")
    elif strategy == "chronological":
        cut = int(len(y) * (1 - fraction))
        if cut <= 0 or cut >= len(y):
            raise ValueError("Chronological split is empty")
        train, test = indexes[:cut], indexes[cut:]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if set(train) & set(test):
        raise AssertionError("Overlapping holdout indices")
    return np.asarray(train), np.asarray(test)
