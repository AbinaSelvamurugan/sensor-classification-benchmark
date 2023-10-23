import numpy as np
import pandas as pd
import pytest

from sensor_classifier.data.splitting import split_indices
from sensor_classifier.evaluation.cross_validation import make_splits


def test_group_holdout_and_cv(synthetic):
    X, y, g = synthetic
    tr, te = split_indices(y, "group", 0.25, 42, g)
    assert not (set(g.iloc[tr]) & set(g.iloc[te]))
    splits = make_splits(X, y, g, "group", 3, 42)
    assert len(splits) == 3
    for a, b in splits:
        assert not (set(g.iloc[a]) & set(g.iloc[b]))


def test_stratified_split(synthetic):
    X, y, _ = synthetic
    a, b = split_indices(y, "stratified", 0.25, 11)
    assert len(set(a) & set(b)) == 0
    assert set(y.iloc[a]) == set(y.iloc[b])


def test_no_group_overlap_in_cv(synthetic):
    X, y, g = synthetic
    with pytest.raises(ValueError):
        make_splits(X, y, g.iloc[:10], "group", 3, 4)


def test_chronological_order():
    y = pd.Series(["a", "b"] * 20)
    tr, te = split_indices(y, "chronological", 0.25, 1)
    assert tr.max() < te.min()
