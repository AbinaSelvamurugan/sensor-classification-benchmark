"""Synthetic software fixtures for tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def synthetic():
    x, y = make_classification(n_samples=90, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=14)
    X = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    labels = pd.Series(np.array(["walk", "sit", "stand"])[y], name="activity")
    groups = pd.Series(np.repeat(np.arange(15), 6), name="subject")
    return X, labels, groups
