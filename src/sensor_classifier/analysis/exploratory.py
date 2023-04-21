"""Save descriptive train-only EDA tables and plots."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensor_classifier.reporting.plots import eda_figures


def explore(X: pd.DataFrame, y: pd.Series, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stats = X.select_dtypes(include="number").describe().T
    stats.to_csv(folder / "train_descriptive_statistics.csv", index_label="feature")
    y.value_counts().sort_index().rename_axis("class").reset_index(name="train_count").to_csv(
        folder / "train_class_counts.csv", index=False)
    eda_figures(X, y, folder)
