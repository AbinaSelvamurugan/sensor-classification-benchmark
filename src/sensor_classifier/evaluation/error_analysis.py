"""Directional errors and unordered confusion pairs."""
from __future__ import annotations

import pandas as pd


def confusion_patterns(raw, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    directed = []
    paired = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and raw[i][j] > 0:
                directed.append({"true_class": actual, "predicted_class": predicted,
                                 "count": int(raw[i][j])})
        for j in range(i + 1, len(labels)):
            count = int(raw[i][j] + raw[j][i])
            if count:
                paired.append({"class_a": actual, "class_b": labels[j], "total_errors": count,
                               "a_to_b": int(raw[i][j]), "b_to_a": int(raw[j][i])})
    direction = pd.DataFrame(directed, columns=["true_class", "predicted_class", "count"])
    pairs = pd.DataFrame(paired, columns=["class_a", "class_b", "total_errors", "a_to_b", "b_to_a"])
    if not direction.empty:
        direction = direction.sort_values("count", ascending=False).reset_index(drop=True)
    if not pairs.empty:
        pairs = pairs.sort_values("total_errors", ascending=False).reset_index(drop=True)
    return direction, pairs
