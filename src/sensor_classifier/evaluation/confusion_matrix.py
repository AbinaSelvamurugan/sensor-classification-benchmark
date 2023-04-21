"""Confusion matrix with fixed true-row/predicted-column conventions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def matrix_pair(y_true, y_pred, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = confusion_matrix(y_true, y_pred, labels=labels)
    denominator = raw.sum(axis=1, keepdims=True)
    norm = np.divide(raw, denominator, out=np.zeros(raw.shape, dtype=float), where=denominator != 0)
    return raw, norm
