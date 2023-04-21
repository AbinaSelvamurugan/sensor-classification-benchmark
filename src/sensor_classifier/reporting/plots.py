"""Deterministic, data-linked EDA, model comparison, and error plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=135, bbox_inches="tight")
    plt.close()


def eda_figures(X: pd.DataFrame, y: pd.Series, folder: Path) -> None:
    """Explore TRAIN only; selected features are explicitly marked illustrative."""
    folder.mkdir(parents=True, exist_ok=True)
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot.bar(ax=ax, color="steelblue")
    ax.set(xlabel="True activity", ylabel="Training windows", title="Training class distribution")
    ax.tick_params(axis="x", rotation=35)
    _save(folder / "train_class_distribution.png")
    numeric = X.select_dtypes(include="number")
    if len(numeric.columns):
        # Deliberately fixed, reproducible subset: first five published features.
        sample = numeric.iloc[:, :min(5, numeric.shape[1])]
        fig, axes = plt.subplots(len(sample.columns), 1, figsize=(9, 2.5 * len(sample.columns)))
        for col, ax in zip(sample.columns, np.atleast_1d(axes)):
            ax.hist(sample[col].dropna(), bins=35, color="steelblue")
            ax.set(xlabel=col, ylabel="Windows")
        fig.suptitle("First numeric features (not importance ranking)")
        _save(folder / "feature_distributions.png")
        corr = numeric.iloc[:, :min(12, numeric.shape[1])].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="coolwarm", ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_title("Pearson correlations: first 12 numeric features")
        _save(folder / "feature_correlations.png")
        if numeric.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            cap = min(1500, len(X))
            subset = np.linspace(0, len(X) - 1, cap, dtype=int)
            # Sampling by evenly spaced row indices avoids large, slow scatterplots.
            sns.scatterplot(x=numeric.iloc[subset, 0], y=numeric.iloc[subset, 1],
                            hue=y.iloc[subset], s=12, linewidth=0, ax=ax)
            ax.set_title("First two numeric features (descriptive association only)")
            ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
            _save(folder / "feature_relationships.png")
        fig, ax = plt.subplots(figsize=(9, 3))
        first = numeric.iloc[:, 0].replace([np.inf, -np.inf], np.nan).dropna()
        sns.boxplot(x=first, ax=ax)
        ax.set(title="Distribution/outliers of first numeric feature", xlabel=numeric.columns[0])
        _save(folder / "outlier_inspection.png")
    missing = X.isna().sum().sort_values(ascending=False).head(25)
    if missing.sum() > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        missing[missing > 0].plot.bar(ax=ax)
        ax.set(title="Training missingness (top 25 features)", ylabel="Missing cells")
        _save(folder / "missingness.png")


def comparison_figure(summary: pd.DataFrame, metric: str, folder: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ordered = summary.sort_values(metric + "_mean", ascending=True)
    ax.barh(ordered["model"], ordered[metric + "_mean"],
            xerr=ordered[metric + "_std"].fillna(0), capsize=3)
    ax.set(xlim=(0, 1.0), xlabel=f"CV {metric}, mean ± fold SD",
           title="Training-only CV comparison (folds are not independent CIs)")
    _save(folder / "cv_model_comparison.png")


def confusion_figures(raw: np.ndarray, labels: list[str], folder: Path) -> None:
    denom = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(raw, denom, out=np.zeros_like(raw, dtype=float), where=denom != 0)
    for matrix, name, fmt in [(raw, "confusion_raw.png", "d"),
                              (normalized, "confusion_normalized.png", ".2f")]:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels,
                    yticklabels=labels, ax=ax)
        ax.set(xlabel="Predicted class", ylabel="True class",
               title="Test confusion matrix: " + ("counts" if fmt == "d" else "row-normalized"))
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        _save(folder / name)


def importance_figure(table: pd.DataFrame, folder: Path, name: str) -> None:
    if table.empty:
        return
    selected = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(selected["feature"], selected["importance"])
    ax.set(xlabel="Importance score (method specified in CSV)", title=name)
    _save(folder / "feature_importance.png")
