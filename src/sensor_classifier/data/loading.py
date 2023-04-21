"""Load original UCI HAR official partitions or user-provided tabular CSV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_classifier.config import Settings
from sensor_classifier.data.splitting import split_indices
from sensor_classifier.utils.reproducibility import sha256

UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
UCI_DOI = "https://doi.org/10.24432/C54S4K"


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series | None
    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series | None
    labels: list[str]
    strategy: str
    manifest: dict

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def _read_ucihar(part: str, folder: Path, names: list[str], label_lookup: dict[int, str]
                 ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    directory = folder / part
    x = pd.read_csv(directory / f"X_{part}.txt", sep=r"\s+", header=None, dtype=np.float64)
    y = pd.read_csv(directory / f"y_{part}.txt", sep=r"\s+", header=None).iloc[:, 0]
    subject = pd.read_csv(directory / f"subject_{part}.txt", sep=r"\s+", header=None).iloc[:, 0]
    if len(x) != len(y) or len(x) != len(subject) or x.shape[1] != len(names):
        raise ValueError(f"UCI HAR {part} dimensions do not match feature/label metadata")
    if not y.isin(label_lookup).all():
        raise ValueError("Unknown UCI HAR activity id")
    x.columns = names
    return x.reset_index(drop=True), y.map(label_lookup).reset_index(drop=True).rename("activity"), subject.reset_index(drop=True).rename("subject")


def load_ucihar(folder: Path) -> Dataset:
    folder = Path(folder)
    files = ["features.txt", "activity_labels.txt"] + [f"{split}/{prefix}_{split}.txt"
             for split in ("train", "test") for prefix in ("X", "y", "subject")]
    missing = [path for path in files if not (folder / path).is_file()]
    if missing:
        raise FileNotFoundError(f"UCI HAR files missing at {folder}: {missing}. Run download_dataset.py")
    features = pd.read_csv(folder / "features.txt", sep=r"\s+", header=None)
    if len(features) != 561 or features.shape[1] != 2:
        raise ValueError("Expected exactly 561 UCI HAR feature definitions")
    # Original feature names repeat. Stable index prefix makes feature keys unique.
    names = [f"f{int(num):03d}__{name}" for num, name in features.itertuples(index=False, name=None)]
    if len(names) != len(set(names)):
        raise ValueError("Feature identifiers are not unique")
    labels = pd.read_csv(folder / "activity_labels.txt", sep=r"\s+", header=None)
    label_lookup = {int(k): str(v) for k, v in labels.itertuples(index=False, name=None)}
    if len(label_lookup) != 6:
        raise ValueError("Expected six UCI HAR activities")
    xtr, ytr, gtr = _read_ucihar("train", folder, names, label_lookup)
    xte, yte, gte = _read_ucihar("test", folder, names, label_lookup)
    if set(gtr) & set(gte):
        raise ValueError("Official subject split is not disjoint")
    hashes = {path: sha256(folder / path) for path in files}
    return Dataset(xtr, ytr, gtr, xte, yte, gte, list(label_lookup.values()),
                   "official_subject_holdout__stratified_group_cv",
                   {"dataset": "UCI HAR Smartphones", "doi": UCI_DOI, "license": "CC BY 4.0",
                    "source_url": UCI_URL, "files_sha256": hashes,
                    "train_subjects": sorted(map(int, gtr.unique())),
                    "test_subjects": sorted(map(int, gte.unique())),
                    "partition": "publisher original"})


def load_csv(settings: Settings) -> Dataset:
    ds = settings.dataset
    path = settings.root / ds["path"]
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise FileNotFoundError(f"Expected a CSV file at {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("CSV empty or column names repeated")
    target = ds["target_column"]
    if target not in frame or frame[target].isna().any():
        raise ValueError(f"Missing target column or null target labels: {target}")
    if frame[target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    group = ds.get("group_column")
    time_col = ds.get("time_column")
    if group and group not in frame:
        raise ValueError(f"Group column missing: {group}")
    if time_col:
        if time_col not in frame:
            raise ValueError(f"Time column missing: {time_col}")
        parsed = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError("Cannot parse all time column values")
        frame = frame.iloc[np.argsort(parsed.to_numpy(), kind="stable")].reset_index(drop=True)
    excluded = [target] + ([group] if group else []) + ([time_col] if time_col else [])
    excluded += list(ds.get("drop_columns") or [])
    if len(set(excluded)) != len(excluded):
        excluded = list(dict.fromkeys(excluded))
    invalid = set(excluded) - set(frame.columns)
    if invalid:
        raise ValueError(f"Excluded columns missing: {sorted(invalid)}")
    features = frame.drop(columns=excluded)
    if features.empty or features.shape[1] == 0:
        raise ValueError("No predictors")
    # Reject identifier-like columns instead of silently ignoring them.
    suspects = [name for name in features if name.lower() in {"id", "subject", "subject_id", "device_id", "session_id", "record_id", "row_id", "user_id"} or name.lower().endswith("_id")]
    if suspects:
        raise ValueError(f"Possible identifier leakage in predictors: {suspects}. Declare group_column or drop_columns explicitly")
    y = frame[target].astype(str)
    for column in features:
        if features[column].astype(str).equals(y):
            raise ValueError(f"Feature {column} is a copy of the target")
        if (not pd.api.types.is_numeric_dtype(features[column])
                and len(features) >= 30
                and features[column].nunique(dropna=True) >= 0.95 * len(features)):
            raise ValueError(f"Predictor {column} looks like a near-unique categorical identifier; "
                             "exclude via drop_columns after review")
    strategy = ds["split_strategy"]
    if group and strategy != "group":
        raise ValueError("CSV with group_column must use group split to prevent group leakage")
    groups = frame[group] if group else None
    if groups is not None and groups.isna().any():
        raise ValueError("Missing group ids")
    train, test = split_indices(y, strategy, float(ds.get("holdout_fraction", 0.25)),
                                settings.seed, groups)
    labels = sorted(y.unique().tolist())
    if set(labels) - set(y.iloc[train].unique()):
        raise ValueError("Training partition missing a class: revise grouping/partition")
    return Dataset(features.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True),
                   groups.iloc[train].reset_index(drop=True) if groups is not None else None,
                   features.iloc[test].reset_index(drop=True), y.iloc[test].reset_index(drop=True),
                   groups.iloc[test].reset_index(drop=True) if groups is not None else None,
                   labels, strategy, {"dataset": str(path), "files_sha256": {path.name: sha256(path)},
                                      "excluded_columns": excluded, "train_indices": train.tolist(),
                                      "test_indices": test.tolist(), "source_type": "user_csv"})


def load_dataset(settings: Settings) -> Dataset:
    if settings.dataset["kind"] == "uci_har":
        return load_ucihar(settings.root / settings.dataset["path"])
    return load_csv(settings)
