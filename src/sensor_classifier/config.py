"""Strict, centralized YAML configuration with relative paths resolved from project root."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    root: Path
    dataset: dict[str, Any]
    preprocessing: dict[str, Any]
    models: dict[str, Any]
    evaluation: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.evaluation["seed"])

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "preprocessing": self.preprocessing,
                "models": self.models, "evaluation": self.evaluation}


def load_config(root: Path, config_dir: str = "configs") -> Settings:
    root = Path(root).resolve()
    folder = root / config_dir
    data: dict[str, dict[str, Any]] = {}
    for name in ("dataset", "preprocessing", "models", "evaluation"):
        path = folder / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing configuration: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}")
        data[name] = raw
    ds, ev, mod = data["dataset"], data["evaluation"], data["models"]
    if ds.get("kind") not in {"uci_har", "csv"}:
        raise ValueError("dataset.kind must be uci_har or csv")
    if not isinstance(ds.get("path"), str) or not ds["path"].strip():
        raise ValueError("dataset.path must be a nonempty string")
    expected = "official" if ds["kind"] == "uci_har" else ds.get("split_strategy")
    if expected not in {"official", "stratified", "group", "chronological"}:
        raise ValueError("Unknown dataset.split_strategy")
    if ds["kind"] == "uci_har" and ds.get("split_strategy") != "official":
        raise ValueError("UCI HAR must retain the official test split")
    if ds["kind"] == "csv":
        if not ds.get("target_column"):
            raise ValueError("CSV requires target_column")
        if expected == "group" and not ds.get("group_column"):
            raise ValueError("Group split requires group_column")
        if expected == "chronological" and not ds.get("time_column"):
            raise ValueError("Chronological split requires time_column")
        frac = ds.get("holdout_fraction", 0.25)
        if not isinstance(frac, (float, int)) or not 0 < frac < 0.5:
            raise ValueError("holdout_fraction must be in (0, 0.5)")
    if int(ev.get("folds", 0)) < 2 or int(ev.get("folds", 0)) > 10:
        raise ValueError("evaluation.folds must be between 2 and 10")
    if ev.get("primary_metric") not in {"macro_f1", "balanced_accuracy", "accuracy"}:
        raise ValueError("Unsupported primary metric")
    if int(ev.get("n_jobs", 1)) == 0 or int(ev.get("n_jobs", 1)) < -1:
        raise ValueError("n_jobs must be -1 or positive")
    if int(ev.get("bootstrap_replicates", 0)) < 0:
        raise ValueError("bootstrap_replicates must be >= 0")
    if not isinstance(mod.get("models"), list) or not mod["models"]:
        raise ValueError("models.models must be a nonempty list")
    if int(mod.get("search_iterations", 0)) < 1:
        raise ValueError("search_iterations must be >=1")
    if not 0 <= float(mod.get("selection_tolerance", 0)) <= 0.1:
        raise ValueError("selection_tolerance outside [0, 0.1]")
    return Settings(root, **data)
