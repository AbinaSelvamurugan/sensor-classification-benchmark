from dataclasses import replace

import pandas as pd
import pytest

from sensor_classifier.config import Settings, load_config
from sensor_classifier.data.loading import load_csv


def csv_settings(tmp_path, strategy="group", **kwargs):
    frame = pd.DataFrame({"x": range(30), "subject": [i // 3 for i in range(30)],
                          "activity": (["walk", "sit", "stand"] * 10)})
    frame.to_csv(tmp_path / "tiny.csv", index=False)
    ds = {"kind": "csv", "path": "tiny.csv", "target_column": "activity",
          "split_strategy": strategy, "holdout_fraction": 0.2,
          "group_column": "subject", "drop_columns": []}
    ds.update(kwargs)
    return Settings(tmp_path, ds, {}, {"models": ["dummy"]}, {"seed": 42})


def test_csv_loading_and_manifest(tmp_path):
    ds = load_csv(csv_settings(tmp_path))
    assert ds.X_train.columns.tolist() == ["x"]
    assert not (set(ds.groups_train) & set(ds.groups_test))
    assert len(ds.X_train) + len(ds.X_test) == 30


def test_csv_target_leakage_rejected(tmp_path):
    settings = csv_settings(tmp_path)
    path = tmp_path / "tiny.csv"
    frame = pd.read_csv(path)
    frame["shortcut"] = frame["activity"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="copy of the target"):
        load_csv(settings)


def test_csv_group_requires_group_split(tmp_path):
    with pytest.raises(ValueError):
        load_csv(csv_settings(tmp_path, strategy="stratified"))


def test_invalid_configuration(tmp_path):
    base = tmp_path / "configs"
    base.mkdir()
    for name, content in {"dataset": "kind: unknown\npath: a.csv\n",
                          "preprocessing": "numeric_imputer: median\n",
                          "models": "models: [dummy]\nsearch_iterations: 2\n",
                          "evaluation": "seed: 2\nfolds: 3\nprimary_metric: macro_f1\n"}.items():
        (base / f"{name}.yaml").write_text(content)
    with pytest.raises(ValueError):
        load_config(tmp_path)
