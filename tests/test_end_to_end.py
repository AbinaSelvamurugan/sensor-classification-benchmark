"""End-to-end CLI integration test on synthetic fixtures."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from sensor_classifier.cli import main


def test_cli_end_to_end_offline(tmp_path):
    X, y = make_classification(n_samples=90, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=5)
    dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    dataset["subject"] = [i // 6 for i in range(90)]
    dataset["target_label"] = pd.Series(y).map({0: "a", 1: "b", 2: "c"})
    (tmp_path / "data").mkdir()
    (tmp_path / "configs").mkdir()
    dataset.to_csv(tmp_path / "data" / "data.csv", index=False)
    files = {
        "dataset": "kind: csv\npath: data/data.csv\ntarget_column: target_label\ngroup_column: subject\nsplit_strategy: group\nholdout_fraction: 0.2\ndrop_columns: []\n",
        "preprocessing": "numeric_imputer: median\ncategorical_imputer: most_frequent\nscale_linear: true\n",
        "models": "models: [dummy, decision_tree, logistic_regression, random_forest, rbf_svm]\nselection_tolerance: 0.005\nsearch_iterations: 2\n",
        "evaluation": "seed: 42\nfolds: 3\nprimary_metric: macro_f1\nbootstrap_replicates: 20\nn_jobs: 1\n",
    }
    for name, content in files.items():
        (tmp_path / "configs" / f"{name}.yaml").write_text(content)
    args = ["--root", str(tmp_path), "--run-name", "software_fixture"]
    main(["run-all", *args, "--tune"])
    saved = tmp_path / "results" / "software_fixture"
    assert (saved / "EVALUATION_REPORT.md").is_file()
    assert (saved / "final_test.json").is_file()
    assert (saved / "cv_class_diagnostics.csv").is_file()
    assert (saved / "figures" / "confusion_raw.png").is_file()
    with pytest.raises(FileExistsError):
        main(["evaluate", *args, "--trust-model"])
    output = tmp_path / "out.csv"
    main(["predict", *args, "--input", str(tmp_path / "data" / "data.csv"),
          "--output", str(output), "--trust-model"])
    assert "prediction" in pd.read_csv(output).columns
    assert (tmp_path / "out.evaluation.json").is_file()
