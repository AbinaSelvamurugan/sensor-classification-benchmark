import numpy as np

from sensor_classifier.evaluation.confusion_matrix import matrix_pair
from sensor_classifier.evaluation.metrics import classification_metrics, cluster_bootstrap


def test_label_order_and_imbalanced_macro():
    true = ["rare", "major", "major", "major"]
    pred = ["major"] * 4
    labels = ["rare", "major"]
    metrics = classification_metrics(true, pred, labels)
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_recall"] == 0.5
    assert metrics["per_class"]["rare"]["recall"] == 0
    assert len(metrics["warnings"]) == 1
    raw, norm = matrix_pair(true, pred, labels)
    assert raw.tolist() == [[0, 1], [0, 3]]
    assert np.allclose(norm.sum(axis=1), 1)


def test_cluster_interval_uses_groups():
    true = np.array(["a", "b"] * 10)
    pred = true.copy()
    groups = np.repeat(np.arange(5), 4)
    result = cluster_bootstrap(true, pred, groups, ["a", "b"], 20, 13)
    assert result["interval_95"] == [1.0, 1.0]
