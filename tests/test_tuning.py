from sensor_classifier.evaluation.cross_validation import make_splits
from sensor_classifier.models.registry import make_pipeline
from sensor_classifier.models.tuning import tune_pipeline


def test_randomized_search_only_train_splits(synthetic):
    X, y, groups = synthetic
    splits = make_splits(X, y, groups, "group", 3, 3)
    model = make_pipeline("decision_tree", X, 3)
    best, details, table = tune_pipeline(model, "decision_tree", X, y, splits, "macro_f1", 3, 2)
    assert len(table) == 2
    assert "classifier__max_depth" in details["best_params"]
    assert len(best.predict(X.iloc[:4])) == 4
