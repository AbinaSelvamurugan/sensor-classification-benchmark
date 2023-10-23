import pytest

from sensor_classifier.models.registry import MODEL_NAMES, make_pipeline


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_pipeline_fits(synthetic, name):
    X, y, _ = synthetic
    model = make_pipeline(name, X, 42).fit(X, y)
    assert set(model.predict(X[:2])).issubset(set(y))


def test_invalid_model(synthetic):
    X, _, _ = synthetic
    with pytest.raises(ValueError):
        make_pipeline("magic", X, 42)
