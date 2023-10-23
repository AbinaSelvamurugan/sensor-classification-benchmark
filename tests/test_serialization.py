import pytest

from sensor_classifier.models.registry import make_pipeline
from sensor_classifier.models.serialization import load_model, save_model


def test_save_load_and_explicit_trust(tmp_path, synthetic):
    X, y, _ = synthetic
    model = make_pipeline("dummy", X, 5).fit(X, y)
    path = tmp_path / "model.joblib"
    digest = save_model(path, model)
    assert len(digest) == 64
    with pytest.raises(PermissionError):
        load_model(path)
    assert len(load_model(path, trust=True).predict(X.iloc[:2])) == 2
    path.with_suffix(".sha256").write_text("bad")
    with pytest.raises(ValueError):
        load_model(path, trust=True)
