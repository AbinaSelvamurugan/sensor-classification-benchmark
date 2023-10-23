import numpy as np
import pandas as pd

from sensor_classifier.data.preprocessing import build_preprocessing


def test_fold_local_imputation_and_scaling():
    # Validation extreme must not affect training preprocessing statistics.
    X = pd.DataFrame({"acc": [0., 2., np.nan, 10000.], "state": ["a", "b", "a", "c"]})
    transformer = build_preprocessing(X, scale=True)
    transformer.fit(X.iloc[:3])
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    assert abs(scaler.mean_[0] - 1.0) < 1e-8
    assert transformer.transform(X.iloc[3:]).shape[0] == 1
    assert np.isfinite(transformer.transform(X.iloc[3:])).all()
