import numpy as np
import pandas as pd
import pytest

from sensor_classifier.data.validation import check_features, replace_inf_with_nan


def test_missing_infinite_and_duplicates():
    X = pd.DataFrame({"acc": [1, np.inf, np.nan, 1], "gyro": [1, 2, 3, 1]})
    y = pd.Series(["a", "b", "a", "b"])
    with pytest.warns(UserWarning):
        quality = check_features(X, y, ["a", "b"])
    assert quality["missing_cells"] == 1
    assert quality["infinite_cells"] == 1
    assert replace_inf_with_nan(X).isna().sum().sum() == 2


def test_invalid_target_and_identifier():
    with pytest.raises(ValueError):
        check_features(pd.DataFrame({"target": [1, 2]}), pd.Series(["x", "y"]), ["x", "y"])
    with pytest.raises(ValueError):
        check_features(pd.DataFrame({"a": [1, 2]}), pd.Series(["x", None]), ["x", "y"])
