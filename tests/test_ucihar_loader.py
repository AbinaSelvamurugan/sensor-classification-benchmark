"""UCI HAR ingestion checks against small fixture files."""
import numpy as np
import pandas as pd
import pytest

from sensor_classifier.data.loading import load_ucihar


def test_ucihar_feature_identifiers_and_subjects(tmp_path):
    folder = tmp_path / "UCI HAR Dataset"
    folder.mkdir()
    (folder / "features.txt").write_text("".join(
        f"{i} tBodyAcc-mean()-X\n" for i in range(1, 562)))
    (folder / "activity_labels.txt").write_text("".join(
        f"{i} ACTIVITY_{i}\n" for i in range(1, 7)))
    for split, subjects in (("train", [1] * 12), ("test", [2] * 6)):
        directory = folder / split
        directory.mkdir()
        np.savetxt(directory / f"X_{split}.txt", np.ones((len(subjects), 561)), fmt="%.2f")
        np.savetxt(directory / f"y_{split}.txt", np.tile(np.arange(1, 7), len(subjects) // 6), fmt="%d")
        np.savetxt(directory / f"subject_{split}.txt", subjects, fmt="%d")
    ds = load_ucihar(folder)
    assert ds.X_train.shape == (12, 561)
    assert ds.feature_names[0] == "f001__tBodyAcc-mean()-X"
    assert ds.feature_names[-1].startswith("f561__")
    assert ds.X_train.columns.is_unique
    assert not set(ds.groups_train) & set(ds.groups_test)
    (folder / "test" / "subject_test.txt").write_text("1\n" * 6)
    with pytest.raises(ValueError, match="subject split"):
        load_ucihar(folder)
