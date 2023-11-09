# Troubleshooting

**Missing UCI data:** Run `python scripts/download_dataset.py` while online, or manually extract UCI dataset 240 into `data/UCI HAR Dataset`. Original file directory names are case sensitive on Linux.

**Dependencies:** Use Python 3.10+, install `pip install -e '.[dev]'`.

**Missing CSV target/classes:** Provide `target_column`, `group_column` and `split_strategy`. A class absent from a training fold is a hard error; a class absent from a validation fold is warned about; lower the fold count or obtain more groups. Group-aware CV cannot repair a dataset with no class coverage across subjects.

**Unexpected identifier leakage:** Rename or specify `group_column`/`drop_columns` for ID-like columns. Remove target proxies; do not put the target in `X`. Do not remove an informative feature merely to improve the score.

**Serializing/deserializing joblib:** `--trust-model` is intentionally mandatory. A digest detects corruption, not malicious replacement of both model and digest. Never load arbitrary third-party joblib files.

**Cannot re-evaluate same test:** Create a new `--run-name` only for fresh experiments. Repeated external-test tuning invalidates unbiased final evaluation.

**Long runtime:** Five models × 3 CV folds on 561 features can take time. `--tune` adds fits and `--permutation-importance` permutes each feature on one CV validation fold and may be costly. Default avoids permutation. Set `folds: 2` for debugging (not a substitute for full assessment).

**Chronological CSV:** TimeSeriesSplit requires training folds to contain every class; the framework raises if not. Correlated overlapping data may need gaps/purging beyond the included splitting strategies.
