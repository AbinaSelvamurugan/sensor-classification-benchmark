# Sensor Data Classification Benchmark
### A Reproducible Machine Learning Evaluation Framework

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

This benchmark evaluates multiclass classification on published engineered inertial-sensor features. The primary dataset is UCI HAR Using Smartphones (UCI dataset 240), which ships derived feature vectors rather than raw accelerometer signals. The focus is on honest estimates of how well a model generalizes to unseen people, rather than on maximizing reported accuracy.

**Data citation:** Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K). Dataset license CC BY 4.0, separate from this codebase's MIT license.

## Dataset rationale and feature meanings

30 participants wore a Samsung Galaxy S II at the waist. Six classes: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`. Triaxial accelerometer and gyroscope readings were sampled at 50 Hz, filtered, segmented into 2.56-second windows with 50% overlap, and converted by the publishers into 561 time- and frequency-domain feature values per window. See the downloaded dataset's `features_info.txt` for each feature family: body vs gravity acceleration; time vs frequency (`t`/`f`); axis (`X`, `Y`, `Z`); magnitude, mean, standard deviation, median absolute deviation, entropy, correlations, signal energy, and frequency measures. Feature names are prefixed `f001__`, `f002__`, etc. because the original dataset's 561 labels are not unique. Original subject IDs and class labels are never predictor columns.

UCI's original published train/test separation by subjects is preserved: 21 train subjects and 9 test subjects (7,352 / 2,947 windows). The training-side CV uses `StratifiedGroupKFold`, which respects within-person correlation and window overlap; a few folds can still lack rare classes, and the code warns when that happens. A random split of windows would leak highly related observations from the same participant.

Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

### Quickstart (bash / macOS / Linux / Windows PowerShell with adjusted activation)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/download_dataset.py
python scripts/inspect_dataset.py --run-name example
python scripts/train_models.py --run-name example
python scripts/evaluate_models.py --run-name example --trust-model
python scripts/generate_report.py --run-name example
```

`--root /path/to/repository` works if running from another directory. Windows activation: `.venv\Scripts\Activate.ps1`. Download requires the public UCI site; if unavailable, manually download UCI dataset 240's archive and extract `UCI HAR Dataset/` under `data/`. No account or API key is required. The downloader prints an archive digest, and run manifests include SHA256 digests of the feature files.

**Alternate single command** (fresh run only):

```bash
sensor-benchmark run-all --run-name full-benchmark
```

### Optional tuning and interpretability

```bash
sensor-benchmark train --run-name tuned --tune
sensor-benchmark evaluate --run-name tuned --trust-model
sensor-benchmark report --run-name tuned

# More expensive: permute *each input feature* on a train-only CV validation fold.
sensor-benchmark train --run-name importance --permutation-importance
```

`--tune` runs a small `RandomizedSearchCV` restricted to training subjects. The baseline model decision is based on untuned CV means and an advance-declared simplicity tolerance. Search's `best_score_` is optimistic and not a nested-CV performance estimate, so the report qualifies it. No more than one held-out evaluation is permitted per run-name; experiments must use new names, and iterating on viewed test scores compromises its value as a clean final evaluation.

### Model selection and metrics

* Majority-class `DummyClassifier` establishes a floor but has limited rare-class recall.
* Standardized regularized `LogisticRegression` (linear reference).
* Constrained `DecisionTreeClassifier` (inspectable nonlinear baseline).
* `RandomForestClassifier` (nonlinear ensemble).
* Scaled RBF `SVC` (nonlinear margin-based model; no calibrated probabilities by default).

Primary scoring is mean CV macro-F1 (equal weight per activity). Choose the least complex model within the configurable `selection_tolerance` of the maximum CV macro-F1 mean. Record per-fold accuracy, balanced accuracy, macro/weighted precision/recall/F1, fitting and prediction times, class counts, missing-class warnings, and training performance. Then refit the selected entire pipeline on training subjects and evaluate once on the reserved original test subjects. The final report documents false positives and negatives, directional errors and paired confusions, class support, and subject-cluster bootstrap intervals. Fold SD is descriptive, not an independent-observation confidence interval.

See `docs/methodology.md`, `docs/metrics.md`, and `docs/model_selection.md`.

### CSV interface

Edit `configs/dataset.yaml`:

```yaml
kind: csv
path: data/my_sensor_features.csv
target_column: activity
group_column: subject_id
split_strategy: group
holdout_fraction: 0.25
drop_columns: [record_id]
```

CSV permits numeric and categorical predictors, fold-local imputers, and one-hot encoding. Identifiers, the target, group, and time column are excluded explicitly; suspected undeclared IDs and exact target copies cause errors. Alternatives are `split_strategy: stratified` for independent examples (remove `group_column`) or `chronological` with `time_column` (remove `group_column`), where temporal sorting precedes a future test block and `TimeSeriesSplit` is used inside training. Beware of near-duplicate correlated records and class disappearance. The loader does not infer recording/session boundaries, causality, or sampling independence on arbitrary CSVs; these must be supplied by the dataset owner. For any grouped CSV dataset, a group holdout is mandatory.

### Optional prediction after training

```bash
sensor-benchmark predict --run-name full-benchmark --input data/new_features.csv --output results/new_predictions.csv --trust-model
```

CSV must contain columns identical to the prefixed UCI input feature names (or the configured custom CSV columns); extra columns are ignored. Output contains predicted labels and probabilities where the estimator supports them. If an `activity` label column exists, its predictions are additionally evaluated in a separate `.evaluation.json` file. Avoid evaluating the same held-out dataset repeatedly for model selection. `joblib` uses pickle serialization: pass `--trust-model` only for trusted local files, after SHA256 verification. Hash matching detects accidental changes, not malicious replacement of both file and digest.

### Automated tests and quality

```bash
pytest -q
ruff check .
ruff format --check .
python -m compileall -q src scripts
```

The tests run on small synthetic fixtures and do not require the UCI download. They cover input validation, split isolation, class imbalance, fold-local imputation, all five estimators, CV, tuning, metrics, binary/multiclass ordering, figures/report generation, serialized pipelines and file hashes. See `tests/`.

### Results layout

`results/<run-name>/` contains `benchmark.log`, `dataset_manifest.json`, `configuration.json`, `environment.json`, `data_quality.json`, `cv_folds.csv`, `cv_class_diagnostics.csv`, `cv_summary.csv`, `selection.json`, `native_importance.csv`, `selected_model.joblib`, `selected_model.sha256`, `final_test.json`, `confusion_directions.csv`, `confusion_pairs.csv`, `figures/`, and `EVALUATION_REPORT.md`. Optional search files (`tuning_search.csv`, `tuned_folds_descriptive.csv`) and permutation files are generated only when opted in. All claims in the report are generated from these run artifacts.

### Maintainer notes

All seeds are centralized in `configs/evaluation.yaml`; fixed seeds do not guarantee bitwise identical results across Python/platform/library versions. Use Python 3.10+ and pinned version ranges in `pyproject.toml`; archive your environment via `python -m pip freeze` for exact re-runs. No dataset is silently deleted. Inf values are counted and converted to missing, then imputed inside folds; duplicates are recorded and retained; target labels are never imputed. Every experimental run includes configuration and data hashes.

For contributions, consult `CONTRIBUTING.md`. For setup issues consult `docs/troubleshooting.md`.

## Continuous integration

The repository ships a Python test workflow at `.github/workflows/ci.yml`. Run the download and benchmark commands above to generate results.
