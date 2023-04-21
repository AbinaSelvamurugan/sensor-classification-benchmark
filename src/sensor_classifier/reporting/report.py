"""Write Markdown narrative from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sensor_classifier.reporting.tables import markdown_table


def create_report(folder: Path) -> Path:
    required = ["configuration.json", "dataset_manifest.json", "data_quality.json", "cv_folds.csv",
                "cv_summary.csv", "selection.json", "final_test.json", "confusion_pairs.csv"]
    absent = [x for x in required if not (folder / x).is_file()]
    if absent:
        raise FileNotFoundError(f"Evaluation/report inputs missing: {absent}")
    read = lambda filename: json.loads((folder / filename).read_text(encoding="utf-8"))
    config, manifest, quality, selection, final = [read(n) for n in (
        "configuration.json", "dataset_manifest.json", "data_quality.json", "selection.json", "final_test.json")]
    cv, folds, pairs = [pd.read_csv(folder / name) for name in (
        "cv_summary.csv", "cv_folds.csv", "confusion_pairs.csv")]
    test = final["metrics"]
    selected = selection["selected_model"]
    chosen = cv.loc[cv.model == selected].iloc[0]
    key = config["evaluation"]["primary_metric"]
    bestcv = float(chosen[key + "_mean"])
    actual_test = float(test[key])
    tuned = isinstance(selection.get("tuning"), dict)
    if tuned:
        gap_text = ("The tuned model differs from the untuned family used for primary model selection; "
                    "a directly comparable unbiased CV-to-test gap is not estimated. "
                    f"Non-nested tuned CV recheck: {selection['tuning']['rechecked_fold_mean']:.4f} "
                    "(optimistic due to using searched folds).")
    else:
        gap_text = (f"Test-minus-CV metric gap: {actual_test - bestcv:+.4f}. "
                    "Differences mix subject variation and selection variability; not a calibrated gap estimate.")
    classes = pd.DataFrame([{"class": k, **v} for k, v in test["per_class"].items()])
    balance = quality["label_counts"]
    bootstrap = final["bootstrap"]
    uncertainty = (f"Test {key} cluster-bootstrap 95% percentile interval: "
                   f"[{bootstrap['interval_95'][0]:.4f}, {bootstrap['interval_95'][1]:.4f}] "
                   f"using {bootstrap['units']} complete subjects and "
                   f"{bootstrap['replicates']} resamples. No significance claim."
                   if bootstrap.get("available") else f"No valid group bootstrap: {bootstrap.get('reason')}. "
                   "Window-wise intervals would be invalid for correlated windows.")
    top_errors = markdown_table(pairs, n=8)
    cv_columns = ["model", key + "_mean", key + "_std", "accuracy_mean", "balanced_accuracy_mean",
                  "fit_seconds_mean"]
    docs = f"""# Sensor Data Classification Benchmark — empirical evaluation

## 1. Dataset and task

Dataset: **{manifest['dataset']}**. Source: {manifest.get('doi', manifest.get('dataset'))}; license: {manifest.get('license', 'User-supplied — verify separately')}.
This experiment used **{len(final['feature_names'])} feature columns**. For UCI HAR these are the publisher's **precomputed time- and frequency-domain sensor features**, not raw measurements. Labels: {', '.join(map(str, final['labels']))}.

## 2. Training-only data quality

{quality['rows']} training windows, {quality['features']} predictors, {quality['classes']} classes; {quality['missing_cells']} missing cells, {quality['infinite_cells']} infinite cells, {quality['duplicate_predictor_rows']} duplicate predictor rows. Training memory {quality['memory_bytes']:,} bytes. Class counts: {balance}. Concerns: {quality['concerns'] or 'none flagged'}. Duplicate windows were retained. Nonfinite inputs were converted to NaN only after counting them and imputed inside pipelines.

Training-only EDA: `figures/train_class_distribution.png`, `figures/feature_distributions.png`, `figures/feature_correlations.png`, `figures/feature_relationships.png`, and `figures/outlier_inspection.png`. Correlation is not causal evidence or an importance ranking; large values and outliers are descriptive and are not automatically errors.

## 3. Holdout and cross-validation

Strategy: **{final['split_strategy']}**. Training rows: {quality['rows']}; **final test rows: {final['test_rows']}**. Train subjects/groups: {final['train_groups']}; test subjects/groups: {final['test_groups']}. Train/test class counts: {balance} / {final['test_class_counts']}.

CV was restricted to the training portion using {config['evaluation']['folds']} splits, seed {config['evaluation']['seed']}. Group membership is kept disjoint for grouped datasets. Fold-level train/validation class counts and recalls are saved in `cv_class_diagnostics.csv`. An already published holdout is never randomly redivided. Overlapping sensor windows and repeated observations may remain correlated within each subject; metrics estimate transfer to held-out subjects, not new sensors or environments.

## 4. Models and preprocessing

Candidates: {', '.join(config['models']['models'])}. Dummy uses the majority-class strategy as a floor. Logistic regression and RBF SVM use training-fold scaling; tree/forest do not. Numeric missing values use train-fold median (or configured mean); categorical columns use train-fold mode and one-hot encoding, ignoring unseen levels at inference. Identifiers, subjects, timestamps, and targets are not features. Pipelines are cloned and refit within every CV fold. No synthetic oversampling is used.

## 5. Metrics and multiclass interpretation

Primary: **{key}**. Accuracy is correct predictions / total. Balanced accuracy is the mean true-class recall. Macro metrics weight each class equally; weighted metrics weight by test support; micro metrics pool all one-vs-rest outcomes (in single-label multiclass data, micro F1 equals accuracy). Precision is fraction of predictions for a class that are correct; recall is fraction of actual class members identified. Undefined precision and recall are explicitly set to zero (`zero_division=0`) and flagged when an output class receives no predictions. This makes rare class failures visible.

## 6. Fold-by-fold model comparison

{markdown_table(cv, cv_columns, n=len(cv))}

Fold-level scores and fold-to-fold variation: `cv_folds.csv`. CV fold standard deviation describes differences across folds, **not a confidence interval** or independent replication; training-window overlap reduces independence.

## 7. Model selection and tuning

Selected **{selected}** from training CV using {selection['metric']} with tolerance {selection['selection_tolerance']}; within tolerance: {', '.join(selection['within_tolerance'])}. Rule: {selection['rationale']}. Untuned family training CV {key}: **{bestcv:.4f}**. Hyperparameter search: {selection.get('tuning', 'Not run')}. If search was performed, its best score is selection-optimistic; there was no nested CV. Test data were not involved in model selection.

## 8. One-time held-out test evaluation

**Accuracy: {test['accuracy']:.4f}**; **balanced accuracy: {test['balanced_accuracy']:.4f}**; macro precision: {test['macro_precision']:.4f}; macro recall: {test['macro_recall']:.4f}; **macro F1: {test['macro_f1']:.4f}**; weighted precision: {test['weighted_precision']:.4f}; weighted recall: {test['weighted_recall']:.4f}; weighted F1: {test['weighted_f1']:.4f}. Micro F1: {test['micro_f1']:.4f}.

Training macro F1 (full train fit): {final['training_metrics']['macro_f1']:.4f}. CV mean {key}: {bestcv:.4f}; test {key}: {actual_test:.4f}. {gap_text} Final full-training fit: {final['fit_seconds']:.3f} s; test prediction: {final['predict_seconds']:.3f} s.

{markdown_table(classes)}

Metric warnings: {test['warnings'] or 'none'}.

## 9. Confusion matrix and error patterns

True classes are **rows** and predicted classes are **columns**. Diagonal cells count correct predictions; off-diagonal cell (i, j) counts samples truly in class i predicted as j (false negatives for i and false positives for j). Row-normalized cells quantify conditional outcomes, including each class's recall along the diagonal. Figures: `figures/confusion_raw.png` and `figures/confusion_normalized.png`. Complete matrix is in `final_test.json`.

Most confused *unordered* class pairs, summing errors in both directions:

{top_errors}

Detailed directed errors: `confusion_directions.csv`. A high overall accuracy does not remove minority-class risk. Compare individual support and recall above before deployment.

## 10. Reliability and interpretability

{uncertainty}

Training-fold importance, if supported: `feature_importance.csv` (permutation on one training-held-out CV fold) and `native_importance.csv` (coefficients or tree impurity importances). Permutation ranks can be unstable with correlated inputs; impurity rankings favor some split patterns and are **not causal explanations**. Importances were not calculated from final test labels.

## 11. Limitations and improvements

Subject count is relatively small; a single train/test partition cannot establish a universal ranking. UCI HAR windows overlap by 50% within subjects and are drawn from one smartphone model and recording protocol. Subject-level separation reduces leakage but does not establish device, session, or deployment-shift robustness. Fold metrics share training subjects; do not claim significance from a naive t-test. All feature engineering is publisher-provided, so this does not benchmark end-to-end raw-signal feature extraction. Benchmark further with external subjects/devices, nested group CV for performance estimation after tuning, explicit calibration and cost-sensitive metrics as deployment needs require. Do not retune after looking at this final test result; use new external test data for further iteration.

## 12. Reproducibility and provenance

Saved files: `configuration.json`, `dataset_manifest.json` (SHA256), `environment.json`, `selection.json`, `cv_folds.csv`, `cv_summary.csv`, `cv_class_diagnostics.csv`, `final_test.json`, `selected_model.joblib`, `selected_model.sha256`, and `figures/`. Serialized scikit-learn models are pickle-based: **load only artifacts you trust**, via explicit opt-in and verified SHA256. Program license MIT; UCI HAR dataset CC BY 4.0 with attribution required.
"""
    path = folder / "EVALUATION_REPORT.md"
    path.write_text(docs, encoding="utf-8")
    return path
