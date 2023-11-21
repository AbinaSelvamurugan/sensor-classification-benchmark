# Sensor Data Classification Benchmark — empirical evaluation

## 1. Dataset and task

Dataset: **UCI HAR Smartphones**. Source: https://doi.org/10.24432/C54S4K; license: CC BY 4.0.
This experiment used **561 feature columns**. For UCI HAR these are the publisher's **precomputed time- and frequency-domain sensor features**, not raw measurements. Labels: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.

## 2. Training-only data quality

7352 training windows, 561 predictors, 6 classes; 0 missing cells, 0 infinite cells, 0 duplicate predictor rows. Training memory 32,995,908 bytes. Class counts: {'LAYING': 1407, 'SITTING': 1286, 'STANDING': 1374, 'WALKING': 1226, 'WALKING_DOWNSTAIRS': 986, 'WALKING_UPSTAIRS': 1073}. Concerns: none flagged. Duplicate windows were retained. Nonfinite inputs were converted to NaN only after counting them and imputed inside pipelines.

Training-only EDA: `figures/train_class_distribution.png`, `figures/feature_distributions.png`, `figures/feature_correlations.png`, `figures/feature_relationships.png`, and `figures/outlier_inspection.png`. Correlation is not causal evidence or an importance ranking; large values and outliers are descriptive and are not automatically errors.

## 3. Holdout and cross-validation

Strategy: **official_subject_holdout__stratified_group_cv**. Training rows: 7352; **final test rows: 2947**. Train subjects/groups: 21; test subjects/groups: 9. Train/test class counts: {'LAYING': 1407, 'SITTING': 1286, 'STANDING': 1374, 'WALKING': 1226, 'WALKING_DOWNSTAIRS': 986, 'WALKING_UPSTAIRS': 1073} / {'LAYING': 537, 'SITTING': 491, 'STANDING': 532, 'WALKING': 496, 'WALKING_DOWNSTAIRS': 420, 'WALKING_UPSTAIRS': 471}.

CV was restricted to the training portion using 3 splits, seed 42. Group membership is kept disjoint for grouped datasets. Fold-level train/validation class counts and recalls are saved in `cv_class_diagnostics.csv`. An already published holdout is never randomly redivided. Overlapping sensor windows and repeated observations may remain correlated within each subject; metrics estimate transfer to held-out subjects, not new sensors or environments.

## 4. Models and preprocessing

Candidates: dummy, logistic_regression, decision_tree, random_forest, rbf_svm. Dummy uses the majority-class strategy as a floor. Logistic regression and RBF SVM use training-fold scaling; tree/forest do not. Numeric missing values use train-fold median (or configured mean); categorical columns use train-fold mode and one-hot encoding, ignoring unseen levels at inference. Identifiers, subjects, timestamps, and targets are not features. Pipelines are cloned and refit within every CV fold. No synthetic oversampling is used.

## 5. Metrics and multiclass interpretation

Primary: **macro_f1**. Accuracy is correct predictions / total. Balanced accuracy is the mean true-class recall. Macro metrics weight each class equally; weighted metrics weight by test support; micro metrics pool all one-vs-rest outcomes (in single-label multiclass data, micro F1 equals accuracy). Precision is fraction of predictions for a class that are correct; recall is fraction of actual class members identified. Undefined precision and recall are explicitly set to zero (`zero_division=0`) and flagged when an output class receives no predictions. This makes rare class failures visible.

## 6. Fold-by-fold model comparison

| model | macro_f1_mean | macro_f1_std | accuracy_mean | balanced_accuracy_mean | fit_seconds_mean |
| --- | --- | --- | --- | --- | --- |
| dummy | 0.0522 | 0.0008 | 0.1859 | 0.1667 | 0.2173 |
| logistic_regression | 0.9209 | 0.0029 | 0.9225 | 0.9204 | 0.6222 |
| decision_tree | 0.8278 | 0.0122 | 0.8355 | 0.8284 | 2.2954 |
| random_forest | 0.8984 | 0.0155 | 0.9029 | 0.8991 | 6.2200 |
| rbf_svm | 0.9161 | 0.0099 | 0.9200 | 0.9176 | 0.8676 |

Fold-level scores and fold-to-fold variation: `cv_folds.csv`. CV fold standard deviation describes differences across folds, **not a confidence interval** or independent replication; training-window overlap reduces independence.

## 7. Model selection and tuning

Selected **logistic_regression** from training CV using macro_f1 with tolerance 0.005; within tolerance: logistic_regression, rbf_svm. Rule: Select simplest predefined estimator within tolerance of top training-only CV mean. Untuned family training CV macro_f1: **0.9209**. Hyperparameter search: Not performed (flag not passed or dummy selected). If search was performed, its best score is selection-optimistic; there was no nested CV. Test data were not involved in model selection.

## 8. One-time held-out test evaluation

**Accuracy: 0.9545**; **balanced accuracy: 0.9532**; macro precision: 0.9575; macro recall: 0.9532; **macro F1: 0.9544**; weighted precision: 0.9563; weighted recall: 0.9545; weighted F1: 0.9545. Micro F1: 0.9545.

Training macro F1 (full train fit): 0.9967. CV mean macro_f1: 0.9209; test macro_f1: 0.9544. Test-minus-CV metric gap: +0.0335. Differences mix subject variation and selection variability; not a calibrated gap estimate. Final full-training fit: 0.834 s; test prediction: 0.025 s.

| class | f1 | precision | recall | support |
| --- | --- | --- | --- | --- |
| LAYING | 0.9944 | 0.9981 | 0.9907 | 537 |
| SITTING | 0.9198 | 0.9685 | 0.8758 | 491 |
| STANDING | 0.9291 | 0.8885 | 0.9737 | 532 |
| WALKING | 0.9676 | 0.9426 | 0.9940 | 496 |
| WALKING_DOWNSTAIRS | 0.9633 | 0.9899 | 0.9381 | 420 |
| WALKING_UPSTAIRS | 0.9520 | 0.9571 | 0.9469 | 471 |

Metric warnings: none.

## 9. Confusion matrix and error patterns

True classes are **rows** and predicted classes are **columns**. Diagonal cells count correct predictions; off-diagonal cell (i, j) counts samples truly in class i predicted as j (false negatives for i and false positives for j). Row-normalized cells quantify conditional outcomes, including each class's recall along the diagonal. Figures: `figures/confusion_raw.png` and `figures/confusion_normalized.png`. Complete matrix is in `final_test.json`.

Most confused *unordered* class pairs, summing errors in both directions:

| class_a | class_b | total_errors | a_to_b | b_to_a |
| --- | --- | --- | --- | --- |
| SITTING | STANDING | 72 | 58 | 14 |
| WALKING | WALKING_UPSTAIRS | 24 | 0 | 24 |
| WALKING_UPSTAIRS | WALKING_DOWNSTAIRS | 19 | 1 | 18 |
| WALKING | WALKING_DOWNSTAIRS | 9 | 3 | 6 |
| STANDING | LAYING | 5 | 0 | 5 |
| WALKING_UPSTAIRS | SITTING | 2 | 0 | 2 |
| WALKING_DOWNSTAIRS | STANDING | 2 | 2 | 0 |
| SITTING | LAYING | 1 | 1 | 0 |

Detailed directed errors: `confusion_directions.csv`. A high overall accuracy does not remove minority-class risk. Compare individual support and recall above before deployment.

## 10. Reliability and interpretability

Test macro_f1 cluster-bootstrap 95% percentile interval: [0.9247, 0.9767] using 9 complete subjects and 300 resamples. No significance claim.

Training-fold importance, if supported: `feature_importance.csv` (permutation on one training-held-out CV fold) and `native_importance.csv` (coefficients or tree impurity importances). Permutation ranks can be unstable with correlated inputs; impurity rankings favor some split patterns and are **not causal explanations**. Importances were not calculated from final test labels.

## 11. Limitations and improvements

Subject count is relatively small; a single train/test partition cannot establish a universal ranking. UCI HAR windows overlap by 50% within subjects and are drawn from one smartphone model and recording protocol. Subject-level separation reduces leakage but does not establish device, session, or deployment-shift robustness. Fold metrics share training subjects; do not claim significance from a naive t-test. All feature engineering is publisher-provided, so this does not benchmark end-to-end raw-signal feature extraction. Benchmark further with external subjects/devices, nested group CV for performance estimation after tuning, explicit calibration and cost-sensitive metrics as deployment needs require. Do not retune after looking at this final test result; use new external test data for further iteration.

## 12. Reproducibility and provenance

Saved files: `configuration.json`, `dataset_manifest.json` (SHA256), `environment.json`, `selection.json`, `cv_folds.csv`, `cv_summary.csv`, `cv_class_diagnostics.csv`, `final_test.json`, `selected_model.joblib`, `selected_model.sha256`, and `figures/`. Serialized scikit-learn models are pickle-based: **load only artifacts you trust**, via explicit opt-in and verified SHA256. Program license MIT; UCI HAR dataset CC BY 4.0 with attribution required.
