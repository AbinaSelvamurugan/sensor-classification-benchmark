# Methodology and leakage threats

## Data source
UCI HAR Smartphones (2013), DOI 10.24432/C54S4K, CC BY 4.0. 561 engineered time/frequency attributes per 2.56 s, 50%-overlapping accelerometer/gyroscope window; six activities. This code does not convert raw inertial signals into features. `features_info.txt` documents feature families, including means, SDs, magnitudes, correlations, frequencies, and energy. `features.txt` has repeated feature names; stable numeric index prefixes ensure unique DataFrame columns.

## What generalizes?
The provided 7,352-row training and 2,947-row testing partitions consist of nonoverlapping **subjects**, not independent windows: 21 distinct training subjects and 9 different test subjects. Training CV groups all windows by subject, based on `StratifiedGroupKFold`. Do not use a random `StratifiedKFold` for UCI HAR. Group-aware CV makes validation participants unseen within each fold; correlated windows within a person remain dependent, so naive confidence intervals that treat windows as independent are invalid. A subject bootstrap resamples whole people when enough groups are present. Nine test subjects still provide a weak basis for narrow intervals.

## One-way experimental protocol
1. Load original partitions. Validate source, dimensions, labels, group IDs, missing, infinities, duplicates and input types. Do not silently remove rows.
2. Inspect and plot *training only*. Capture class imbalance and obvious quality problems.
3. Build five sklearn pipelines, **preprocessing fit only inside** each training-side CV fold. Impute numeric NA, optional standardization, and one-hot encode categorical values. Infinity is explicitly counted before replacement with NA. Neither group/subject nor target is a predictor.
4. Validate folds for group overlap, missing training classes and missing validation classes. Validation class absences emit warnings with zero-score policy.
5. Compare on 3 training-side folds via macro-F1 (default); save each metric and runtime. Deterministic simplicity tie-break: predeclared tolerance 0.005. Do not select on final test performance.
6. Optionally tune the selected family using training folds only. This is **not nested CV**: the best search score is selected and optimistic. Avoid interpreting it as a new independent estimate. Baseline vs tuned descriptive CV is available.
7. Fit chosen pipeline to all training subjects, persist SHA256 + serialized pipeline, and measure training metrics (diagnostic only).
8. Open original test **once**, predict, compute fixed-label metrics/confusion matrices, and cluster-bootstrap a descriptive uncertainty interval. No parameter changes after evaluation.
9. Build Markdown report from saved output. Figures are generated from the calculated values.

## Dataset variations
For independent CSV rows, stratify the holdout and use shuffled stratified CV. For grouped CSVs, use group holdout and StratifiedGroupKFold. Chronological CSV data are sorted by time before a future holdout and TimeSeriesSplit within earlier training data. Grouped **and** temporal requirements simultaneously cannot currently be represented; redesign the split function for the particular scientific experiment. TimeSeriesSplit does not automatically enforce time gaps to prevent overlapping windows, so avoid use with overlapping recordings unless you implement a purged/embargo split. The CSV classifier cannot know hidden user IDs, disguised leaked targets, changes in sensor calibration, or recording chronology.

## Error analysis and interpretation
The model's confusion matrix uses rows as truth, columns as predictions; derived directional errors and unordered pair counts are saved. Minority recalls and class support matter. Permutation importance (optional, expensive) operates only on a CV validation fold and is conditional on correlated features; forest impurity importance and linear coefficients have their own biases. No association here is causal evidence.

## Reporting notes
Train vs CV vs test performance measure different sampling scenarios. CV score SD across folds is not a significance test or independent-sample confidence interval. Bootstrap over people still cannot fully represent subject/device distribution changes, experimenter effects, or feature extractor quality. The original published test set may have been widely used in other research, creating public-benchmark reuse bias even if this code evaluates it once.
