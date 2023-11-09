# Classification metrics (explicit multiclass labels)

- Accuracy: `# correct / # total`; can hide poor rare-class performance.
- Per-class precision: `TP/(TP+FP)`; `FP` for class C comes from other true rows predicted C.
- Per-class recall: `TP/(TP+FN)`; `FN` for class C comes from row C predicted as other classes.
- Per-class F1: `2PR/(P+R)` if defined; otherwise configured zero.
- Balanced accuracy: average per-class recall of classes observed in the ground truth (sklearn implementation).
- Macro: unweighted mean of per-class scores, giving rare activities equal voting power.
- Weighted: mean per-class scores weighted by *true* class support, potentially dominated by common classes.
- Micro: pools TP/FP/FN across all classes before taking ratios; for single-label multiclass prediction micro F1 = accuracy.

The benchmark always passes the explicit train-dataset class order to `confusion_matrix`, `precision_recall_fscore_support`, and `classification_report`. `zero_division=0` is declared, not silently omitted; the metrics JSON includes warnings for classes with no predictions or no true observations. No binary ROC/AUC is calculated for the six-class UCI HAR baseline, as it would require selecting a meaningful multiclass one-vs-rest policy and probability/score handling. RBF SVM is not probability-calibrated by default.

Raw confusion matrix: truth rows and prediction columns. Normalized confusion matrix: divide each true-class row by its sample count; diagonal is recall. Every off-diagonal cell is a false negative for its **true** class and a false positive for its **predicted** class. The `confusion_pairs.csv` table sums both directions separately from the directional table.

Subject-cluster percentile bootstrap estimates variability from resampling the limited held-out subjects (not overlapping windows). This is an approximate descriptive interval, not a definitive uncertainty statement for deployment, nor a significance test for model comparisons. Classification metric computation is deterministic given y_true/y_pred and label order.
