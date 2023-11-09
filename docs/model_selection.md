# Selection policy

**Before test evaluation**, score each candidate on the same valid training CV splits using default `macro_f1`. The five candidates use fixed seeds and deliberately bounded complexity. Compute each mean CV score; find its maximum; select the lowest-complexity model with mean within 0.005 (configurable) of maximum. This favors a simpler solution when fold variation and point-estimate differences make ranks unconvincing. The order is dummy, decision tree, logistic regression, random forest, RBF SVM. That order is a pedagogical maintenance preference, not universal truth. Selection is stored in `selection.json`.

When `--tune` is enabled, only the selected *family* is tuned, with RandomizedSearchCV on these same training folds. Save `tuning_search.csv` and an explicitly optimistic best CV score. Rechecking the winning hyperparameters on the same folds is descriptive only, not a fresh independent comparison. A statistically cleaner extension would nest a hyperparameter search inside an outer group CV.

Fitting the final pipeline on all training data permits measuring train performance and its relation to CV and test, but train accuracy alone cannot make a model the winner. The once-only test score reports performance on previously held-out persons, and future changes after viewing the test result demand a new independent test set.
