"""Reproducible benchmark stage runner and optional prediction utility."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_classifier.analysis.data_quality import quality_report
from sensor_classifier.analysis.exploratory import explore
from sensor_classifier.config import load_config
from sensor_classifier.data.loading import Dataset, load_dataset
from sensor_classifier.data.validation import replace_inf_with_nan
from sensor_classifier.evaluation.confusion_matrix import matrix_pair
from sensor_classifier.evaluation.cross_validation import evaluate_cv, make_splits, summarize_folds
from sensor_classifier.evaluation.error_analysis import confusion_patterns
from sensor_classifier.evaluation.metrics import classification_metrics, cluster_bootstrap
from sensor_classifier.evaluation.model_selection import select_model
from sensor_classifier.models.registry import MODEL_NAMES, make_pipeline
from sensor_classifier.models.serialization import load_model, save_model
from sensor_classifier.models.training import explain_on_cv_validation, fit_selected, model_native_importance
from sensor_classifier.models.tuning import tune_pipeline
from sensor_classifier.reporting.plots import comparison_figure, confusion_figures, importance_figure
from sensor_classifier.reporting.report import create_report
from sensor_classifier.utils.logging import configure_logging
from sensor_classifier.utils.reproducibility import runtime_info, seed_everything, write_json

LOG = logging.getLogger(__name__)


def _paths(root: Path, run: str) -> Path:
    if not run or not all(ch.isalnum() or ch in "_-" for ch in run):
        raise ValueError("--run-name must contain letters, numbers, underscores or hyphens")
    return root / "results" / run


def _get_data(settings) -> Dataset:
    ds = load_dataset(settings)
    ds.X_train = replace_inf_with_nan(ds.X_train)
    ds.X_test = replace_inf_with_nan(ds.X_test)
    return ds


def inspect(settings, folder: Path) -> None:
    original = load_dataset(settings)
    ds = original
    quality = quality_report(original.X_train, original.y_train, original.labels)
    ds.X_train = replace_inf_with_nan(ds.X_train)
    write_json(folder / "data_quality.json", quality)
    write_json(folder / "dataset_manifest.json", ds.manifest)
    explore(ds.X_train, ds.y_train, folder / "figures")
    LOG.info("Training-only EDA created at %s", folder)


def _manifest_check(folder: Path, ds: Dataset, settings) -> None:
    manifest_file = folder / "dataset_manifest.json"
    if not manifest_file.is_file() or json.loads(manifest_file.read_text()) != ds.manifest:
        raise ValueError("Dataset manifest differs from training run: refusing evaluation")
    config = json.loads((folder / "configuration.json").read_text())
    if config != settings.to_dict():
        raise ValueError("Configuration changed since training: refusing evaluation")


def train(settings, folder: Path, tune: bool = False, permutation: bool = False) -> None:
    if (folder / "final_test.json").exists():
        raise FileExistsError("This run already has final test results; use a new --run-name")
    ds = _get_data(settings)
    # Quality is calculated from original input so infinity counts are not lost.
    original_ds = load_dataset(settings)
    quality = quality_report(original_ds.X_train, original_ds.y_train, ds.labels)
    write_json(folder / "data_quality.json", quality)
    write_json(folder / "configuration.json", settings.to_dict())
    write_json(folder / "environment.json", runtime_info())
    write_json(folder / "dataset_manifest.json", ds.manifest)
    explore(ds.X_train, ds.y_train, folder / "figures")
    cv = make_splits(ds.X_train, ds.y_train, ds.groups_train, ds.strategy,
                     int(settings.evaluation["folds"]), settings.seed)
    all_folds, all_diagnostics = [], []
    for name in settings.models["models"]:
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown model in config: {name}")
        model = make_pipeline(name, ds.X_train, settings.seed, {**settings.preprocessing, "n_jobs": settings.evaluation.get("n_jobs", 1)})
        fold_metrics, diagnostic = evaluate_cv(model, ds.X_train, ds.y_train,
                                               cv, ds.labels, name)
        all_folds.append(fold_metrics)
        all_diagnostics.append(diagnostic)
    folds = pd.concat(all_folds, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    summary = summarize_folds(folds)
    folds.to_csv(folder / "cv_folds.csv", index=False)
    diagnostics.to_csv(folder / "cv_class_diagnostics.csv", index=False)
    summary.to_csv(folder / "cv_summary.csv", index=False)
    metric = settings.evaluation["primary_metric"]
    selection = select_model(summary, metric, float(settings.models["selection_tolerance"]))
    winner = selection["selected_model"]
    model = make_pipeline(winner, ds.X_train, settings.seed, {**settings.preprocessing, "n_jobs": settings.evaluation.get("n_jobs", 1)})
    if tune and winner != "dummy":
        model, details, table = tune_pipeline(model, winner, ds.X_train, ds.y_train, cv,
                                               metric, settings.seed, int(settings.models["search_iterations"]),
                                               int(settings.evaluation.get("n_jobs", 1)))
        table.to_csv(folder / "tuning_search.csv", index=False)
        selection["tuning"] = details
        # Training-only comparison with selected untuned model (same folds reused;
        # clearly flagged as selection-biased, NOT a nested unbiased estimate).
        tuned_fold, tuned_diag = evaluate_cv(model, ds.X_train, ds.y_train, cv, ds.labels,
                                            winner + "__tuned_non_nested")
        tuned_fold.to_csv(folder / "tuned_folds_descriptive.csv", index=False)
        selection["tuning"]["rechecked_fold_mean"] = float(tuned_fold[metric].mean())
    else:
        selection["tuning"] = "Not performed (flag not passed or dummy selected)"
    # Permutation is computed on one training-only CV validation fold, never final test.
    if permutation:
        tr, va = cv[0]
        importance = explain_on_cv_validation(model, ds.X_train, ds.y_train, tr, va,
                                               settings.seed, metric)
        importance.to_csv(folder / "feature_importance.csv", index=False)
        importance_figure(importance, folder / "figures", "Permutation importance on train CV validation fold")
    model, stats = fit_selected(model, ds.X_train, ds.y_train, ds.labels)
    native = model_native_importance(model)
    native.to_csv(folder / "native_importance.csv", index=False)
    if not permutation and not native.empty:
        importance_figure(native, folder / "figures", "Native model importances (not causal)")
    selection.update({"feature_names": ds.feature_names, "labels": ds.labels,
                      "training_metrics": stats["train_metrics"],
                      "full_train_fit_seconds": stats["fit_seconds"],
                      "model_params": {"estimator": repr(model.named_steps["classifier"]),
                                       "preprocessor": repr(model.named_steps["preprocess"])},
                      "model_sha256": save_model(folder / "selected_model.joblib", model)})
    write_json(folder / "selection.json", selection)
    comparison_figure(summary, metric, folder / "figures")
    LOG.info("Selected from CV: %s; files saved to %s", winner, folder)


def evaluate(settings, folder: Path, trust_model: bool) -> None:
    if (folder / "final_test.json").exists():
        raise FileExistsError("Final test already consumed for this run. Refusing repeated evaluation")
    ds = _get_data(settings)
    _manifest_check(folder, ds, settings)
    selection = json.loads((folder / "selection.json").read_text())
    if ds.feature_names != selection["feature_names"] or ds.labels != selection["labels"]:
        raise ValueError("Mismatch between test feature/label schema and training")
    model_file = folder / "selected_model.joblib"
    if selection["model_sha256"] != model_file.with_suffix(".sha256").read_text().strip():
        raise ValueError("Selection/model checksum mismatch")
    model = load_model(model_file, trust=trust_model)
    start = time.perf_counter()
    predictions = model.predict(ds.X_test)
    predict_seconds = time.perf_counter() - start
    metrics = classification_metrics(ds.y_test, predictions, ds.labels)
    raw, _ = matrix_pair(ds.y_test, predictions, ds.labels)
    confusion_figures(raw, ds.labels, folder / "figures")
    directed, paired = confusion_patterns(raw, ds.labels)
    directed.to_csv(folder / "confusion_directions.csv", index=False)
    paired.to_csv(folder / "confusion_pairs.csv", index=False)
    out = {"model": selection["selected_model"], "labels": ds.labels,
           "feature_names": ds.feature_names, "metrics": metrics,
           "split_strategy": ds.strategy, "test_rows": len(ds.y_test),
           "test_class_counts": {str(k): int(v) for k, v in ds.y_test.value_counts().sort_index().items()},
           "train_groups": int(ds.groups_train.nunique()) if ds.groups_train is not None else None,
           "test_groups": int(ds.groups_test.nunique()) if ds.groups_test is not None else None,
           "fit_seconds": selection["full_train_fit_seconds"],
           "predict_seconds": predict_seconds,
           "training_metrics": selection["training_metrics"],
           "bootstrap": cluster_bootstrap(ds.y_test, predictions, ds.groups_test, ds.labels,
                                          int(settings.evaluation["bootstrap_replicates"]), settings.seed,
                                          settings.evaluation["primary_metric"])}
    write_json(folder / "final_test.json", out)
    LOG.info("Held-out test evaluation completed ONCE, %s %.4f", settings.evaluation["primary_metric"],
             metrics[settings.evaluation["primary_metric"]])


def predict(folder: Path, input_file: Path, output: Path, trust_model: bool) -> None:
    selection = json.loads((folder / "selection.json").read_text())
    frame = pd.read_csv(input_file)
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    configured_target = config["dataset"].get("target_column", "activity")
    target_col = configured_target if configured_target in frame else None
    if frame.columns.duplicated().any():
        raise ValueError("Input has repeated column names")
    required = selection["feature_names"]
    absent = set(required) - set(frame)
    if absent:
        raise ValueError(f"Prediction input missing {len(absent)} features; examples: {sorted(absent)[:5]}")
    model = load_model(folder / "selected_model.joblib", trust=trust_model)
    X = replace_inf_with_nan(frame[required])
    predictions = model.predict(X)
    result = pd.DataFrame({"prediction": predictions})
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
            for idx, label in enumerate(model.classes_):
                result["probability__" + str(label)] = probabilities[:, idx]
        except (AttributeError, NotImplementedError):
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    if target_col:
        result["actual"] = frame[target_col]
        if set(frame[target_col].unique()).issubset(set(selection["labels"])):
            summary = classification_metrics(frame[target_col], predictions, selection["labels"])
            write_json(output.with_suffix(".evaluation.json"), summary)
    result.to_csv(output, index=False)
    LOG.info("Wrote %s rows of predictions to %s", len(result), output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "train", "evaluate", "report", "run-all", "predict"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--run-name", default="benchmark")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--trust-model", action="store_true", help="Explicitly trust locally created joblib artifact")
    parser.add_argument("--input", type=Path, help="Prediction input CSV")
    parser.add_argument("--output", type=Path, help="Prediction output CSV")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    folder = _paths(root, args.run_name)
    configure_logging(folder)
    seed_everything(load_config(root, args.config_dir).seed)
    if args.command == "predict":
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        predict(folder, args.input, args.output, args.trust_model)
        return
    if args.command == "report":
        print(create_report(folder))
        return
    settings = load_config(root, args.config_dir)
    if args.command == "inspect":
        inspect(settings, folder)
    elif args.command == "train":
        train(settings, folder, args.tune, args.permutation_importance)
    elif args.command == "evaluate":
        evaluate(settings, folder, args.trust_model)
    elif args.command == "run-all":
        if (folder / "final_test.json").exists():
            raise FileExistsError("Use a new run name; held-out test already evaluated")
        inspect(settings, folder)
        train(settings, folder, args.tune, args.permutation_importance)
        evaluate(settings, folder, trust_model=True)
        print(create_report(folder))


if __name__ == "__main__":
    main()
