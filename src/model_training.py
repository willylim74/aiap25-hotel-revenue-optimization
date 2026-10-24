"""
Model Training Layer.

Reproduces eda.ipynb Sections 15 (split), 17-18 (baseline registries + evaluation) and
22-24 (hyperparameter tuning + final test-set evaluation), driven entirely by
`config.yaml` rather than hardcoded literals. The only deliberate simplification versus
the notebook: no per-model checkpoint ledger — that existed solely to survive Jupyter
cell timeouts, which a plain script run doesn't need (see the approved project plan).

Entry points, used in this order by main.py:
    1. split_data(df, config)                            -> two independent 70/15/15 splits
    2. build_pipelines(config)                            -> one preprocessing Pipeline per task
    3. fit_and_evaluate_baselines(task, config, ...)      -> EDA Section 17-18
    4. tune_models(task, config, ...)                     -> EDA Section 22
    5. select_and_evaluate_final_model(task, config, ...) -> EDA Section 24 (one-time test eval)
"""
import importlib
import time

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: required before the next import
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from data_preparation import build_preprocessing_pipeline


# =============================================================================
# Helpers: instantiate a sklearn estimator / parse a param_grid from config.yaml
# =============================================================================
def _instantiate(class_path: str, params: dict):
    """`class_path` is a fully-qualified dotted path, e.g. 'sklearn.ensemble.RandomForestClassifier'."""
    module_name, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**(params or {}))


def _parse_param_grid(param_grid: dict) -> dict:
    """A literal list is used as-is (works for both GridSearchCV and HalvingRandomSearchCV).
    A {type: randint|uniform, low, high} dict is parsed into a scipy.stats distribution —
    randint(low, high) is high-exclusive, matching scipy directly; uniform's low/high here is
    the actual sampled interval, converted to scipy's loc/scale form (loc=low, scale=high-low)."""
    parsed = {}
    for key, spec in param_grid.items():
        if isinstance(spec, dict) and "type" in spec:
            if spec["type"] == "randint":
                parsed[key] = randint(spec["low"], spec["high"])
            elif spec["type"] == "uniform":
                parsed[key] = uniform(loc=spec["low"], scale=spec["high"] - spec["low"])
            else:
                raise ValueError(f"Unknown param_grid distribution type: {spec['type']!r}")
        else:
            parsed[key] = spec
    return parsed


def _build_registry(task: str, config: dict):
    """Returns (registry: {name -> fresh estimator instance}, sample_weight_models: set of names
    that need sample_weight passed at fit time instead of class_weight, e.g. Gradient Boosting)."""
    models_cfg = config["models"][task]
    registry = {}
    sample_weight_models = set()
    for name, spec in models_cfg.items():
        registry[name] = _instantiate(spec["class"], spec.get("params", {}))
        if spec.get("use_sample_weight"):
            sample_weight_models.add(name)
    return registry, sample_weight_models


# =============================================================================
# 1. Train / validation / test split (EDA Section 15)
# =============================================================================
def split_data(df: pd.DataFrame, config: dict) -> dict:
    """Two independent 70/15/15 splits — Task 1 stratified on the (imbalanced) outcome target,
    Task 2 unstratified, with zero-ADR bookings excluded before splitting (EDA Section 10.5).
    Returns {'task1': {...}, 'task2': {...}}, each with X_train/X_val/X_test,
    y_train/y_val/y_test, feats."""
    rs = config["random_state"]
    feats_cfg = config["features"]
    cat_feats = feats_cfg["categorical"]
    num_base = feats_cfg["numeric_base"]
    leakage_cols = set(feats_cfg["leakage_cols"])

    split_cfg = config["split"]
    val_test_size = split_cfg["val_ratio"] + split_cfg["test_ratio"]
    test_of_tmp = split_cfg["test_ratio"] / val_test_size

    # ---------------- Task 1: classification ----------------
    feats_t1 = cat_feats + num_base + feats_cfg["task1_extra_numeric"]
    assert set(feats_t1).isdisjoint(leakage_cols), "Leakage column found in Task 1 feature list!"

    target1 = config["target"]["task1"]
    X1, y1 = df[feats_t1], df[target1]

    strat1 = y1 if split_cfg["task1_stratify"] else None
    X1_train, X1_tmp, y1_train, y1_tmp = train_test_split(
        X1, y1, test_size=val_test_size, random_state=rs, stratify=strat1
    )
    strat1_tmp = y1_tmp if split_cfg["task1_stratify"] else None
    X1_val, X1_test, y1_val, y1_test = train_test_split(
        X1_tmp, y1_tmp, test_size=test_of_tmp, random_state=rs, stratify=strat1_tmp
    )

    # ---------------- Task 2: regression ----------------
    target2 = config["target"]["task2"]
    df_task2 = df[df[target2] > 0].copy() if config["task2"]["exclude_zero_adr"] else df.copy()

    feats_t2 = cat_feats + num_base
    task2_leakage = leakage_cols | {target2}
    assert set(feats_t2).isdisjoint(task2_leakage), "Leakage column found in Task 2 feature list!"

    X2, y2 = df_task2[feats_t2], df_task2[target2]

    strat2 = y2 if split_cfg["task2_stratify"] else None
    X2_train, X2_tmp, y2_train, y2_tmp = train_test_split(
        X2, y2, test_size=val_test_size, random_state=rs, stratify=strat2
    )
    X2_val, X2_test, y2_val, y2_test = train_test_split(
        X2_tmp, y2_tmp, test_size=test_of_tmp, random_state=rs
    )

    return {
        "task1": dict(
            X_train=X1_train, X_val=X1_val, X_test=X1_test,
            y_train=y1_train, y_val=y1_val, y_test=y1_test, feats=feats_t1,
        ),
        "task2": dict(
            X_train=X2_train, X_val=X2_val, X_test=X2_test,
            y_train=y2_train, y_val=y2_val, y_test=y2_test, feats=feats_t2,
        ),
    }


# =============================================================================
# 2. Preprocessing pipelines (EDA Section 16.2) — one per task, fit on train only
# =============================================================================
def build_pipelines(config: dict):
    """Task 1's numeric columns include adr_clean (a legitimate feature there); Task 2's don't
    (it's the Task 2 target). Both include the same 11 engineered features and the same
    categorical columns. Returns (pipe1, pipe2), UNFITTED."""
    feats_cfg = config["features"]
    cat_feats = feats_cfg["categorical"]
    num_base = feats_cfg["numeric_base"]
    engineered = feats_cfg["engineered_numeric"]

    num_cols_t1 = num_base + feats_cfg["task1_extra_numeric"] + engineered
    num_cols_t2 = num_base + engineered

    pipe1 = build_preprocessing_pipeline(num_cols_t1, cat_feats, config)
    pipe2 = build_preprocessing_pipeline(num_cols_t2, cat_feats, config)
    return pipe1, pipe2


# =============================================================================
# 3. Evaluation metric suites (EDA Section 18 / Section 22's eval_classifier / eval_regressor)
# =============================================================================
def eval_classifier(model, X, y_true, y_true_bin) -> dict:
    """RidgeClassifier has no predict_proba (only decision_function) — its ROC-AUC/PR-AUC are
    reported honestly as NaN rather than faked from decision_function."""
    pred = model.predict(X)
    row = {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        row["roc_auc_macro"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        row["pr_auc_macro"] = average_precision_score(y_true_bin, proba, average="macro")
    else:
        row["roc_auc_macro"] = np.nan
        row["pr_auc_macro"] = np.nan
    return row


def eval_regressor(model, X, y_true) -> dict:
    pred = model.predict(X)
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "MSE": mean_squared_error(y_true, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
        "R2": r2_score(y_true, pred),
    }


# =============================================================================
# 4. Baseline model fitting + evaluation (EDA Section 17-18)
# =============================================================================
def fit_and_evaluate_baselines(task: str, config: dict, pipe, split: dict):
    """Fits every model in the task's baseline registry once, at config-specified params, on the
    preprocessed training split; evaluates on the preprocessed validation split. Returns
    (fitted_models: {name -> fitted estimator}, results_df, X_train_proc, X_val_proc, X_test_proc).
    The preprocessing pipeline is fit here (on X_train only) and reused by tune_models() /
    select_and_evaluate_final_model() for this task, so it is fit exactly once."""
    X_train, X_val = split["X_train"], split["X_val"]
    y_train, y_val = split["y_train"], split["y_val"]

    X_train_proc = pipe.fit_transform(X_train)
    X_val_proc = pipe.transform(X_val)
    X_test_proc = pipe.transform(split["X_test"])

    registry, sample_weight_models = _build_registry(task, config)

    fitted = {}
    rows = []

    if task == "task1":
        classes = sorted(set(split["y_train"]) | set(split["y_val"]) | set(split["y_test"]))
        y_val_bin = label_binarize(y_val, classes=classes)
        sample_weight = compute_sample_weight("balanced", y_train) if sample_weight_models else None

        for name, model in registry.items():
            if name in sample_weight_models:
                model.fit(X_train_proc, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train_proc, y_train)
            fitted[name] = model
            rows.append({"model": name, **eval_classifier(model, X_val_proc, y_val, y_val_bin)})

        results_df = pd.DataFrame(rows).set_index("model").sort_values("pr_auc_macro", ascending=False)
    else:
        for name, model in registry.items():
            model.fit(X_train_proc, y_train)
            fitted[name] = model
            rows.append({"model": name, **eval_regressor(model, X_val_proc, y_val)})

        results_df = pd.DataFrame(rows).set_index("model").sort_values("R2", ascending=False)

    return fitted, results_df, X_train_proc, X_val_proc, X_test_proc


# =============================================================================
# 5. Hyperparameter tuning (EDA Section 22)
# =============================================================================
def tune_models(task: str, config: dict, X_train_proc, y_train, X_val_proc, y_val):
    """Runs GridSearchCV / HalvingRandomSearchCV per model, per config['tuning'][task] — search
    method and grid are chosen per model empirically (see README), not by a blanket
    linear-vs-tree rule. Task 1 deliberately excludes Ridge and Lasso from tuning (they aren't in
    config['tuning']['task1']['models'] at all). Returns (tuned_models, results_df) where
    results_df is indexed by model name with the full validation metric suite + best_params,
    ranked by the task's evaluation ranking_metric."""
    tuning_cfg = config["tuning"]
    task_tuning_cfg = tuning_cfg[task]
    cv = tuning_cfg["cv"]
    scoring = task_tuning_cfg["scoring"]
    rs = config["random_state"]
    n_train = len(y_train)

    if task == "task1":
        classes = sorted(set(y_train) | set(y_val))
        y_val_bin = label_binarize(y_val, classes=classes)

        def eval_fn(model):
            return eval_classifier(model, X_val_proc, y_val, y_val_bin)

        ranking_metric = config["evaluation"]["task1"]["ranking_metric"]
    else:
        def eval_fn(model):
            return eval_regressor(model, X_val_proc, y_val)

        ranking_metric = config["evaluation"]["task2"]["ranking_metric"]

    tuned_models = {}
    rows = []
    for name, model_tune_cfg in task_tuning_cfg["models"].items():
        base_spec = config["models"][task][name]
        base_params = base_spec.get("params", {}) or {}
        estimator = _instantiate(base_spec["class"], base_params)
        param_grid = _parse_param_grid(model_tune_cfg["param_grid"])

        # Parallelize the search itself (across CV folds / candidates) unless the base
        # estimator already parallelizes its own fit (e.g. Random Forest's n_jobs=-1) —
        # stacking both would oversubscribe the same cores with nested parallelism and
        # tends to be slower, not faster, than picking one level to parallelize.
        search_n_jobs = 1 if "n_jobs" in base_params else -1

        t0 = time.time()
        if model_tune_cfg["search_method"] == "grid":
            search = GridSearchCV(estimator, param_grid, scoring=scoring, cv=cv, n_jobs=search_n_jobs)
        elif model_tune_cfg["search_method"] == "halving":
            search = HalvingRandomSearchCV(
                estimator, param_grid, resource="n_samples",
                max_resources=n_train, min_resources=max(1000, n_train // 30),
                factor=3, cv=cv, scoring=scoring, random_state=rs, n_jobs=search_n_jobs,
                n_candidates=model_tune_cfg.get("n_candidates", "exhaust"),
            )
        else:
            raise ValueError(f"Unknown search_method: {model_tune_cfg['search_method']!r}")

        search.fit(X_train_proc, y_train)
        fit_time = time.time() - t0

        best_model = search.best_estimator_
        tuned_models[name] = best_model
        val_metrics = eval_fn(best_model)
        rows.append({"model": name, **val_metrics, "best_params": str(search.best_params_)})
        print(f"[{task}::{name}] tuned in {fit_time:.1f}s, cv_score={search.best_score_:.4f}, "
              f"best_params={search.best_params_}")

    results_df = pd.DataFrame(rows).set_index("model").sort_values(ranking_metric, ascending=False)
    return tuned_models, results_df


# =============================================================================
# 6. Final model selection + one-time test-set evaluation (EDA Section 24)
# =============================================================================
def select_and_evaluate_final_model(task: str, config: dict, tuned_models: dict,
                                     tuned_results_df, X_test_proc, y_test):
    """Picks the tuned model with the best validation score on the task's ranking metric, then
    evaluates it ONCE on the held-out test split — the only place X_test/y_test are used."""
    ranking_metric = config["evaluation"][task]["ranking_metric"]
    best_name = tuned_results_df[ranking_metric].idxmax()
    best_model = tuned_models[best_name]

    if task == "task1":
        classes = sorted(set(y_test))
        y_test_bin = label_binarize(y_test, classes=classes)
        test_metrics = eval_classifier(best_model, X_test_proc, y_test, y_test_bin)
    else:
        test_metrics = eval_regressor(best_model, X_test_proc, y_test)

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "val_score": tuned_results_df.loc[best_name, ranking_metric],
        "test_metrics": test_metrics,
    }
