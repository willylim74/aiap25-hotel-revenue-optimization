#!/usr/bin/env python3
"""
Orchestration entry point for the AIAP end-to-end ML pipeline.

Straight-line flow, no hidden control flow:
    load config -> load data/hotel.db -> prepare_data() -> split_data() -> build_pipelines()
    -> baseline fit+eval (both tasks) -> hyperparameter tuning (both tasks)
    -> final model selection + one-time test evaluation (both tasks)
    -> persist final models (joblib) + all results tables/manifest to outputs/

Run from the project root:
    python src/main.py                      # uses src/config.yaml
    python src/main.py --config other.yaml   # overrides the config path

This reproduces eda.ipynb's own modelling results (Sections 13-24) outside a notebook —
see README.md for the mapping between each stage below and its EDA section.
"""
import argparse
import json
import os
import time

import joblib
import pandas as pd
import yaml

from data_preparation import prepare_data
from model_training import (
    build_pipelines,
    fit_and_evaluate_baselines,
    select_and_evaluate_final_model,
    split_data,
    tune_models,
)

TASK_NUM = {"task1": "1", "task2": "2"}
TASK_LABEL = {"task1": "Task 1 (classification — outcome)", "task2": "Task 2 (regression — adr_clean)"}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _out_path(config: dict, key: str, task: str, base_dir: str) -> str:
    template = config["output"][key]
    filename = template.format(task=TASK_NUM[task])
    return os.path.join(base_dir, filename)


def run_task(task: str, config: dict, split: dict, pipe, output_dir: str, models_dir: str) -> dict:
    """Runs baseline fitting, tuning, and final test evaluation for one task. Persists that
    task's results tables and final model, and returns the summary used for the manifest."""
    print(f"\n{'=' * 70}\n{TASK_LABEL[task]}\n{'=' * 70}")

    # ---- Baselines (EDA Section 17-18) ----
    print(f"[{task}] fitting baseline registry...")
    fitted_baselines, baseline_results, X_train_proc, X_val_proc, X_test_proc = \
        fit_and_evaluate_baselines(task, config, pipe, split)
    baseline_results.to_csv(_out_path(config, "baseline_results", task, output_dir))
    print(f"[{task}] baseline results (validation split):\n{baseline_results.round(4)}")

    # ---- Hyperparameter tuning (EDA Section 22) ----
    print(f"\n[{task}] tuning models: {list(config['tuning'][task]['models'].keys())}")
    tuned_models, tuned_results = tune_models(
        task, config, X_train_proc, split["y_train"], X_val_proc, split["y_val"]
    )
    tuned_results.to_csv(_out_path(config, "tuned_results", task, output_dir))
    tuned_results[["best_params"]].to_csv(_out_path(config, "tuned_best_params", task, output_dir))
    print(f"[{task}] tuned results (validation split):\n{tuned_results.drop(columns='best_params').round(4)}")

    # ---- Final model selection + one-time test evaluation (EDA Section 24) ----
    final = select_and_evaluate_final_model(
        task, config, tuned_models, tuned_results, X_test_proc, split["y_test"]
    )
    print(f"\n[{task}] selected model: {final['best_model_name']} "
          f"(validation {config['evaluation'][task]['ranking_metric']} = {final['val_score']:.4f})")
    print(f"[{task}] FINAL TEST-SET result — {final['best_model_name']}:")
    for k, v in final["test_metrics"].items():
        print(f"    {k}: {v:.4f}" if pd.notna(v) else f"    {k}: N/A")

    # ---- Persist the final model + its fitted preprocessing pipeline together ----
    # The pipeline was fit on this task's training split inside fit_and_evaluate_baselines(),
    # and is the exact transform the persisted model expects at inference time.
    model_path = _out_path(config, "final_model", task, models_dir)
    joblib.dump({"model": final["best_model"], "pipeline": pipe, "feats": split["feats"]}, model_path)
    print(f"[{task}] final model + pipeline saved to {model_path}")

    return {
        "task": task,
        "baseline_best": baseline_results.iloc[0].name,
        "tuned_best": final["best_model_name"],
        "validation_score": float(final["val_score"]),
        "ranking_metric": config["evaluation"][task]["ranking_metric"],
        "test_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in final["test_metrics"].items()},
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end AIAP training pipeline.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                         help="Path to config.yaml (default: src/config.yaml)")
    args = parser.parse_args()

    t0 = time.time()
    config = load_config(args.config)

    output_dir = config["data"]["output_dir"]
    models_dir = config["data"]["models_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading and cleaning data from {config['data']['db_path']} (table: {config['data']['table']}) ...")
    df_model = prepare_data(config)
    print(f"Cleaned dataset: {df_model.shape[0]:,} rows, {df_model.shape[1]} columns")

    print("Splitting data (70/15/15, per-task)...")
    splits = split_data(df_model, config)
    for task in ("task1", "task2"):
        s = splits[task]
        print(f"  {task}: train={len(s['X_train']):,}  val={len(s['X_val']):,}  test={len(s['X_test']):,}")

    print("Building preprocessing pipelines...")
    pipe1, pipe2 = build_pipelines(config)
    pipes = {"task1": pipe1, "task2": pipe2}

    manifest = {"tasks": {}}
    for task in ("task1", "task2"):
        manifest["tasks"][task] = run_task(task, config, splits[task], pipes[task], output_dir, models_dir)

    # ---- Closing summary (EDA Section 25) ----
    summary_rows = []
    for task in ("task1", "task2"):
        t = manifest["tasks"][task]
        summary_rows.append({
            "task": TASK_LABEL[task],
            "model": t["tuned_best"],
            "ranking_metric": t["ranking_metric"],
            "validation_score": round(t["validation_score"], 4),
            "test_score": round(t["test_metrics"][t["ranking_metric"]], 4)
            if t["test_metrics"].get(t["ranking_metric"]) is not None else None,
        })
    closing_summary = pd.DataFrame(summary_rows).set_index("task")
    closing_summary.to_csv(os.path.join(output_dir, config["output"]["closing_summary"]))
    print(f"\n{'=' * 70}\nClosing summary\n{'=' * 70}\n{closing_summary}")

    manifest["elapsed_seconds"] = round(time.time() - t0, 1)
    manifest_path = os.path.join(output_dir, config["output"]["manifest"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPipeline complete in {manifest['elapsed_seconds']}s. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
