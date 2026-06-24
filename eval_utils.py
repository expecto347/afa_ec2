"""
Shared utilities for saving evaluation results across all evaluate_ec2_*.py scripts.

Two-level persistence:
  1. Per-run curve:  <run_dir>/eval_results.csv   (budget × metrics, for plotting)
  2. Global summary: results_log.csv               (one row per run, for comparison)

Typical checkpoint path structure:
    logs/<run_name>/ec2_train/checkpoints/<ckpt>.ckpt
So Path(ckpt).parents[2] == logs/<run_name>/
"""

import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def run_dir_from_checkpoint(checkpoint_path: str) -> str:
    return str(Path(checkpoint_path).resolve().parents[2])


def save_eval_results(run_dir: str, results: list, columns: list) -> str:
    """Write per-budget curve to <run_dir>/eval_results.csv.

    Args:
        results: list of tuples, one per budget step.
        columns: column names matching the tuple layout.
    """
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "eval_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in results:
            writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])
    print(f"Eval curve saved → {out_path}")
    return out_path


def auc_under_curve(budgets: list, values: list) -> float:
    """Normalized area under a metric-vs-budget curve (trapezoidal, range [0,1])."""
    b = np.array(budgets, dtype=float)
    v = np.array(values, dtype=float)
    if b[-1] == b[0]:
        return float(v[0])
    return float(np.trapz(v, b) / (b[-1] - b[0]))


def append_to_results_log(
    run_name: str,
    dataset: str,
    summary: dict,
    log_path: str = "results_log.csv",
) -> None:
    """Append one summary row to the global results_log.csv.

    Args:
        summary: flat dict of scalar metrics (e.g. final_acc, auc_acc_curve, …).
    """
    row = {
        "run_name": run_name,
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **summary,
    }
    exists = Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Summary appended → {log_path}  (run: {run_name})")
