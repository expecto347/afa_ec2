"""Summarize hyperparameter sweep results (k × n_bins) from eval_results.csv files.

Usage:
    python -u summarize_hps.py

Scans logs/ec2_<dataset>_k<K>_b<B>_seed0/eval_results.csv (last row = max budget),
prints per-dataset 3×3 tables for final acc / final auroc, and optionally
compares against the default config (k=50, bins=10) from results_log.csv.
"""

import csv
import glob
import os
import re

import numpy as np

RUN_RE = re.compile(r"ec2_(?P<ds>[\w]+)_k(?P<k>\d+)_b(?P<bins>\d+)_seed(?P<seed>\d+)")

DEFAULT_K, DEFAULT_BINS = 50, 10


def read_curve(csv_path):
    """Return (budget list, acc list, auroc list)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        budgets = [float(r["budget"]) for r in rows]
        accs = [float(r["ec2_acc"]) for r in rows]
        aurocs = [float(r["ec2_auroc"]) for r in rows]
    except KeyError:  # positional fallback
        with open(csv_path) as f:
            vals = [list(r) for r in csv.reader(f)][1:]
        budgets = [float(v[0]) for v in vals]
        accs = [float(v[1]) for v in vals]
        aurocs = [float(v[2]) for v in vals]
    return np.array(budgets), np.array(accs), np.array(aurocs)


def curve_auc(budget, vals):
    """Trapezoidal AUC normalized to [0, 1] over the budget range."""
    b = budget - budget.min()
    if b[-1] == 0:
        return float(vals[-1])
    return float(np.trapezoid(vals, b) / b[-1])


def collect_eval_results(log_dir="logs"):
    """ds -> {(k, bins) -> dict(final_acc, final_auroc, auc_acc, auc_auroc)}"""
    results = {}
    for path in glob.glob(os.path.join(log_dir, "ec2_*_k*_b*_seed*/eval_results.csv")):
        m = RUN_RE.search(os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        ds, k, bins = m.group("ds"), int(m.group("k")), int(m.group("bins"))
        curve = read_curve(path)
        if curve is None:
            continue
        budgets, accs, aurocs = curve
        results.setdefault(ds, {})[(k, bins)] = {
            "final_acc": float(accs[-1]),
            "final_auroc": float(aurocs[-1]),
            "auc_acc": curve_auc(budgets, accs),
            "auc_auroc": curve_auc(budgets, aurocs),
        }
    return results


def load_results_log(path="results_log.csv"):
    """dataset -> {run_name: {final_acc, final_auroc, timestamp}}"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            try:
                out.setdefault(ds, {})[row["run_name"]] = {
                    "final_acc": float(row["final_acc"]),
                    "final_auroc": float(row["final_auroc"]),
                    "timestamp": row["timestamp"],
                }
            except (KeyError, ValueError):
                continue
    return out


def print_grid(ds, table, metric_key):
    keys = sorted(table.keys(), key=lambda kv: kv[0])
    ks = sorted({k for k, _ in keys})
    bins = sorted({b for _, b in keys})
    header = "k \\ bins | " + " | ".join(f"b={b:<7}" for b in bins)
    print(f"  {metric_key} ({ds}):")
    print("  " + header)
    print("  " + "-" * len(header))
    for k in ks:
        cells = []
        for b in bins:
            v = table.get((k, b))
            cells.append(f"{v[metric_key]:.4f}" if v else "    -    ")
        print(f"   k={k:<4} | " + " | ".join(cells))


def main():
    results = collect_eval_results()
    if not results:
        print("No eval_results.csv found yet.")
        return

    log = load_results_log()

    for ds in sorted(results.keys()):
        table = results[ds]
        print(f"\n=== {ds} ===")
        print_grid(ds, table, "auc_acc")
        print_grid(ds, table, "auc_auroc")
        print_grid(ds, table, "final_acc")
        print_grid(ds, table, "final_auroc")

        best = {}
        for metric in ["auc_acc", "auc_auroc", "final_acc", "final_auroc"]:
            k, b = max(table.items(), key=lambda kv: kv[1][metric])[0]
            best[metric] = (k, b, table[(k, b)][metric])
            print(f"  BEST {metric:<9}: k={best[metric][0]} bins={best[metric][1]} -> {best[metric][2]:.4f}")

        # default-config reference from results_log.csv
        if ds in log:
            entries = log[ds]
            recent = sorted(entries.values(), key=lambda e: e["timestamp"], reverse=True)
            if recent:
                r = recent[0]
                print(f"  (results_log newest @ {r['timestamp']}: "
                      f"final_acc={r['final_acc']:.4f} final_auroc={r['final_auroc']:.4f})")


if __name__ == "__main__":
    main()
