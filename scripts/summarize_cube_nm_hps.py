"""
CUBE-NM HPS 汇总: 合并 results_log.csv 的 AUC/准确率与 context_stats.json 的
context-first 统计，输出对比表格。

用法: apptainer exec afa_env.sif python -u scripts/summarize_cube_nm_hps.py [run_name ...]
     不带参数时汇总所有 cube_hps_* 运行。
"""

import csv
import json
import os
import sys

LOG = "results_log.csv"
BASE = "logs"


def main():
    run_filter = sys.argv[1:] or None

    # 读 results_log（每 run 取最后一行）
    summaries = {}
    with open(LOG) as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "cube_nm":
                continue
            summaries[row["run_name"]] = row

    runs = sorted(
        (r for r in summaries if run_filter is None or r in run_filter),
        key=lambda r: float(summaries[r].get("auc_acc_curve", 0)),
        reverse=True,
    )

    print(f"{'run_name':<24} {'AUC':>7} {'acc@7':>6} {'acc@5':>6} {'acc@4':>6} "
          f"{'step1_ctx':>9} {'ctx10':>6} {'1st_ctx_step':>12} {'perfect@':>8}")
    print("-" * 100)

    for r in runs:
        s = summaries[r]
        auc = float(s.get("auc_acc_curve", 0))
        accs = {}
        csv_path = os.path.join(BASE, r, "eval_results.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    accs[int(float(row["budget"]))] = float(row["ec2_acc"])
        ctx = {}
        ctx_path = os.path.join(BASE, r, "context_stats.json")
        if os.path.exists(ctx_path):
            with open(ctx_path) as f:
                ctx = json.load(f)

        step1_ctx = ctx.get("first_step_picks_context_frac", float("nan"))
        ctx10 = ctx.get("frac_context_within_first_10_steps", float("nan"))
        first_ctx = ctx.get("mean_step_of_first_context_selection", float("nan"))
        perfect = ctx.get("budget_of_first_perfect_acc", -1)
        print(f"{r:<24} {auc:>7.4f} {accs.get(7, float('nan')):>6.3f} "
              f"{accs.get(5, float('nan')):>6.3f} {accs.get(4, float('nan')):>6.3f} "
              f"{step1_ctx:>9.3f} {ctx10:>6.3f} {first_ctx:>12.2f} {perfect:>8}")


if __name__ == "__main__":
    main()
