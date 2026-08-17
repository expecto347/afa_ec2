"""
Export Syn1/2/3 data to CSV files for the baseline repos (DIME / dynamic-selection / aaco).

Same generation seed and split as ec2_sefa.py (--syn_seed 8139, 60k/10k/10k),
so EC2 and all baselines see exactly the same data. Raw values are exported;
each baseline applies its own StandardScaler internally (same as grid_stability).

Usage:
    apptainer exec --nv afa_env.sif python -u prepare_syn.py [--dataset syn1|syn2|syn3|all] [--out_root /data/users/haozhes]
"""

import argparse
import os

import numpy as np
import pandas as pd

from ec2_sefa import generate_syn_data

FEATURE_NAMES = [f"x{i}" for i in range(1, 12)]


def export_dataset(dataset, out_root):
    rule = int(dataset[-1])
    X_tr, y_tr, X_val, y_val, X_te, y_te = generate_syn_data(rule, 8139)

    out_dir = os.path.join(out_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    for name, X, y in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)]:
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df["label"] = y
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {dataset}/{name}.csv  shape={df.shape}  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["syn1", "syn2", "syn3", "all"])
    parser.add_argument("--out_root", type=str, default="/data/users/haozhes")
    args = parser.parse_args()

    datasets = ["syn1", "syn2", "syn3"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        export_dataset(ds, args.out_root)


if __name__ == "__main__":
    main()
