"""
One-time preprocessing for the California Housing dataset (sklearn).

The regression target (median house value, in $100k) is discretized into 4
equally-sized bins via quantiles, turning the task into 4-way classification
(same convention as the SEFA/AFA-benchmark preprocessing).  All 8 features are
continuous — no feature_groups.json is needed.

Downloads automatically via sklearn on first run and produces three CSVs in
--output_dir:
    train.csv  (16512)
    val.csv    (2064)
    test.csv   (2064)

Run once before launching run_california_housing_seeds.sh:
    apptainer exec --nv afa_env.sif python -u prepare_california_housing.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

DEFAULT_OUT = "/data/users/haozhes/california_housing"
NUM_CLASSES = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.output_dir
    if all(os.path.exists(os.path.join(out, f)) for f in ["train.csv", "val.csv", "test.csv"]):
        print(f"CSVs already exist in {out}/ — skipping download and preprocessing.")
        return

    print("Fetching California Housing via sklearn...")
    ds = fetch_california_housing(as_frame=True)
    X = ds.data.values.astype(np.float32)
    y_reg = ds.target.values
    feature_cols = list(ds.data.columns)
    print(f"Downloaded: {len(X)} rows, {X.shape[1]} features: {feature_cols}")

    # Discretize the target into NUM_CLASSES equally-sized bins (quantile cut)
    y_sorted = np.sort(y_reg)
    bins = [y_sorted[int(i * len(y_sorted) / NUM_CLASSES)] for i in range(1, NUM_CLASSES)]
    bins.append(y_sorted[-1])
    y = np.digitize(y_reg, bins, right=True).astype(np.int64)
    assert np.all(np.unique(y) == np.arange(NUM_CLASSES)), \
        "Labels should be consecutive integers 0..3"

    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y
    print(f"Class distribution: {np.bincount(y)}")

    # Stratified 80 / 10 / 10 split
    train, temp = train_test_split(df, test_size=0.2, random_state=args.seed,
                                   stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=args.seed,
                                 stratify=temp["label"])
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    os.makedirs(out, exist_ok=True)
    train.to_csv(os.path.join(out, "train.csv"), index=False)
    val.to_csv(os.path.join(out, "val.csv"), index=False)
    test.to_csv(os.path.join(out, "test.csv"), index=False)
    print(f"Saved CSVs to {out}/")


if __name__ == "__main__":
    main()
