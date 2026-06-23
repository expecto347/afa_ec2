"""
One-time preprocessing for the UCI Bank Marketing dataset.

Expects:
    bank-additional-full.csv  at  --input_csv  (default path below)

Produces three CSVs in --output_dir:
    train.csv  (70%)
    val.csv    (15%)
    test.csv   (15%)

Categorical columns are one-hot encoded; the binary target y (yes/no)
is mapped to 1/0 and placed as the last column named "label".

Run once before launching run_bank_marketing.sh:
    apptainer exec --nv afa_env.sif python -u prepare_bank_marketing.py
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_CSV = "/data/users/haozhes/bank_marketing/bank-additional-full.csv"
DEFAULT_OUT = "/data/users/haozhes/bank_marketing"

CATEGORICAL_COLS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  type=str, default=DEFAULT_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(
            f"CSV not found: {args.input_csv}\n"
            "Download bank-additional-full.csv from "
            "https://archive.ics.uci.edu/dataset/222/bank+marketing "
            f"and place it at {args.input_csv}"
        )

    df = pd.read_csv(args.input_csv, sep=";")
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    df["label"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Move label to last column
    label = df.pop("label")
    df["label"] = label

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Build feature_groups: map each original feature → list of column indices
    feature_cols = [c for c in df.columns if c != "label"]
    d_in = len(feature_cols)
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}

    feature_groups = []
    used = set()
    # Continuous features come before dummies; add them as singleton groups first
    for col in feature_cols:
        if col in used:
            continue
        # Check if this column is a dummy for a categorical (name contains "_")
        matched = False
        for cat in CATEGORICAL_COLS:
            if col.startswith(cat + "_"):
                matched = True
                break
        if not matched:
            feature_groups.append([col_to_idx[col]])
            used.add(col)

    # Now add one group per categorical (all its dummies together)
    for cat in CATEGORICAL_COLS:
        dummy_cols = [c for c in feature_cols if c.startswith(cat + "_")]
        if dummy_cols:
            feature_groups.append([col_to_idx[c] for c in dummy_cols])
            used.update(dummy_cols)

    pos  = df["label"].sum()
    print(f"Features after encoding: d_in={d_in}")
    print(f"Feature groups: {len(feature_groups)} acquisition units")
    print(f"Class distribution: 1 (yes)={pos} ({100*pos/len(df):.1f}%), "
          f"0 (no)={len(df)-pos} ({100*(len(df)-pos)/len(df):.1f}%)")

    train_val, test = train_test_split(df, test_size=0.15, random_state=args.seed, stratify=df["label"])
    train, val      = train_test_split(train_val, test_size=0.15/0.85, random_state=args.seed, stratify=train_val["label"])

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    os.makedirs(args.output_dir, exist_ok=True)
    train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val.to_csv(  os.path.join(args.output_dir, "val.csv"),   index=False)
    test.to_csv( os.path.join(args.output_dir, "test.csv"),  index=False)

    groups_path = os.path.join(args.output_dir, "feature_groups.json")
    with open(groups_path, "w") as f:
        json.dump(feature_groups, f)
    print(f"Saved CSVs and feature_groups.json to {args.output_dir}/")


if __name__ == "__main__":
    main()
