"""
One-time preprocessing for the UCI Chronic Kidney Disease dataset (ID 336).

Downloads automatically via ucimlrepo and produces three CSVs in --output_dir:
    train.csv  (70%)
    val.csv    (15%)
    test.csv   (15%)

The 10 nominal attributes (all binary after cleaning) are one-hot encoded with
drop_first=True, yielding one binary column per nominal feature — no
feature_groups.json is needed.  Missing values are imputed from the training
split only (median for numeric, mode for nominal) to avoid data leakage.

Run once before launching run_ckd.sh:
    apptainer exec --nv afa_env.sif python -u prepare_ckd.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_OUT = "/data/users/haozhes/ckd"

# All 10 nominal attributes — all are binary after cleaning → drop_first=True
NOMINAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

# 14 numeric / ordinal-as-numeric attributes
# ucimlrepo delivers white blood cell count as "wbcc" and red blood cell count as "rbcc"
NUMERIC_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
                "hemo", "pcv", "wbcc", "rbcc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.output_dir
    if all(os.path.exists(os.path.join(out, f)) for f in ["train.csv", "val.csv", "test.csv"]):
        print(f"CSVs already exist in {out}/ — skipping download and preprocessing.")
        return

    print("Downloading Chronic Kidney Disease dataset via ucimlrepo (ID 336)...")
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=336)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    print(f"Downloaded: {len(X)} rows, {X.shape[1]} features")

    # Strip leading/trailing whitespace and tabs from nominal columns
    for col in NOMINAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().str.lower()
            X[col] = X[col].replace("nan", np.nan)

    # Build label: ckd→1, notckd→0  (handle '\tckd', ' ckd', etc.)
    raw_target = y.iloc[:, 0].astype(str).str.strip().str.lower()
    labels = raw_target.apply(lambda v: 1 if ("ckd" in v and "not" not in v) else 0)

    df = X.copy()
    df["label"] = labels.values

    pos = df["label"].sum()
    print(f"Class distribution: ckd={pos} ({100*pos/len(df):.1f}%), "
          f"notckd={len(df)-pos} ({100*(len(df)-pos)/len(df):.1f}%)")

    # Stratified 70 / 15 / 15 split
    train_val, test = train_test_split(df, test_size=0.15, random_state=args.seed,
                                       stratify=df["label"])
    train, val = train_test_split(train_val, test_size=0.15 / 0.85,
                                  random_state=args.seed, stratify=train_val["label"])
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    # --- Impute missing values (fit on train only) ---
    for col in NUMERIC_COLS:
        if col not in train.columns:
            continue
        med = train[col].median()
        train[col] = train[col].fillna(med)
        val[col]   = val[col].fillna(med)
        test[col]  = test[col].fillna(med)

    for col in NOMINAL_COLS:
        if col not in train.columns:
            continue
        modes = train[col].dropna().mode()
        mode_val = modes.iloc[0] if len(modes) > 0 else "unknown"
        train[col] = train[col].fillna(mode_val)
        val[col]   = val[col].fillna(mode_val)
        test[col]  = test[col].fillna(mode_val)

    # --- One-hot encode nominal columns (drop_first → 1 binary column each) ---
    train = pd.get_dummies(train, columns=NOMINAL_COLS, drop_first=True)
    val   = pd.get_dummies(val,   columns=NOMINAL_COLS, drop_first=True)
    test  = pd.get_dummies(test,  columns=NOMINAL_COLS, drop_first=True)

    # Align val/test to train's column set (rare levels may be absent)
    train_cols = train.columns.tolist()
    val  = val.reindex(columns=train_cols, fill_value=0)
    test = test.reindex(columns=train_cols, fill_value=0)

    # Convert bool columns to int
    for split in [train, val, test]:
        bool_cols = split.select_dtypes(include="bool").columns
        split[bool_cols] = split[bool_cols].astype(int)

    # Move label to last column
    for split in [train, val, test]:
        lbl = split.pop("label")
        split["label"] = lbl

    feature_cols = [c for c in train.columns if c != "label"]
    d_in = len(feature_cols)
    print(f"Features after encoding: d_in={d_in}")
    print(f"Columns: {feature_cols}")

    os.makedirs(out, exist_ok=True)
    train.to_csv(os.path.join(out, "train.csv"), index=False)
    val.to_csv(  os.path.join(out, "val.csv"),   index=False)
    test.to_csv( os.path.join(out, "test.csv"),  index=False)
    print(f"Saved CSVs to {out}/")


if __name__ == "__main__":
    main()
