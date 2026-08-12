"""
One-time preprocessing for the MiniBooNE particle identification dataset
(UCI ID 199, https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification).

Binary classification: electron-neutrino signal (label 1) vs background
(label 0).  The raw data is class-imbalanced (36499 signal / 93565
background); we enforce balance by downsampling the shuffled background
events to 40000, giving 76499 examples total.  Features are reduced to the
20 selected with STG (same subset as the SEFA/AFA-benchmark preprocessing):
    [2, 3, 6, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
     34, 39, 40, 41, 42, 43, 44]

All features are continuous — no feature_groups.json is needed.  Produces
three CSVs in --output_dir:
    train.csv  (56499)
    val.csv    (10000)
    test.csv   (10000)

If MiniBooNE_PID.txt is not present in --output_dir, it is downloaded from
UCI automatically.  Run once before launching run_miniboone_seeds.sh:
    apptainer exec --nv afa_env.sif python -u prepare_miniboone.py
"""

import os
import argparse
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_OUT = "/data/users/haozhes/miniboone"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip"

# STG-selected feature indices
BEST_FEATURES = [2, 3, 6, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                 34, 39, 40, 41, 42, 43, 44]

VAL_SIZE = 10_000
TEST_SIZE = 10_000
NEG_KEEP = 40_000  # background events kept to enforce balance


def download_raw(out):
    """Download and unzip the UCI MiniBooNE archive into `out`."""
    zip_path = os.path.join(out, "miniboone.zip")
    print(f"Downloading MiniBooNE from UCI: {UCI_ZIP_URL}")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    os.remove(zip_path)
    print("Download and extraction complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.output_dir
    if all(os.path.exists(os.path.join(out, f)) for f in ["train.csv", "val.csv", "test.csv"]):
        print(f"CSVs already exist in {out}/ — skipping download and preprocessing.")
        return

    os.makedirs(out, exist_ok=True)
    raw_path = os.path.join(out, "MiniBooNE_PID.txt")
    if not os.path.exists(raw_path):
        download_raw(out)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found even after download. "
            "Get it manually from https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification"
        )

    print("Loading raw data...")
    X = pd.read_csv(raw_path, sep=r"\s+", skiprows=1, header=None).to_numpy(dtype=np.float32)
    num_pos_neg = pd.read_csv(raw_path, sep=r"\s+", nrows=1, header=None).to_numpy()
    num_positive, num_negative = int(num_pos_neg[0, 0]), int(num_pos_neg[0, 1])
    assert X.shape[0] == num_positive + num_negative, \
        f"Row count {X.shape[0]} != {num_positive}+{num_negative} in first line"
    print(f"Raw: {num_positive} signal / {num_negative} background, {X.shape[1]} features")

    # Downsample background to enforce class balance
    rng = np.random.RandomState(args.seed)
    X_pos = X[:num_positive]
    X_neg = X[num_positive:]
    X_neg = X_neg[rng.permutation(X_neg.shape[0])][:NEG_KEEP]

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.zeros(X.shape[0], dtype=np.int64)
    y[:num_positive] = 1
    print(f"After balancing: {num_positive} signal / {len(X_neg)} background")

    # Keep only the STG-selected features
    X = X[:, BEST_FEATURES]

    col_names = [f"f{i}" for i in range(len(BEST_FEATURES))]
    df = pd.DataFrame(X, columns=col_names)
    df["label"] = y

    # Stratified split -> 56499 train / 10000 val / 10000 test
    train, temp = train_test_split(df, test_size=VAL_SIZE + TEST_SIZE,
                                   random_state=args.seed, stratify=df["label"])
    val, test = train_test_split(temp, test_size=TEST_SIZE,
                                 random_state=args.seed, stratify=temp["label"])
    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")

    train.to_csv(os.path.join(out, "train.csv"), index=False)
    val.to_csv(os.path.join(out, "val.csv"), index=False)
    test.to_csv(os.path.join(out, "test.csv"), index=False)
    print(f"Saved CSVs to {out}/")


if __name__ == "__main__":
    main()
