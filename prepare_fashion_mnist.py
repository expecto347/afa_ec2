"""
One-time preprocessing for Fashion MNIST as a tabular dataset.

The 784 pixels are flattened and reduced to the 20 most predictive ones
(selected with STG, same subset as the SEFA/AFA-benchmark preprocessing):
    [10, 38, 121, 146, 202, 246, 248, 341, 343, 362,
     406, 434, 454, 490, 546, 574, 580, 602, 742, 770]

Raw Fashion MNIST files are downloaded once by torchvision into
data/FashionMNIST/raw on the first run.  Produces three CSVs in
--output_dir:
    train.csv  (50000)
    val.csv    (10000)
    test.csv   (10000, the official Fashion MNIST test set)

All features are continuous pixel intensities in [0, 1] — no
feature_groups.json is needed.  Run once before launching
run_fashion_mnist_seeds.sh:
    apptainer exec --nv afa_env.sif python -u prepare_fashion_mnist.py
"""

import os
import argparse
import numpy as np
import pandas as pd

DEFAULT_OUT = "/data/users/haozhes/fashion_mnist"
DEFAULT_ROOT = "data"  # repo-relative: raw files land in data/FashionMNIST/raw

# STG-selected pixel indices (flattened 28x28)
BEST_FEATURES = [10, 38, 121, 146, 202, 246, 248, 341, 343, 362,
                 406, 434, 454, 490, 546, 574, 580, 602, 742, 770]

TRAIN_SIZE = 50_000
VAL_SIZE = 10_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--data_root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.output_dir
    if all(os.path.exists(os.path.join(out, f)) for f in ["train.csv", "val.csv", "test.csv"]):
        print(f"CSVs already exist in {out}/ — skipping preprocessing.")
        return

    import torch
    import torchvision.transforms as T
    from torchvision.datasets import FashionMNIST

    print("Loading Fashion MNIST (train+val pool and official test set)...")
    fmnist_train_val = FashionMNIST(root=args.data_root, train=True, download=True,
                                    transform=T.ToTensor())
    fmnist_test = FashionMNIST(root=args.data_root, train=False, download=True,
                               transform=T.ToTensor())

    X_train_val = torch.stack([b[0].view(-1) for b in fmnist_train_val]).numpy().astype(np.float32)
    y_train_val = np.array([b[1] for b in fmnist_train_val], dtype=np.int64)
    X_test = torch.stack([b[0].view(-1) for b in fmnist_test]).numpy().astype(np.float32)
    y_test = np.array([b[1] for b in fmnist_test], dtype=np.int64)

    # Keep only the STG-selected pixels
    X_train_val = X_train_val[:, BEST_FEATURES]
    X_test = X_test[:, BEST_FEATURES]

    # Shuffle the 60k pool and split into train / val
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(X_train_val.shape[0])
    X_train_val, y_train_val = X_train_val[perm], y_train_val[perm]

    X_train = X_train_val[:TRAIN_SIZE]
    y_train = y_train_val[:TRAIN_SIZE]
    X_val = X_train_val[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    y_val = y_train_val[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    col_names = [f"p{idx}" for idx in BEST_FEATURES]

    def to_df(X, y):
        df = pd.DataFrame(X, columns=col_names)
        df["label"] = y
        return df

    train, val, test = to_df(X_train, y_train), to_df(X_val, y_val), to_df(X_test, y_test)

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"Features: d_in={len(col_names)}, classes={np.unique(y_train).size}")
    print(f"Train class distribution: {np.bincount(y_train)}")

    os.makedirs(out, exist_ok=True)
    train.to_csv(os.path.join(out, "train.csv"), index=False)
    val.to_csv(os.path.join(out, "val.csv"), index=False)
    test.to_csv(os.path.join(out, "test.csv"), index=False)
    print(f"Saved CSVs to {out}/")


if __name__ == "__main__":
    main()
