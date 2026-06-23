"""
Generic tabular CSV training entry point for the EC2 kNN algorithm.

Usage:
    apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
        --train_csv /path/to/train.csv \
        --val_csv   /path/to/val.csv \
        --test_csv  /path/to/test.csv \
        [--label_col LABEL_COL]   # column name or int index; default: last column

d_in and d_out are inferred automatically from the data.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from ec2_core import (
    MaskLayer, Predictor, PredictorPretrainer,
    EC2Estimator, compute_feature_thresholds, build_trainer,
)


class TabularDataset(Dataset):
    """Loads any CSV for tabular classification. Label column is configurable."""

    def __init__(self, csv_file, label_col=-1, mean=None, std=None):
        df = pd.read_csv(csv_file)

        if isinstance(label_col, str):
            label_series = df[label_col]
            feature_df = df.drop(columns=[label_col])
        else:
            col_idx = label_col if label_col >= 0 else df.shape[1] + label_col
            label_series = df.iloc[:, col_idx]
            feature_df = df.drop(df.columns[col_idx], axis=1)

        features = feature_df.values.astype(np.float32)
        if mean is not None and std is not None:
            features = (features - mean) / (std + 1e-8)
        self.features = features
        self.labels = label_series.values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def _infer_dims(csv_file, label_col):
    df = pd.read_csv(csv_file)
    if isinstance(label_col, str):
        d_in = df.shape[1] - 1
        d_out = int(df[label_col].nunique())
    else:
        col_idx = label_col if label_col >= 0 else df.shape[1] + label_col
        d_in = df.shape[1] - 1
        d_out = int(df.iloc[:, col_idx].nunique())
    return d_in, d_out


def _parse_label_col(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--label_col", type=_parse_label_col, default=-1,
                        help="Label column: column name string or integer index (default: -1 = last column)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features", type=int, default=None,
                        help="Max features to acquire per step (default: d_in)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--run_name", type=str, default="ec2_tabular_trial_0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--k", type=int, default=50, help="Number of kNN neighbors")
    parser.add_argument("--knn_subsample", type=int, default=None,
                        help="Cap X_train reference set for kNN to this many samples "
                             "(reduces the [B,N,d] distance tensor when N is large)")
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    d_in, d_out = _infer_dims(args.train_csv, args.label_col)
    max_features = args.max_features if args.max_features is not None else d_in
    print(f"Inferred d_in={d_in}, d_out={d_out}, max_features={max_features}")

    raw_train_df = pd.read_csv(args.train_csv)
    if isinstance(args.label_col, str):
        X_raw = raw_train_df.drop(columns=[args.label_col]).values.astype(np.float32)
    else:
        col_idx = args.label_col if args.label_col >= 0 else raw_train_df.shape[1] + args.label_col
        X_raw = raw_train_df.drop(raw_train_df.columns[col_idx], axis=1).values.astype(np.float32)

    # Fix 2: only standardize continuous columns; leave binary one-hot columns as 0/1
    is_binary = np.array([np.all(np.isin(X_raw[:, j], [0.0, 1.0])) for j in range(d_in)])
    cont_cols = np.where(~is_binary)[0]
    feat_mean = np.zeros(d_in, dtype=np.float32)
    feat_std  = np.ones(d_in, dtype=np.float32)
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        scaler.fit(X_raw[:, cont_cols])
        feat_mean[cont_cols] = scaler.mean_.astype(np.float32)
        feat_std[cont_cols]  = scaler.scale_.astype(np.float32)
    print(f"Continuous columns: {len(cont_cols)}, binary columns: {is_binary.sum()}")
    X_train_tensor = torch.tensor((X_raw - feat_mean) / (feat_std + 1e-8))

    if isinstance(args.label_col, str):
        y_raw = raw_train_df[args.label_col].values.astype(np.int64)
    else:
        col_idx = args.label_col if args.label_col >= 0 else raw_train_df.shape[1] + args.label_col
        y_raw = raw_train_df.iloc[:, col_idx].values.astype(np.int64)
    y_train_tensor = torch.tensor(y_raw)
    feat_thresholds = compute_feature_thresholds(X_train_tensor)

    # Fix 3: load feature groups (produced by prepare_bank_marketing.py)
    groups_path = os.path.join(os.path.dirname(os.path.abspath(args.train_csv)), "feature_groups.json")
    feature_groups = None
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            feature_groups = json.load(f)
        n_groups = len(feature_groups)
        print(f"Loaded feature_groups.json: {n_groups} groups from {d_in} columns")
        max_features = min(args.max_features, n_groups) if args.max_features is not None else n_groups
    else:
        max_features = args.max_features if args.max_features is not None else d_in

    train_data = TabularDataset(args.train_csv, args.label_col, mean=feat_mean, std=feat_std)
    val_data   = TabularDataset(args.val_csv,   args.label_col, mean=feat_mean, std=feat_std)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False,
                                  pin_memory=True, num_workers=args.num_workers)
    test_dataloader  = None
    if args.test_csv:
        test_data = TabularDataset(args.test_csv, args.label_col, mean=feat_mean, std=feat_std)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                     pin_memory=True, num_workers=args.num_workers)

    hidden  = args.hidden
    dropout = args.dropout

    predictor = Predictor(d_in, d_out, hidden, dropout)
    mask_layer = MaskLayer(mask_size=d_in, append=True)

    value_network = nn.Sequential(
        nn.Linear(d_in * 2, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_in),
    )

    print("Pretraining Predictor...")
    pred_pretrainer = PredictorPretrainer(predictor, mask_layer, lr=1e-3)
    pred_logger = TensorBoardLogger(args.log_dir, name=args.run_name, version="pretrain_predictor")
    pred_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pred_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pred_trainer = build_trainer(args.pretrain_epochs, pred_logger, pred_callbacks, args.gpu, precision=32)
    pred_trainer.fit(pred_pretrainer, train_dataloader, val_dataloader)

    if test_dataloader is not None:
        print("Testing Predictor...")
        pred_trainer.test(pred_pretrainer, dataloaders=test_dataloader)

    if args.pretrain_only:
        print("Pretraining completed. Exiting because --pretrain_only is set.")
        return

    print("Training EC2 Estimator...")
    ec2_estimator = EC2Estimator(
        value_network, predictor, mask_layer,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        feat_thresholds=feat_thresholds,
        feat_mean=feat_mean,
        feat_std=feat_std,
        k=args.k,
        knn_subsample=args.knn_subsample,
        lr=1e-3, min_lr=1e-6,
        max_features=max_features, eps=0.1,
        patience=args.patience,
        disable_early_stopping=args.disable_early_stopping,
        val_monitor='auroc',
        feature_groups=feature_groups,
    )

    logger = TensorBoardLogger(args.log_dir, name=args.run_name, version="ec2_train")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="AUROC Val/Predictor Mean", mode="max",
        filename="best_val_perf_model", verbose=False,
    )

    train_trainer = build_trainer(
        args.train_epochs, logger,
        [checkpoint_callback, TQDMProgressBar(refresh_rate=50)],
        args.gpu, precision=32,
    )
    train_trainer.fit(ec2_estimator, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
