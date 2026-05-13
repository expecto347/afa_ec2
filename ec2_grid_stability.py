import argparse
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
    MaskLayer, Predictor, PVAE, PVAEPretrainer, PredictorPretrainer,
    EC2Estimator, compute_feature_thresholds, build_trainer,
)


class GridStabilityDataset(Dataset):
    def __init__(self, csv_file, mean=None, std=None):
        df = pd.read_csv(csv_file)
        features = df.iloc[:, :-1].values.astype(np.float32)
        if mean is not None and std is not None:
            features = (features - mean) / (std + 1e-8)
        self.features = features
        self.labels = df.iloc[:, -1].values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument("--data_dir", type=str, default="/data/users/haozhes/DIME/experiments/grid_stability/data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_only", action="store_true", help="Only pretrain PVAE and predictor")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--k", type=int, default=50, help="Number of kNN neighbors")
    parser.add_argument("--patience", type=int, default=20, help="LR scheduler patience and early-stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    raw_train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    X_raw = raw_train_df.iloc[:, :-1].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_raw)
    feat_mean = scaler.mean_.astype(np.float32)
    feat_std  = scaler.scale_.astype(np.float32)
    X_train_tensor = torch.tensor((X_raw - feat_mean) / (feat_std + 1e-8))
    y_train_tensor = torch.tensor(raw_train_df.iloc[:, -1].values.astype(np.int64))
    feat_thresholds = compute_feature_thresholds(X_train_tensor)

    train_data = GridStabilityDataset(os.path.join(args.data_dir, "train.csv"), mean=feat_mean, std=feat_std)
    val_data   = GridStabilityDataset(os.path.join(args.data_dir, "val.csv"),   mean=feat_mean, std=feat_std)
    test_data  = GridStabilityDataset(os.path.join(args.data_dir, "test.csv"),  mean=feat_mean, std=feat_std)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    d_in = 12
    d_out = 2
    hidden = 512
    dropout = 0.3

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

    run_description = f"ec2_grid_stability_trial_0"

    print("Pretraining Predictor...")
    pred_pretrainer = PredictorPretrainer(predictor, mask_layer, lr=1e-3)
    pred_logger = TensorBoardLogger(args.log_dir, name=run_description, version="pretrain_predictor")
    pred_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pred_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pred_trainer = build_trainer(args.pretrain_epochs, pred_logger, pred_callbacks, args.gpu, precision=32)
    pred_trainer.fit(pred_pretrainer, train_dataloader, val_dataloader)

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
        lr=1e-3, min_lr=1e-6,
        max_features=args.max_features, eps=0.1,
        patience=args.patience,
        disable_early_stopping=args.disable_early_stopping,
    )

    logger = TensorBoardLogger(args.log_dir, name=run_description, version="ec2_train")
    checkpoint_callback_perf = ModelCheckpoint(
        save_top_k=1, monitor="Acc Val/Predictor Mean", mode="max", filename="best_val_perf_model", verbose=False
    )

    train_trainer = build_trainer(args.train_epochs, logger, [checkpoint_callback_perf, TQDMProgressBar(refresh_rate=50)], args.gpu, precision=32)
    train_trainer.fit(ec2_estimator, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
