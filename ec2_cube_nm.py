"""
EC2 kNN training for CUBE-NM (afa_context_v2):
  - AFAContextDataset with noisy settings, n_contexts=5 → 55 features, 8 classes
  - StandardScaler normalization
  - Predictor pretraining + EC2 kNN Gini estimator (no aux head)
"""

import argparse
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from afabench.common.datasets.datasets import AFAContextDataset

from ec2_core import (
    MaskLayer, Predictor, PredictorPretrainer,
    EC2Estimator, compute_feature_thresholds, build_trainer,
)

# CUBE-NM (afa_context_v2) noise configuration
CUBE_NM_KWARGS = dict(
    context_feature_std=0,
    informative_feature_std=0,
    non_informative_feature_mean=0.5,
    non_informative_feature_std=0,
    use_cheap_context_features=True,
    n_contexts=5,
)


class CubeNMDataset(Dataset):
    """Wraps an AFAContextDataset subset.

    Converts one-hot labels to integer class indices and applies StandardScaler
    normalization. All data is materialized upfront for fast DataLoader iteration.
    """

    def __init__(self, afa_subset, mean=None, std=None):
        xs, ys = [], []
        for i in range(len(afa_subset)):
            x, y = afa_subset[i]
            xs.append(x)
            ys.append(torch.argmax(y).long())
        features = torch.stack(xs)   # [N, 55]
        labels   = torch.stack(ys)   # [N]
        if mean is not None and std is not None:
            m = torch.tensor(mean, dtype=torch.float32)
            s = torch.tensor(std,  dtype=torch.float32)
            features = (features - m) / (s + 1e-8)
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (64 keeps peak kNN intermediate tensor ~100MB on GPU)")
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features", type=int, default=14,
                        help="Acquisition steps per training episode (max 55 = full budget)")
    parser.add_argument("--n_samples_total", type=int, default=6250,
                        help="Total samples generated; 80%% train / 20%% val → 5000/1250")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--k", type=int, default=50,
                        help="kNN neighbors (50/5000 ≈ 1%% of training set)")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Discretization bins for kNN EC2 targets")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.1,
                        help="Epsilon-greedy exploration probability")
    parser.add_argument("--eps_decay", type=float, default=0.2)
    parser.add_argument("--eps_steps", type=int, default=8)
    parser.add_argument("--random_first_step", action="store_true",
                        help="First acquisition of every episode is random (better coverage)")
    parser.add_argument("--pretrain_retention_max", type=float, default=1.0,
                        help="Max per-feature retention in pretrain masks (1.0 = default "
                             "U(0,1); lower biases toward low budgets)")
    parser.add_argument("--pretrain_retention_mix", type=float, default=0.0,
                        help="Fraction of pretrain masks kept at full range when "
                             "pretrain_retention_max < 1 (preserves high-budget ability)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--run_name", type=str, default="ec2_cube_nm_trial_0")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    # --- Generate dataset and split 80/20 ---
    full_dataset = AFAContextDataset(
        n_samples=args.n_samples_total,
        seed=args.seed,
        **CUBE_NM_KWARGS,
    )

    np.random.seed(args.seed)
    val_size  = int(len(full_dataset) * 0.2)
    val_inds  = np.sort(np.random.choice(len(full_dataset), size=val_size, replace=False))
    train_inds = np.setdiff1d(np.arange(len(full_dataset)), val_inds)
    print(f"Split: {len(train_inds)} train / {len(val_inds)} val")

    train_subset = full_dataset.create_subset(train_inds)
    val_subset   = full_dataset.create_subset(val_inds)

    # --- Fit StandardScaler on raw training features ---
    X_raw = np.stack([train_subset[i][0].numpy() for i in range(len(train_subset))])
    scaler = StandardScaler()
    scaler.fit(X_raw)
    feat_mean = scaler.mean_.astype(np.float32)
    feat_std  = scaler.scale_.astype(np.float32)

    train_data = CubeNMDataset(train_subset, mean=feat_mean, std=feat_std)
    val_data   = CubeNMDataset(val_subset,   mean=feat_mean, std=feat_std)

    X_train_tensor = train_data.features.clone()
    y_train_tensor = train_data.labels.clone()
    feat_thresholds = compute_feature_thresholds(X_train_tensor, n_bins=args.n_bins)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dataloader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False,
                                  pin_memory=True, num_workers=args.num_workers)

    d_in    = 55
    d_out   = 8
    hidden  = args.hidden
    dropout = args.dropout

    predictor  = Predictor(d_in, d_out, hidden, dropout)
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

    # --- Phase 1: Pretrain Predictor ---
    print("Pretraining Predictor...")
    pred_pretrainer = PredictorPretrainer(
        predictor, mask_layer, lr=args.lr,
        mask_retention_max=args.pretrain_retention_max,
        mask_retention_mix=args.pretrain_retention_mix,
    )
    pred_logger = TensorBoardLogger(args.log_dir, name=args.run_name, version="pretrain_predictor")
    pred_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pred_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pred_trainer = build_trainer(
        args.pretrain_epochs, pred_logger, pred_callbacks, args.gpu, precision=32
    )
    pred_trainer.fit(pred_pretrainer, train_dataloader, val_dataloader)
    pred_trainer.test(pred_pretrainer, dataloaders=val_dataloader)

    if args.pretrain_only:
        print("Pretraining completed. Exiting.")
        return

    # --- Phase 2: Train EC2 Estimator (kNN Gini targets) ---
    print("Training EC2 Estimator...")
    ec2_estimator = EC2Estimator(
        value_network, predictor, mask_layer,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        feat_thresholds=feat_thresholds,
        feat_mean=feat_mean,
        feat_std=feat_std,
        k=args.k,
        lr=args.lr, min_lr=1e-6,
        max_features=args.max_features,
        eps=args.eps,
        eps_decay=args.eps_decay,
        eps_steps=args.eps_steps,
        random_first_step=args.random_first_step,
        patience=args.patience,
        disable_early_stopping=args.disable_early_stopping,
    )

    logger = TensorBoardLogger(args.log_dir, name=args.run_name, version="ec2_train")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="Acc Val/Predictor Mean", mode="max",
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
