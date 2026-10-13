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
    finetune_predictor_on_policy, lookahead_init_scores,
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
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for Gini discretization in kNN EC2 targets")
    parser.add_argument("--patience", type=int, default=20, help="LR scheduler patience and early-stopping patience")
    parser.add_argument("--run_name", type=str, default="ec2_grid_stability_trial_0")
    parser.add_argument("--restore_best_predictor", action="store_true",
                        help="After pretraining, reload the predictor weights of the best validation epoch")
    parser.add_argument("--predictor_dropout", type=float, default=0.0,
                        help="Dropout applied inside the predictor (0 = historical behaviour)")
    parser.add_argument("--finetune_predictor", action="store_true",
                        help="Phase 3: fine-tune the predictor on the learned policy's masks")
    parser.add_argument("--unbiased_target", action="store_true",
                        help="Unbiased (U-statistic) estimate of the kNN EC2 target")
    parser.add_argument("--region_target", action="store_true",
                        help="Use decision-region labels (predictor's full-feature argmax) in the target")
    parser.add_argument("--lookahead_init", type=int, default=0,
                        help="Anchor the first acquisition to an L-step lookahead EC2 target (0 = off)")
    parser.add_argument("--lookahead_bins", type=int, default=4)
    parser.add_argument("--fast_knn", action="store_true",
                        help="Compute the masked kNN distance with matmuls (same neighbours up to ties)")
    parser.add_argument("--stop_patience", type=int, default=None,
                        help="Stop when the validation metric has not improved for this many epochs")
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
    feat_thresholds = compute_feature_thresholds(X_train_tensor, n_bins=args.n_bins)

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
    predictor.active_dropout = args.predictor_dropout
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

    run_description = args.run_name

    print("Pretraining Predictor...")
    pred_pretrainer = PredictorPretrainer(predictor, mask_layer, lr=1e-3)
    pred_logger = TensorBoardLogger(args.log_dir, name=run_description, version="pretrain_predictor")
    pred_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pred_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pred_trainer = build_trainer(args.pretrain_epochs, pred_logger, pred_callbacks, args.gpu, precision=32)
    pred_trainer.fit(pred_pretrainer, train_dataloader, val_dataloader)
    if args.restore_best_predictor:
        pred_pretrainer.restore_best()

    print("Testing Predictor...")
    pred_trainer.test(pred_pretrainer, dataloaders=test_dataloader)

    if args.pretrain_only:
        print("Pretraining completed. Exiting because --pretrain_only is set.")
        return

    print("Training EC2 Estimator...")
    init_scores = None
    if args.lookahead_init > 0:
        init_scores = lookahead_init_scores(X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy(),
                                            L=args.lookahead_init, M=args.lookahead_bins)
        print(f"Lookahead (L={args.lookahead_init}, M={args.lookahead_bins}) first-step ranking: "
              f"{init_scores.argsort()[:5].tolist()}")
    y_target = None
    if args.region_target:
        predictor.eval()
        with torch.no_grad():
            dev = next(predictor.parameters()).device
            ones = torch.ones(min(len(X_train_tensor), 4096), X_train_tensor.shape[1], device=dev)
            y_target = torch.cat([
                predictor(mask_layer(X_train_tensor[i:i + 4096].to(dev), ones[:len(X_train_tensor[i:i + 4096])])).argmax(1).cpu()
                for i in range(0, len(X_train_tensor), 4096)]).to(y_train_tensor.dtype)
        print(f"Region labels: agree with observed labels on {(y_target == y_train_tensor).float().mean():.3f} of train")
    ec2_estimator = EC2Estimator(
        value_network, predictor, mask_layer,
        unbiased_target=args.unbiased_target, y_target=y_target, init_scores=init_scores,
        fast_knn=args.fast_knn, stop_patience=args.stop_patience,
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

    if args.finetune_predictor:
        best_ckpt = train_trainer.checkpoint_callback.best_model_path
        print(f"Fine-tuning predictor on policy trajectories (from {best_ckpt})")
        ec2_estimator.load_state_dict(torch.load(best_ckpt, map_location="cpu")["state_dict"])
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        finetune_predictor_on_policy(ec2_estimator, train_dataloader, val_dataloader,
                                     n_steps=ec2_estimator.max_features, device=device)
        ft_path = best_ckpt.replace(".ckpt", "-ft.ckpt")
        train_trainer.save_checkpoint(ft_path)
        print(f"Saved fine-tuned checkpoint: {ft_path}")


if __name__ == "__main__":
    main()
