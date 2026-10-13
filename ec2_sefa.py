"""
Training entry point for SEFA benchmark datasets: Syn1, Syn2, Syn3, Metabric.

Syn1/2/3 are generated inline (no external files needed).
Metabric requires pre-processed .pt files from SEFA — see CLAUDE.md prerequisites.

Usage:
    apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
        --dataset syn1 --run_name ec2_syn1_trial_0 --patience 20
"""

import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ec2_core import (
    finetune_predictor_on_policy, lookahead_init_scores,
    EC2Estimator,
    MaskLayer,
    Predictor,
    PredictorPretrainer,
    build_trainer,
    compute_feature_thresholds,
)

SEFA_DEFAULT_DATA_DIR = "/data/users/haozhes/SEFA/datasets/data"


def generate_syn_data(rule, seed):
    """Generate synthetic INVASE data for rule 1/2/3 (same formula as SEFA's create_synthetic.py).

    Returns six numpy arrays: X_train, y_train, X_val, y_val, X_test, y_test.
    Split: 60k / 10k / 10k.
    """
    rng = np.random.RandomState(seed)
    n = 80_000
    X = rng.normal(size=(n, 11)).astype(np.float32)

    syn1 = 4 * X[:, 0] * X[:, 1]
    syn2 = 1.2 * np.sum(X[:, 2:6] ** 2, axis=-1) - 4.2
    syn3 = -10 * np.sin(0.2 * X[:, 6]) + np.abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9]) - 2.4

    if rule == 1:
        logit1, logit2 = syn1, syn2
    elif rule == 2:
        logit1, logit2 = syn1, syn3
    elif rule == 3:
        logit1, logit2 = syn2, syn3
    else:
        raise ValueError(f"Unknown rule {rule}")

    feat11_lt0 = (X[:, -1] < 0).astype(np.float32)
    logit = feat11_lt0 * logit1 + (1.0 - feat11_lt0) * logit2
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(logit)), size=n).astype(np.int64)

    return (X[:60_000], y[:60_000],
            X[60_000:70_000], y[60_000:70_000],
            X[70_000:], y[70_000:])


def load_metabric(sefa_data_dir):
    """Load pre-standardized Metabric tensors from SEFA .pt files."""
    path = os.path.join(sefa_data_dir, "metabric")
    missing = [f for f in ["X_train_std.pt", "y_train.pt"] if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(
            f"Metabric .pt files not found in {path}. "
            "Run SEFA's create_metabric.py after downloading the Kaggle CSV — see CLAUDE.md."
        )
    X_train = torch.load(os.path.join(path, "X_train_std.pt"), weights_only=True).float()
    X_val   = torch.load(os.path.join(path, "X_val_std.pt"),   weights_only=True).float()
    X_test  = torch.load(os.path.join(path, "X_test_std.pt"),  weights_only=True).float()
    y_train = torch.load(os.path.join(path, "y_train.pt"),     weights_only=True).long()
    y_val   = torch.load(os.path.join(path, "y_val.pt"),       weights_only=True).long()
    y_test  = torch.load(os.path.join(path, "y_test.pt"),      weights_only=True).long()
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_data(args):
    """Load/generate data for args.dataset.

    Returns:
        X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t  — torch tensors
        feat_mean, feat_std  — np.float32 arrays used for EC2Estimator buffers
    """
    if args.dataset.startswith("syn"):
        rule = int(args.dataset[-1])
        X_tr, y_tr, X_val, y_val, X_te, y_te = generate_syn_data(rule, args.syn_seed)
        scaler = StandardScaler().fit(X_tr)
        feat_mean = scaler.mean_.astype(np.float32)
        feat_std  = scaler.scale_.astype(np.float32)

        def norm(X):
            return torch.tensor((X - feat_mean) / (feat_std + 1e-8))

        return (norm(X_tr), torch.tensor(y_tr),
                norm(X_val), torch.tensor(y_val),
                norm(X_te), torch.tensor(y_te),
                feat_mean, feat_std)

    elif args.dataset == "metabric":
        X_tr, y_tr, X_val, y_val, X_te, y_te = load_metabric(args.sefa_data_dir)
        # Data is pre-standardized; store identity params as convention
        d_in = X_tr.shape[1]
        feat_mean = np.zeros(d_in, dtype=np.float32)
        feat_std  = np.ones(d_in,  dtype=np.float32)
        return X_tr, y_tr, X_val, y_val, X_te, y_te, feat_mean, feat_std

    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["syn1", "syn2", "syn3", "metabric"])
    parser.add_argument("--sefa_data_dir", type=str, default=SEFA_DEFAULT_DATA_DIR)
    parser.add_argument("--syn_seed", type=int, default=8139, help="RNG seed for synthetic data generation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features", type=int, default=None, help="Features acquired per episode (default: d_in)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--run_name", type=str, default=None, help="TensorBoard run name (default: ec2_<dataset>_trial_0)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--disable_early_stopping", action="store_true")
    parser.add_argument("--k", type=int, default=50, help="kNN neighbors for EC2 target computation")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for Gini discretization in kNN EC2 targets")
    parser.add_argument("--patience", type=int, default=20)
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
    parser.add_argument("--knn_subsample", type=int, default=None,
                        help="Cap the kNN reference set to this many training samples")
    parser.add_argument("--lr", type=float, default=1e-3, help="Value-network learning rate")
    parser.add_argument("--stop_patience", type=int, default=None,
                        help="Stop when the validation metric has not improved for this many epochs")
    parser.add_argument("--random_first_step", action="store_true",
                        help="Force step-0 action to be uniformly random during training")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"ec2_{args.dataset}_trial_0"

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, feat_mean, feat_std = prepare_data(args)

    d_in  = X_train_t.shape[1]
    d_out = int(y_train_t.max().item()) + 1
    max_features = args.max_features if args.max_features is not None else d_in
    print(f"Dataset={args.dataset}  d_in={d_in}  d_out={d_out}  max_features={max_features}")
    print(f"Train={len(X_train_t)}  Val={len(X_val_t)}  Test={len(X_test_t)}")

    feat_thresholds = compute_feature_thresholds(X_train_t, n_bins=args.n_bins)

    train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size,
                          shuffle=True, pin_memory=True, num_workers=args.num_workers, drop_last=True)
    val_dl   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size,
                          shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_dl  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size,
                          shuffle=False, pin_memory=True, num_workers=args.num_workers)

    mask_layer    = MaskLayer(mask_size=d_in, append=True)
    predictor     = Predictor(d_in, d_out, args.hidden, args.dropout)
    predictor.active_dropout = args.predictor_dropout
    value_network = nn.Sequential(
        nn.Linear(d_in * 2, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, d_in),
    )

    print("Pretraining Predictor...")
    pred_pretrainer = PredictorPretrainer(predictor, mask_layer, lr=1e-3)
    pred_logger = TensorBoardLogger(args.log_dir, name=args.run_name, version="pretrain_predictor")
    pred_callbacks = [TQDMProgressBar(refresh_rate=50)]
    if not args.disable_early_stopping:
        pred_callbacks.append(EarlyStopping(monitor="Loss Val/Total", patience=10, mode="min"))
    pred_trainer = build_trainer(args.pretrain_epochs, pred_logger, pred_callbacks, args.gpu)
    pred_trainer.fit(pred_pretrainer, train_dl, val_dl)
    if args.restore_best_predictor:
        pred_pretrainer.restore_best()
    pred_trainer.test(pred_pretrainer, dataloaders=test_dl)

    if args.pretrain_only:
        print("Done (--pretrain_only).")
        return

    print("Training EC2 Estimator...")
    init_scores = None
    if args.lookahead_init > 0:
        init_scores = lookahead_init_scores(X_train_t.cpu().numpy(), y_train_t.cpu().numpy(),
                                            L=args.lookahead_init, M=args.lookahead_bins)
        print(f"Lookahead (L={args.lookahead_init}, M={args.lookahead_bins}) first-step ranking: "
              f"{init_scores.argsort()[:5].tolist()}")
    y_target = None
    if args.region_target:
        predictor.eval()
        with torch.no_grad():
            dev = next(predictor.parameters()).device
            ones = torch.ones(min(len(X_train_t), 4096), X_train_t.shape[1], device=dev)
            y_target = torch.cat([
                predictor(mask_layer(X_train_t[i:i + 4096].to(dev), ones[:len(X_train_t[i:i + 4096])])).argmax(1).cpu()
                for i in range(0, len(X_train_t), 4096)]).to(y_train_t.dtype)
        print(f"Region labels: agree with observed labels on {(y_target == y_train_t).float().mean():.3f} of train")
    ec2_estimator = EC2Estimator(
        value_network, predictor, mask_layer,
        unbiased_target=args.unbiased_target, y_target=y_target, init_scores=init_scores,
        fast_knn=args.fast_knn, stop_patience=args.stop_patience,
        X_train=X_train_t, y_train=y_train_t,
        feat_thresholds=feat_thresholds,
        feat_mean=feat_mean, feat_std=feat_std,
        k=args.k, lr=args.lr, min_lr=1e-6, knn_subsample=args.knn_subsample,
        max_features=max_features, eps=0.1,
        patience=args.patience,
        disable_early_stopping=args.disable_early_stopping,
        random_first_step=args.random_first_step,
    )

    logger = TensorBoardLogger(args.log_dir, name=args.run_name, version="ec2_train")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="Acc Val/Predictor Mean", mode="max",
        filename="best_val_perf_model", verbose=False,
    )
    train_trainer = build_trainer(
        args.train_epochs, logger,
        [checkpoint_callback, TQDMProgressBar(refresh_rate=50)],
        args.gpu,
    )
    train_trainer.fit(ec2_estimator, train_dl, val_dl)

    if args.finetune_predictor:
        best_ckpt = train_trainer.checkpoint_callback.best_model_path
        print(f"Fine-tuning predictor on policy trajectories (from {best_ckpt})")
        ec2_estimator.load_state_dict(torch.load(best_ckpt, map_location="cpu")["state_dict"])
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        finetune_predictor_on_policy(ec2_estimator, train_dl, val_dl,
                                     n_steps=ec2_estimator.max_features, device=device)
        ft_path = best_ckpt.replace(".ckpt", "-ft.ckpt")
        train_trainer.save_checkpoint(ft_path)
        print(f"Saved fine-tuned checkpoint: {ft_path}")


if __name__ == "__main__":
    main()
