"""
Diagnose the step-1 (and step-2) acquisition choices of a trained syn model.

Dumps, for the first N test samples: raw feature values + step-1 and step-2
selected features, to a CSV. Analysis is done offline (numpy/sklearn on the
dump) — this script only runs the greedy loop and saves.

Usage (SLURM):
    apptainer exec --nv afa_env.sif python -u diagnose_syn_step1.py \
        --dataset syn1 --seed 0 --n 3000
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ec2_core import EC2Estimator, MaskLayer, Predictor
from ec2_sefa import prepare_data

FEAT_NAMES = [f"x{i}" for i in range(1, 12)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["syn1", "syn2", "syn3"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--syn_seed", type=int, default=8139)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join("/data/users/haozhes/AFA_EC2/logs",
                             f"ec2_{args.dataset}_seed{args.seed}",
                             "ec2_train", "checkpoints", "best_val_perf_model.ckpt")
    print(f"Loading {ckpt_path}")
    temp_ckpt       = torch.load(ckpt_path, map_location="cpu")
    X_train         = temp_ckpt["state_dict"]["X_train"]
    y_train         = temp_ckpt["state_dict"]["y_train"]
    feat_thresholds = temp_ckpt["state_dict"]["feat_thresholds"]
    feat_mean       = temp_ckpt["state_dict"]["feat_mean"].numpy()
    feat_std        = temp_ckpt["state_dict"]["feat_std"].numpy()

    d_in  = X_train.shape[1]
    d_out = int(y_train.max().item()) + 1
    max_features = d_in

    _, _, _, _, X_test_t, y_test_t, feat_mean_np, feat_std_np = prepare_data(args)
    X_test_raw = X_test_t * torch.tensor(feat_std_np + 1e-8) + torch.tensor(feat_mean_np)

    x_all, y_all = X_test_raw[:args.n], y_test_t[:args.n]
    dl = DataLoader(TensorDataset(x_all, y_all), batch_size=512, shuffle=False)

    mask_layer    = MaskLayer(mask_size=d_in, append=True)
    predictor     = Predictor(d_in, d_out, args.hidden, args.dropout)
    value_network = nn.Sequential(
        nn.Linear(d_in * 2, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, d_in),
    )
    model = EC2Estimator.load_from_checkpoint(
        ckpt_path,
        value_network=value_network, predictor=predictor, mask_layer=mask_layer,
        X_train=X_train, y_train=y_train, feat_thresholds=feat_thresholds,
        feat_mean=feat_mean, feat_std=feat_std,
        lr=1e-3, max_features=max_features, eps=0.1, strict=False,
    ).to(device)
    model.eval()

    choices = {1: [], 2: []}
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            mask = torch.zeros(len(x), d_in, dtype=x.dtype, device=device)
            for step in range(1, 3):
                pred = model.value_network(model.mask_layer(x, mask)) + 1e6 * mask
                best = torch.argmin(pred, dim=1)
                choices[step].append(best.cpu().numpy())
                mask_upd = torch.zeros_like(mask).scatter_(1, best.unsqueeze(1), 1.0)
                mask = torch.max(mask, mask_upd)

    df = pd.DataFrame(x_all.numpy(), columns=FEAT_NAMES)
    df["label"] = y_all.numpy()
    df["step1"] = np.concatenate(choices[1])
    df["step2"] = np.concatenate(choices[2])
    out = os.path.join("/data/users/haozhes/AFA_EC2/logs", f"syn_step1_diag_{args.dataset}_s{args.seed}.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print(df["step1"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
