"""
Generic evaluation script for EC2 checkpoints trained with ec2_tabular.py.

d_in and d_out are inferred automatically from the checkpoint's registered buffers
(X_train and y_train), so you only need to supply hidden/dropout if they differ
from the defaults (512 / 0.3).

Usage:
    apptainer exec --nv afa_env.sif python -u evaluate_ec2_tabular.py \
        --checkpoint logs/ec2_tabular_trial_0/ec2_train/checkpoints/best_val_perf_model.ckpt \
        --test_csv   /path/to/test.csv \
        [--label_col LABEL_COL]
"""

import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from ec2_tabular import TabularDataset, _parse_label_col
from ec2_core import Predictor, EC2Estimator, MaskLayer


def evaluate_ec2(checkpoint_path, test_csv, label_col, gpu, hidden, dropout):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    temp_ckpt = torch.load(checkpoint_path, map_location='cpu')
    feat_mean       = temp_ckpt['state_dict']['feat_mean'].numpy()
    feat_std        = temp_ckpt['state_dict']['feat_std'].numpy()
    X_train         = temp_ckpt['state_dict']['X_train']
    y_train         = temp_ckpt['state_dict']['y_train']
    feat_thresholds = temp_ckpt['state_dict']['feat_thresholds']

    d_in  = X_train.shape[1]
    d_out = int(y_train.max().item()) + 1
    max_features = d_in
    print(f"Inferred from checkpoint: d_in={d_in}, d_out={d_out}")

    print("Loading test dataset...")
    test_dataset = TabularDataset(test_csv, label_col=label_col, mean=feat_mean, std=feat_std)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True, num_workers=4)

    print("Initializing model components...")
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

    model = EC2Estimator.load_from_checkpoint(
        checkpoint_path,
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        X_train=X_train,
        y_train=y_train,
        feat_thresholds=feat_thresholds,
        feat_mean=feat_mean,
        feat_std=feat_std,
        lr=1e-3,
        max_features=max_features,
        eps=0.1,
        strict=False,
    ).to(device)

    model.eval()

    all_preds = [[] for _ in range(max_features + 1)]
    all_preds_random = [[] for _ in range(max_features + 1)]
    all_y = []

    print("Evaluating EC2 Estimator and Random Baseline...")

    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            mask = torch.zeros(len(x), d_in, dtype=x.dtype, device=x.device)
            x_masked = model.mask_layer(x, mask)
            pred_main = model.estimate_class_frequency_mc(x, mask, x_masked)
            all_preds[0].append(pred_main.cpu())
            all_preds_random[0].append(pred_main.cpu())
            all_y.append(y.cpu())

            mask_ec2  = torch.zeros(len(x), d_in, dtype=x.dtype, device=x.device)
            mask_rand = torch.zeros(len(x), d_in, dtype=x.dtype, device=x.device)

            for feature_step in range(1, max_features + 1):
                # EC2 greedy selection
                x_masked_ec2 = model.mask_layer(x, mask_ec2)
                pred_ec2 = model.value_network(x_masked_ec2)
                pred_ec2 += 1e6 * mask_ec2
                best_idx = torch.argmin(pred_ec2, dim=1)
                mask_update = torch.zeros_like(mask_ec2)
                mask_update.scatter_(1, best_idx.unsqueeze(1), 1.0)
                mask_ec2 = torch.max(mask_ec2, mask_update)
                x_masked_ec2 = model.mask_layer(x, mask_ec2)
                all_preds[feature_step].append(
                    model.estimate_class_frequency_mc(x, mask_ec2, x_masked_ec2).cpu()
                )

                # Random selection
                rand_scores = torch.rand(len(x), d_in, device=device) - 1e6 * mask_rand
                best_idx_rand = torch.argmax(rand_scores, dim=1)
                mask_update_rand = torch.zeros_like(mask_rand)
                mask_update_rand.scatter_(1, best_idx_rand.unsqueeze(1), 1.0)
                mask_rand = torch.max(mask_rand, mask_update_rand)
                x_masked_rand = model.mask_layer(x, mask_rand)
                all_preds_random[feature_step].append(
                    model.estimate_class_frequency_mc(x, mask_rand, x_masked_rand).cpu()
                )

    y_true = torch.cat(all_y).numpy()

    print("\n" + "="*80)
    print(f"{'Num Features':<15} | {'EC2 Accuracy':<15} | {'EC2 AUROC':<15} | {'Rand Accuracy':<15} | {'Rand AUROC':<15}")
    print("="*80)

    results = []
    for step in range(max_features + 1):
        preds = torch.cat(all_preds[step])
        preds_rand = torch.cat(all_preds_random[step])

        pred_labels = torch.argmax(preds, dim=1).numpy()
        pred_labels_rand = torch.argmax(preds_rand, dim=1).numpy()
        acc      = accuracy_score(y_true, pred_labels)
        acc_rand = accuracy_score(y_true, pred_labels_rand)

        # AUROC: binary uses class-1 probability; multiclass uses OVR macro
        if d_out == 2:
            probs      = preds[:, 1].numpy()
            probs_rand = preds_rand[:, 1].numpy()
            auc_kwargs = {}
        else:
            probs      = preds.numpy()
            probs_rand = preds_rand.numpy()
            auc_kwargs = {"multi_class": "ovr", "average": "macro"}

        try:
            auc_score = roc_auc_score(y_true, probs, **auc_kwargs)
        except ValueError:
            auc_score = float('nan')
        try:
            auc_score_rand = roc_auc_score(y_true, probs_rand, **auc_kwargs)
        except ValueError:
            auc_score_rand = float('nan')

        results.append((step, acc, auc_score, acc_rand, auc_score_rand))
        print(f"{step:<15} | {acc:<15.4f} | {auc_score:<15.4f} | {acc_rand:<15.4f} | {auc_score_rand:<15.4f}")

    print("="*80)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--label_col', type=_parse_label_col, default=-1,
                        help="Label column: name or int index (default: -1 = last column)")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)

    args = parser.parse_args()

    evaluate_ec2(args.checkpoint, args.test_csv, args.label_col, args.gpu, args.hidden, args.dropout)
