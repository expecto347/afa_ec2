"""
Evaluate a trained CUBE-NM EC2 kNN checkpoint.

Reports accuracy at benchmark budget levels [0, 3, 5, 7, 10, 14] for both
EC2 greedy acquisition and a random baseline. Test set is generated with
seed = train_seed + 100 (separate from training data).
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from afabench.common.datasets.datasets import AFAContextDataset
from ec2_core import Predictor, EC2Estimator, MaskLayer
from ec2_cube_nm import CubeNMDataset, CUBE_NM_KWARGS

BENCHMARK_BUDGETS = [0, 3, 5, 7, 10, 14]
D_IN    = 55
D_OUT   = 8
HIDDEN  = 512
DROPOUT = 0.3


def evaluate(checkpoint_path, gpu, seed, n_test, batch_size, output_plot):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    feat_mean       = ckpt["state_dict"]["feat_mean"].numpy()
    feat_std        = ckpt["state_dict"]["feat_std"].numpy()
    X_train         = ckpt["state_dict"]["X_train"]
    y_train         = ckpt["state_dict"]["y_train"]
    feat_thresholds = ckpt["state_dict"]["feat_thresholds"]

    # Test set: separate seed from training
    test_afa = AFAContextDataset(n_samples=n_test, seed=seed + 100, **CUBE_NM_KWARGS)
    test_data = CubeNMDataset(test_afa, mean=feat_mean, std=feat_std)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Test set: {len(test_data)} samples")

    predictor  = Predictor(D_IN, D_OUT, HIDDEN, DROPOUT)
    mask_layer = MaskLayer(mask_size=D_IN, append=True)
    value_network = nn.Sequential(
        nn.Linear(D_IN * 2, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, D_IN),
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
        max_features=D_IN,
        eps=0.0,
    ).to(device)
    model.eval()

    max_budget = max(BENCHMARK_BUDGETS)
    all_preds_ec2  = [[] for _ in range(max_budget + 1)]
    all_preds_rand = [[] for _ in range(max_budget + 1)]
    all_y = []

    print("Evaluating...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            mask_ec2  = torch.zeros(len(x), D_IN, dtype=x.dtype, device=device)
            mask_rand = torch.zeros(len(x), D_IN, dtype=x.dtype, device=device)

            # Budget 0: no features
            x_masked = model.mask_layer(x, mask_ec2)
            pred0 = model.estimate_class_frequency_mc(x, mask_ec2, x_masked)
            all_preds_ec2[0].append(pred0.cpu())
            all_preds_rand[0].append(pred0.cpu())
            all_y.append(y.cpu())

            for step in range(1, max_budget + 1):
                # EC2 greedy
                x_masked_ec2 = model.mask_layer(x, mask_ec2)
                scores = model.value_network(x_masked_ec2)
                scores += 1e6 * mask_ec2                      # penalize already-acquired
                best = torch.argmin(scores, dim=1)
                upd = torch.zeros_like(mask_ec2)
                upd.scatter_(1, best.unsqueeze(1), 1.0)
                mask_ec2 = torch.max(mask_ec2, upd)
                x_masked_ec2 = model.mask_layer(x, mask_ec2)
                all_preds_ec2[step].append(
                    model.estimate_class_frequency_mc(x, mask_ec2, x_masked_ec2).cpu()
                )

                # Random baseline
                rand_scores = torch.rand(len(x), D_IN, device=device)
                rand_scores -= 1e6 * mask_rand
                best_rand = torch.argmax(rand_scores, dim=1)
                upd_rand = torch.zeros_like(mask_rand)
                upd_rand.scatter_(1, best_rand.unsqueeze(1), 1.0)
                mask_rand = torch.max(mask_rand, upd_rand)
                x_masked_rand = model.mask_layer(x, mask_rand)
                all_preds_rand[step].append(
                    model.estimate_class_frequency_mc(x, mask_rand, x_masked_rand).cpu()
                )

    y_true = torch.cat(all_y).numpy()

    print(f"\n{'Budget':<10} | {'EC2 Acc':<12} | {'Random Acc':<12}")
    print("=" * 40)
    ec2_accs, rand_accs = [], []
    for budget in range(max_budget + 1):
        p_ec2  = torch.argmax(torch.cat(all_preds_ec2[budget]),  dim=1).numpy()
        p_rand = torch.argmax(torch.cat(all_preds_rand[budget]), dim=1).numpy()
        acc_ec2  = accuracy_score(y_true, p_ec2)
        acc_rand = accuracy_score(y_true, p_rand)
        ec2_accs.append(acc_ec2)
        rand_accs.append(acc_rand)
        if budget in BENCHMARK_BUDGETS:
            print(f"{budget:<10} | {acc_ec2:<12.4f} | {acc_rand:<12.4f}")

    # Plot
    budgets = list(range(max_budget + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(budgets, ec2_accs,  marker="o", label="EC2 (kNN)")
    plt.plot(budgets, rand_accs, marker="s", linestyle="--", label="Random")
    plt.axhline(y=1/8, color="gray", linestyle=":", label="Chance (12.5%)")
    plt.xlabel("Features Acquired")
    plt.ylabel("Accuracy")
    plt.title("CUBE-NM: EC2 kNN vs Random Baseline")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"\nPlot saved to {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu",         type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=0,
                        help="Must match the seed used during training")
    parser.add_argument("--n_test",      type=int,   default=1000)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--output_plot", type=str,   default="cube_nm_eval.png")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.gpu, args.seed, args.n_test, args.batch_size, args.output_plot)
