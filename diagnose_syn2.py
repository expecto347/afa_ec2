"""
Diagnostics for branch-2 failure on Syn2.

Diagnosis 1 — Force x₁₁ first:
    Run evaluation with feature 10 (x₁₁) forced as the first acquisition.
    If branch-2 mean steps drops significantly from 10.29 → ~6, routing is
    the sole cause of failure.

Diagnosis 2 — Step-0 value_network scores:
    Print mean EC2 score per feature when mask=all-zeros, separately for
    branch-1 and branch-2 test samples.  Confirms whether the value network
    already prefers x₁,x₂ at step 0, or whether bias emerges later.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ec2_core import EC2Estimator, MaskLayer, Predictor
from ec2_sefa import SEFA_DEFAULT_DATA_DIR, prepare_data
from evaluate_ec2_sefa import _SYN_RELEVANT, steps_to_cover_relevant


ROUTING_FEATURE = 10  # x₁₁, 0-based


def load_model(args, device):
    temp_ckpt = torch.load(args.checkpoint, map_location="cpu")
    X_train = temp_ckpt["state_dict"]["X_train"]
    y_train = temp_ckpt["state_dict"]["y_train"]
    feat_thresholds = temp_ckpt["state_dict"]["feat_thresholds"]
    feat_mean = temp_ckpt["state_dict"]["feat_mean"].numpy()
    feat_std  = temp_ckpt["state_dict"]["feat_std"].numpy()

    d_in  = X_train.shape[1]
    d_out = int(y_train.max().item()) + 1
    print(f"d_in={d_in}  d_out={d_out}")

    mask_layer    = MaskLayer(mask_size=d_in, append=True)
    predictor     = Predictor(d_in, d_out, args.hidden, args.dropout)
    value_network = nn.Sequential(
        nn.Linear(d_in * 2, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, d_in),
    )

    model = EC2Estimator.load_from_checkpoint(
        args.checkpoint,
        value_network=value_network, predictor=predictor, mask_layer=mask_layer,
        X_train=X_train, y_train=y_train, feat_thresholds=feat_thresholds,
        feat_mean=feat_mean, feat_std=feat_std,
        lr=1e-3, max_features=d_in, eps=0.1,
    ).to(device)
    model.eval()
    return model, d_in, feat_mean, feat_std


def run_diag1_forced_routing(model, test_dl, X_test_raw, d_in, device):
    """Force x₁₁ at step 0, then run greedy EC2 for remaining steps."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 1: Force x₁₁ (feature 10) as first acquisition")
    print("=" * 70)

    masks_forced_batches = []
    masks_ec2_batches    = []

    with torch.no_grad():
        for x, _ in test_dl:
            x = x.to(device)
            B = len(x)

            # --- Forced policy ---
            mask_f = torch.zeros(B, d_in, dtype=x.dtype, device=device)
            mask_f[:, ROUTING_FEATURE] = 1.0          # force x₁₁ first
            batch_masks_forced = [mask_f.bool().cpu()]

            for _ in range(1, d_in):
                pred = model.value_network(model.mask_layer(x, mask_f)) + 1e6 * mask_f
                best = torch.argmin(pred, dim=1)
                one_hot = torch.zeros_like(mask_f).scatter_(1, best.unsqueeze(1), 1.0)
                mask_f = torch.max(mask_f, one_hot)
                batch_masks_forced.append(mask_f.bool().cpu())

            masks_forced_batches.append(batch_masks_forced)

            # --- Normal EC2 (for reference) ---
            mask_e = torch.zeros(B, d_in, dtype=x.dtype, device=device)
            batch_masks_ec2 = []
            for _ in range(d_in):
                pred = model.value_network(model.mask_layer(x, mask_e)) + 1e6 * mask_e
                best = torch.argmin(pred, dim=1)
                one_hot = torch.zeros_like(mask_e).scatter_(1, best.unsqueeze(1), 1.0)
                mask_e = torch.max(mask_e, one_hot)
                batch_masks_ec2.append(mask_e.bool().cpu())

            masks_ec2_batches.append(batch_masks_ec2)

    def concat_history(batches):
        return [
            torch.cat([batches[b][s] for b in range(len(batches))], dim=0)
            for s in range(d_in)
        ]

    masks_forced_full = concat_history(masks_forced_batches)
    masks_ec2_full    = concat_history(masks_ec2_batches)

    steps_forced = steps_to_cover_relevant("syn2", X_test_raw, masks_forced_full)
    steps_ec2    = steps_to_cover_relevant("syn2", X_test_raw, masks_ec2_full)

    branch1 = (X_test_raw[:, ROUTING_FEATURE] < 0)
    rel_map  = _SYN_RELEVANT["syn2"]
    opt_b1   = len(rel_map[True])
    opt_b2   = len(rel_map[False])

    def _stats(steps_t, mask):
        s = steps_t[mask]
        within = (s <= d_in).float().mean().item()
        mean   = s[s <= d_in].float().mean().item() if (s <= d_in).any() else float("nan")
        return mean, within

    print(f"Theoretical minimum: branch1={opt_b1}, branch2={opt_b2}")
    print(f"\n{'Branch':<10} | {'Forced-x₁₁ steps':<18} | {'EC2 steps':<12}")
    print("-" * 46)
    for lbl, msk in [("branch1", branch1), ("branch2", ~branch1)]:
        mf, _ = _stats(steps_forced, msk)
        me, _ = _stats(steps_ec2,    msk)
        print(f"{lbl:<10} | {mf:<18.2f} | {me:<12.2f}")
    mf_all = steps_forced[steps_forced <= d_in].float().mean().item()
    me_all = steps_ec2[steps_ec2 <= d_in].float().mean().item()
    print(f"{'overall':<10} | {mf_all:<18.2f} | {me_all:<12.2f}")
    print("-" * 46)
    print("Interpretation: if forced-branch2 << EC2-branch2, routing is the main issue.")


def run_diag2_step0_scores(model, test_dl, X_test_raw, d_in, device):
    """Print mean value_network output at step 0, split by branch."""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 2: Value-network scores at step 0 (mask=all-zeros)")
    print("=" * 70)

    all_scores = []  # [N, d_in]
    with torch.no_grad():
        for x, _ in test_dl:
            x = x.to(device)
            mask = torch.zeros(len(x), d_in, dtype=x.dtype, device=device)
            scores = model.value_network(model.mask_layer(x, mask))  # [B, d_in]
            all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, dim=0)  # [N, d_in]
    branch1    = (X_test_raw[:, ROUTING_FEATURE] < 0)

    scores_b1 = all_scores[branch1].mean(dim=0)   # [d_in]
    scores_b2 = all_scores[~branch1].mean(dim=0)  # [d_in]
    scores_all = all_scores.mean(dim=0)            # [d_in]

    # Rank (lower score = model prefers this feature under argmin)
    rank_b1  = scores_b1.argsort().argsort() + 1
    rank_b2  = scores_b2.argsort().argsort() + 1
    rank_all = scores_all.argsort().argsort() + 1

    print(f"{'Feature':<10} | {'Score (all)':<14} | {'Rank (all)':<12} | "
          f"{'Score (b1)':<14} | {'Rank (b1)':<10} | "
          f"{'Score (b2)':<14} | {'Rank (b2)':<10}")
    print("-" * 92)
    for i in range(d_in):
        marker = " ← x₁₁ (routing)" if i == ROUTING_FEATURE else ""
        print(f"x{i+1:<9} | {scores_all[i]:<14.4f} | {rank_all[i].item():<12} | "
              f"{scores_b1[i]:<14.4f} | {rank_b1[i].item():<10} | "
              f"{scores_b2[i]:<14.4f} | {rank_b2[i].item():<10}{marker}")
    print("-" * 92)
    print("Lower score = model prefers this feature (argmin). Rank 1 = most preferred.")
    print("\nKey question: is x₁₁ (feature 10) ranked near the top overall?")
    print("If x₁,x₂ (features 0,1) dominate at step 0, bias starts immediately.")
    print("If all scores are nearly equal, bias only emerges at step 1+.")

    # Also show step-1 scores AFTER acquiring x₁ (feature 0) to see how
    # x₁₁ and x₂ compare — the critical moment where branch-2 gets derailed.
    print("\n--- Step-1 scores after forcing x₁ (feature 0) ---")
    step1_scores = []
    with torch.no_grad():
        for x, _ in test_dl:
            x = x.to(device)
            mask = torch.zeros(len(x), d_in, dtype=x.dtype, device=device)
            mask[:, 0] = 1.0  # x₁ already observed
            scores = model.value_network(model.mask_layer(x, mask))  # [B, d_in]
            step1_scores.append(scores.cpu())

    step1_scores = torch.cat(step1_scores, dim=0)
    s1_b1  = step1_scores[branch1].mean(dim=0)
    s1_b2  = step1_scores[~branch1].mean(dim=0)
    s1_all = step1_scores.mean(dim=0)
    r1_all = s1_all.argsort().argsort() + 1
    r1_b1  = s1_b1.argsort().argsort() + 1
    r1_b2  = s1_b2.argsort().argsort() + 1

    print(f"{'Feature':<10} | {'Score (all)':<14} | {'Rank (all)':<12} | "
          f"{'Score (b1)':<14} | {'Rank (b1)':<10} | "
          f"{'Score (b2)':<14} | {'Rank (b2)':<10}")
    print("-" * 92)
    for i in range(d_in):
        if i == 0:
            continue  # already observed
        marker = " ← x₁₁" if i == ROUTING_FEATURE else (
                 " ← x₂  (branch1 key)" if i == 1 else "")
        print(f"x{i+1:<9} | {s1_all[i]:<14.4f} | {r1_all[i].item():<12} | "
              f"{s1_b1[i]:<14.4f} | {r1_b1[i].item():<10} | "
              f"{s1_b2[i]:<14.4f} | {r1_b2[i].item():<10}{marker}")
    print("-" * 92)
    print("If x₂ (feature 1) is ranked #1 for branch-2 samples, that confirms "
          "the step-1 bias hypothesis.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset",    type=str, default="syn2")
    parser.add_argument("--sefa_data_dir", type=str, default=SEFA_DEFAULT_DATA_DIR)
    parser.add_argument("--syn_seed",   type=int, default=8139)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--hidden",     type=int, default=512)
    parser.add_argument("--dropout",    type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, d_in, feat_mean, feat_std = load_model(args, device)

    _, _, _, _, X_test_t, y_test_t, feat_mean_np, feat_std_np = prepare_data(args)
    X_test_raw = X_test_t * torch.tensor(feat_std_np + 1e-8) + torch.tensor(feat_mean_np)

    test_dl = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=256, shuffle=False, pin_memory=True, num_workers=4,
    )

    run_diag2_step0_scores(model, test_dl, X_test_raw, d_in, device)
    run_diag1_forced_routing(model, test_dl, X_test_raw, d_in, device)


if __name__ == "__main__":
    main()
