"""
Evaluation script for EC2 checkpoints trained with ec2_sefa.py.

d_in and d_out are inferred automatically from the checkpoint's registered buffers.

Usage:
    apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
        --checkpoint logs/ec2_syn1_trial_0/ec2_train/checkpoints/best_val_perf_model.ckpt \
        --dataset syn1

Note: --syn_seed must match the value used during training (default 8139).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from ec2_core import EC2Estimator, MaskLayer, Predictor
from ec2_sefa import SEFA_DEFAULT_DATA_DIR, prepare_data


# Relevant feature indices (0-based) per dataset and branch.
# x_11 is index 10.  Branch is determined by x_11 < 0.
_SYN_RELEVANT = {
    "syn1": {True:  [0, 1, 10],          # x_1, x_2, x_11
             False: [2, 3, 4, 5, 10]},   # x_3-x_6, x_11
    "syn2": {True:  [0, 1, 10],          # x_1, x_2, x_11
             False: [6, 7, 8, 9, 10]},   # x_7-x_10, x_11
    "syn3": {True:  [2, 3, 4, 5, 10],   # x_3-x_6, x_11
             False: [6, 7, 8, 9, 10]},   # x_7-x_10, x_11
}


def steps_to_cover_relevant(dataset, X_test_raw, mask_history):
    """
    For each test instance count the acquisition step at which the mask
    first covers ALL ground-truth relevant features (including x_11).

    Args:
        dataset:       "syn1" | "syn2" | "syn3"
        X_test_raw:    un-normalized test features [N, 11], used only to
                       read x_11 sign (index 10)
        mask_history:  list of [N, d_in] bool tensors, one per step
                       (mask_history[s] is the mask after step s)

    Returns:
        steps: int tensor [N], step index (1-based) when coverage is complete.
               max_features+1 means never covered within the budget.
    """
    relevant_map = _SYN_RELEVANT[dataset]
    N = X_test_raw.shape[0]
    branch1 = (X_test_raw[:, 10] < 0)               # [N] bool
    max_steps = len(mask_history)

    steps = torch.full((N,), max_steps + 1, dtype=torch.long)

    for s, mask in enumerate(mask_history, start=1):
        # relevant feature set per instance
        rel_b1 = torch.tensor(relevant_map[True],  dtype=torch.long)
        rel_b2 = torch.tensor(relevant_map[False], dtype=torch.long)

        covered_b1 = mask[:, rel_b1].all(dim=1)  # [N]
        covered_b2 = mask[:, rel_b2].all(dim=1)

        covered = torch.where(branch1, covered_b1, covered_b2)  # [N]
        newly_done = covered & (steps == max_steps + 1)
        steps[newly_done] = s

    return steps


def evaluate_ec2(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    temp_ckpt       = torch.load(args.checkpoint, map_location="cpu")
    X_train         = temp_ckpt["state_dict"]["X_train"]
    y_train         = temp_ckpt["state_dict"]["y_train"]
    feat_thresholds = temp_ckpt["state_dict"]["feat_thresholds"]
    feat_mean       = temp_ckpt["state_dict"]["feat_mean"].numpy()
    feat_std        = temp_ckpt["state_dict"]["feat_std"].numpy()

    d_in  = X_train.shape[1]
    d_out = int(y_train.max().item()) + 1
    max_features = d_in
    print(f"Inferred from checkpoint: d_in={d_in}, d_out={d_out}")

    # Load test split with same normalization as training.
    # For syn datasets we also keep the raw (un-normalized) features to read x_11 sign.
    _, _, _, _, X_test_t, y_test_t, feat_mean_np, feat_std_np = prepare_data(args)

    # Reconstruct raw X_test for ground-truth branch detection (syn only)
    if args.dataset.startswith("syn"):
        X_test_raw = X_test_t * torch.tensor(feat_std_np + 1e-8) + torch.tensor(feat_mean_np)
    else:
        X_test_raw = None

    test_dl = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=256, shuffle=False, pin_memory=True, num_workers=4,
    )

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
        lr=1e-3, max_features=max_features, eps=0.1,
        strict=False,
    ).to(device)
    model.eval()

    all_preds        = [[] for _ in range(max_features + 1)]
    all_preds_random = [[] for _ in range(max_features + 1)]
    all_y = []
    # Per-batch mask history for coverage metric (syn datasets only)
    # masks_ec2_batches[b][s] = mask tensor after step s for batch b
    masks_ec2_batches   = []
    masks_rand_batches  = []

    print("Evaluating EC2 greedy vs. random baseline...")
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            mask      = torch.zeros(len(x), d_in, dtype=x.dtype, device=device)
            mask_rand = torch.zeros(len(x), d_in, dtype=x.dtype, device=device)

            x_masked = model.mask_layer(x, mask)
            p0 = model.estimate_class_frequency_mc(x, mask, x_masked).cpu()
            all_preds[0].append(p0)
            all_preds_random[0].append(p0)
            all_y.append(y.cpu())

            batch_masks_ec2  = []
            batch_masks_rand = []

            for step in range(1, max_features + 1):
                # EC2 greedy
                pred_ec2  = model.value_network(model.mask_layer(x, mask)) + 1e6 * mask
                best      = torch.argmin(pred_ec2, dim=1)
                mask_upd  = torch.zeros_like(mask).scatter_(1, best.unsqueeze(1), 1.0)
                mask      = torch.max(mask, mask_upd)
                batch_masks_ec2.append(mask.bool().cpu())
                all_preds[step].append(
                    model.estimate_class_frequency_mc(x, mask, model.mask_layer(x, mask)).cpu()
                )

                # Random baseline
                rand_s    = torch.rand(len(x), d_in, device=device) - 1e6 * mask_rand
                best_r    = torch.argmax(rand_s, dim=1)
                mask_upd_r = torch.zeros_like(mask_rand).scatter_(1, best_r.unsqueeze(1), 1.0)
                mask_rand  = torch.max(mask_rand, mask_upd_r)
                batch_masks_rand.append(mask_rand.bool().cpu())
                all_preds_random[step].append(
                    model.estimate_class_frequency_mc(x, mask_rand, model.mask_layer(x, mask_rand)).cpu()
                )

            masks_ec2_batches.append(batch_masks_ec2)
            masks_rand_batches.append(batch_masks_rand)

    y_true     = torch.cat(all_y).numpy()
    auc_kwargs = {} if d_out == 2 else {"multi_class": "ovr", "average": "macro"}

    print("\n" + "=" * 85)
    print(f"{'Features':<12} | {'EC2 Acc':<12} | {'EC2 AUROC':<12} | {'Rand Acc':<12} | {'Rand AUROC':<12}")
    print("=" * 85)

    for step in range(max_features + 1):
        preds  = torch.cat(all_preds[step])
        p_rand = torch.cat(all_preds_random[step])

        acc   = accuracy_score(y_true, torch.argmax(preds, dim=1).numpy())
        acc_r = accuracy_score(y_true, torch.argmax(p_rand, dim=1).numpy())

        prob   = preds[:, 1].numpy()  if d_out == 2 else preds.numpy()
        prob_r = p_rand[:, 1].numpy() if d_out == 2 else p_rand.numpy()
        try:
            auc   = roc_auc_score(y_true, prob,   **auc_kwargs)
        except ValueError:
            auc   = float("nan")
        try:
            auc_r = roc_auc_score(y_true, prob_r, **auc_kwargs)
        except ValueError:
            auc_r = float("nan")

        print(f"{step:<12} | {acc:<12.4f} | {auc:<12.4f} | {acc_r:<12.4f} | {auc_r:<12.4f}")

    print("=" * 85)

    # --- Coverage metric (synthetic datasets only) ---
    if args.dataset.startswith("syn"):
        # Concatenate per-batch mask histories: shape [max_features, N, d_in]
        # masks_ec2_batches[b][s] → cat over b for each s
        num_steps = max_features
        masks_ec2_full  = [torch.cat([masks_ec2_batches[b][s]  for b in range(len(masks_ec2_batches))],  dim=0)
                           for s in range(num_steps)]
        masks_rand_full = [torch.cat([masks_rand_batches[b][s] for b in range(len(masks_rand_batches))], dim=0)
                           for s in range(num_steps)]

        steps_ec2  = steps_to_cover_relevant(args.dataset, X_test_raw, masks_ec2_full)
        steps_rand = steps_to_cover_relevant(args.dataset, X_test_raw, masks_rand_full)

        # Compute per-branch statistics
        branch1 = (X_test_raw[:, 10] < 0)
        rel_map = _SYN_RELEVANT[args.dataset]
        opt_b1  = len(rel_map[True])   # theoretical minimum steps (branch 1)
        opt_b2  = len(rel_map[False])  # theoretical minimum steps (branch 2)

        def _summary(steps_t, mask):
            s = steps_t[mask]
            within_budget = (s <= num_steps).float().mean().item()
            mean_steps = s[s <= num_steps].float().mean().item() if (s <= num_steps).any() else float("nan")
            return mean_steps, within_budget

        mean_ec2_b1,  cov_ec2_b1  = _summary(steps_ec2,  branch1)
        mean_ec2_b2,  cov_ec2_b2  = _summary(steps_ec2,  ~branch1)
        mean_rand_b1, cov_rand_b1 = _summary(steps_rand, branch1)
        mean_rand_b2, cov_rand_b2 = _summary(steps_rand, ~branch1)
        mean_ec2_all  = steps_ec2[steps_ec2   <= num_steps].float().mean().item()
        mean_rand_all = steps_rand[steps_rand  <= num_steps].float().mean().item()

        n_b1 = branch1.sum().item()
        n_b2 = (~branch1).sum().item()

        print(f"\n--- Steps to acquire all relevant features (lower = better) ---")
        print(f"Dataset: {args.dataset}   N={len(X_test_raw)}  "
              f"(branch1 x₁₁<0: {n_b1},  branch2 x₁₁≥0: {n_b2})")
        print(f"Theoretical minimum: branch1={opt_b1}, branch2={opt_b2}")
        print(f"{'Branch':<12} | {'EC2 mean steps':<18} | {'EC2 % covered':<16} | "
              f"{'Rand mean steps':<18} | {'Rand % covered':<16}")
        print("-" * 90)
        print(f"{'branch1':<12} | {mean_ec2_b1:<18.2f} | {cov_ec2_b1:<16.3f} | "
              f"{mean_rand_b1:<18.2f} | {cov_rand_b1:<16.3f}")
        print(f"{'branch2':<12} | {mean_ec2_b2:<18.2f} | {cov_ec2_b2:<16.3f} | "
              f"{mean_rand_b2:<18.2f} | {cov_rand_b2:<16.3f}")
        print(f"{'overall':<12} | {mean_ec2_all:<18.2f} | {'':16} | "
              f"{mean_rand_all:<18.2f} |")
        print("-" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--dataset",       type=str, required=True, choices=["syn1", "syn2", "syn3", "metabric"])
    parser.add_argument("--sefa_data_dir", type=str, default=SEFA_DEFAULT_DATA_DIR)
    parser.add_argument("--syn_seed",      type=int, default=8139,
                        help="Must match the value used during training")
    parser.add_argument("--gpu",     type=int,   default=0)
    parser.add_argument("--hidden",  type=int,   default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    evaluate_ec2(args)
