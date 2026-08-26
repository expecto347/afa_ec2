"""
Analyze the feature acquisition ORDER of trained EC2 models on Syn1/2/3,
mirroring the CUBE-NM feature-order analysis (evaluate_ec2_cube_nm.py).

Ground truth (INVASE-style, see ec2_sefa.generate_syn_data):
  x11 (index 10) is a gating feature — its sign picks which features define y.
    syn1: x11<0 -> {x1,x2,x11}    x11>=0 -> {x3,x4,x5,x6,x11}
    syn2: x11<0 -> {x1,x2,x11}    x11>=0 -> {x7,x8,x9,x10,x11}
    syn3: x11<0 -> {x3,x4,x5,x6,x11}  x11>=0 -> {x7,x8,x9,x10,x11}
  Optimal strategy: pick x11 FIRST, then the informative features of the branch.

For each (dataset, seed) we run the exact greedy EC2 loop used by
evaluate_ec2_sefa.py and record which feature is picked at each step per test
sample. Results are aggregated across seeds:

  - per-step selection frequency (overall + split by x11 branch)
  - gating-first rate: fraction of samples that pick x11 at step 1
  - steps until x11 is first observed
  - steps to cover ALL ground-truth relevant features (incl. x11), per branch
  - wasted picks: features acquired that are neither x11 nor branch-relevant
    (counted until coverage is complete)

Outputs:
  - logs/syn{N}_feature_order.json   — all statistics
  - syn{N}_feature_order.png         — selection-frequency heatmaps
    (rows x1..x11, cols Step 1..11; overall + branch1 + branch2 panels)

Usage:
    apptainer exec --nv afa_env.sif python -u analyze_syn_feature_order.py \
        --dataset syn1 [--seeds 0 1 2] [--gpu 0]
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ec2_core import EC2Estimator, MaskLayer, Predictor
from ec2_sefa import prepare_data
from evaluate_ec2_sefa import _SYN_RELEVANT

FEAT_NAMES = [f"x{i}" for i in range(1, 12)]
GATING_IDX = 10  # x11

LOGS_ROOT = "/data/users/haozhes/AFA_EC2/logs"


def run_greedy_selection(checkpoint, args):
    """Replicate the greedy EC2 eval loop; return per-step selections [steps, N]."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    temp_ckpt       = torch.load(checkpoint, map_location="cpu")
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
        checkpoint,
        value_network=value_network, predictor=predictor, mask_layer=mask_layer,
        X_train=X_train, y_train=y_train, feat_thresholds=feat_thresholds,
        feat_mean=feat_mean, feat_std=feat_std,
        lr=1e-3, max_features=max_features, eps=0.1,
        strict=False,
    ).to(device)
    model.eval()

    N = X_test_t.shape[0]
    selections = np.zeros((max_features, N), dtype=np.int64)

    with torch.no_grad():
        idx = 0
        for x, _ in test_dl:
            x = x.to(device)
            mask = torch.zeros(len(x), d_in, dtype=x.dtype, device=device)
            for step in range(max_features):
                pred_ec2 = model.value_network(model.mask_layer(x, mask)) + 1e6 * mask
                best = torch.argmin(pred_ec2, dim=1)
                mask_upd = torch.zeros_like(mask).scatter_(1, best.unsqueeze(1), 1.0)
                mask = torch.max(mask, mask_upd)
                selections[step, idx:idx + len(x)] = best.cpu().numpy()
            idx += len(x)

    return selections, X_test_raw


def coverage_stats(dataset, selections, X_test_raw):
    """Steps-to-cover-all-relevant + wasted-pick statistics."""
    N = selections.shape[1]
    max_steps = selections.shape[0]
    branch1 = (X_test_raw[:, 10] < 0).numpy()
    rel_map = _SYN_RELEVANT[dataset]

    # first step at which x11 is observed (1-based; max_steps+1 = never)
    first_gating = np.full(N, max_steps + 1)
    for s in range(max_steps):
        newly = (selections[s] == GATING_IDX) & (first_gating == max_steps + 1)
        first_gating[newly] = s + 1

    # steps until the full relevant set (incl x11) is covered
    cover_steps = np.full(N, max_steps + 1, dtype=int)
    for s in range(max_steps):
        covered_b1 = np.all([(selections[:s+1] == f).any(axis=0) for f in rel_map[True]], axis=0)
        covered_b2 = np.all([(selections[:s+1] == f).any(axis=0) for f in rel_map[False]], axis=0)
        covered = np.where(branch1, covered_b1, covered_b2)
        newly = covered & (cover_steps == max_steps + 1)
        cover_steps[newly] = s + 1

    # wasted picks before coverage: features neither x11 nor branch-relevant
    wasted = np.zeros(N, dtype=int)
    relevant_all = sorted(set(rel_map[True] + rel_map[False]))
    irrelevant = [f for f in range(11) if f not in relevant_all and f != GATING_IDX]
    for s in range(max_steps):
        active = cover_steps > s + 1  # still not covered before this step completed
        wasted += active & np.isin(selections[s], irrelevant)

    return {
        "n_branch1": int(branch1.sum()),
        "n_branch2": int((~branch1).sum()),
        "first_gating_step": {"mean": float(first_gating[first_gating <= max_steps].mean()),
                              "pct_first": float((first_gating == 1).mean()),
                              "pct_within2": float((first_gating <= 2).mean())},
        "cover_steps": {
            "branch1": {"mean": float(cover_steps[branch1][cover_steps[branch1] <= max_steps].mean()),
                        "pct_covered": float((cover_steps[branch1] <= max_steps).mean()),
                        "theoretical_min": len(rel_map[True])},
            "branch2": {"mean": float(cover_steps[~branch1][cover_steps[~branch1] <= max_steps].mean()),
                        "pct_covered": float((cover_steps[~branch1] <= max_steps).mean()),
                        "theoretical_min": len(rel_map[False])},
        },
        "wasted_picks_mean": float(wasted.mean()),
        "wasted_picks_frac_zero": float((wasted == 0).mean()),
    }


def per_step_counts(selections_list, branch_masks_list):
    """Aggregate selection counts across seeds: [feature, step] overall and per branch."""
    total = sum(s.shape[1] for s in selections_list)
    counts_all = np.zeros((11, 11))
    counts_b1  = np.zeros((11, 11))
    counts_b2  = np.zeros((11, 11))
    n_b1 = n_b2 = 0
    for sel, bmask in zip(selections_list, branch_masks_list):
        n_b1 += int(bmask.sum()); n_b2 += int((~bmask).sum())
        for s in range(sel.shape[0]):
            np.add.at(counts_all, (sel[s], s), 1)
            np.add.at(counts_b1,  (sel[s, bmask], s), 1)
            np.add.at(counts_b2,  (sel[s, ~bmask], s), 1)
    return counts_all, counts_b1, counts_b2, total, n_b1, n_b2


def _recursive_mean(entries):
    """Average nested dict/list-of-numbers structures across seeds."""
    sample = entries[0]
    if isinstance(sample, dict):
        return {k: _recursive_mean([e[k] for e in entries]) for k in sample}
    if isinstance(sample, (list, tuple)):
        return [_recursive_mean([e[i] for e in entries]) for i in range(len(sample))]
    return float(np.mean(entries))


def print_tables(dataset, c_all, c_b1, c_b2, total, n_b1, n_b2):
    print(f"\n{'='*92}\nDataset {dataset} — per-step most-picked feature (freq = fraction of samples)\n{'='*92}")
    print(f"{'Step':<6} | {'overall':<24} | {'branch1 x11<0':<24} | {'branch2 x11>=0':<24}")
    print("-" * 92)
    for s in range(11):
        def top(c):
            f = int(np.argmax(c[:, s]))
            return f"{FEAT_NAMES[f]} ({c[f, s] / c[:, s].sum():.3f})"
        print(f"{s+1:<6} | {top(c_all):<24} | {top(c_b1):<24} | {top(c_b2):<24}")
    print(f"\n(step-1 = {FEAT_NAMES[GATING_IDX]} fraction: overall {c_all[GATING_IDX,0]/total:.3f}, "
          f"branch1 {c_b1[GATING_IDX,0]/n_b1:.3f}, branch2 {c_b2[GATING_IDX,0]/n_b2:.3f})")

    # full frequency matrix (overall)
    print(f"\nPer-step selection frequency (overall, fraction per step column):")
    print(f"{'Feat':<6} | " + " ".join(f"{s+1:>6}" for s in range(11)))
    print("-" * 6 + "-+-" + "-" * 6 * 11)
    for f in range(11):
        row = " ".join(f"{c_all[f, s] / c_all[:, s].sum():6.3f}" for s in range(11))
        print(f"{FEAT_NAMES[f]:<6} | {row}")


def save_heatmap(dataset, c_all, c_b1, c_b2, path):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))
    titles = ["Overall", "Branch 1 (x11 < 0)", "Branch 2 (x11 >= 0)"]
    for ax, c, title in zip(axes, [c_all, c_b1, c_b2], titles):
        frac = c / np.maximum(1, c.sum(axis=0, keepdims=True))
        im = ax.imshow(frac, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(11))
        ax.set_xticklabels([f"S{i+1}" for i in range(11)], fontsize=8)
        ax.set_yticks(range(11))
        ax.set_yticklabels(FEAT_NAMES, fontsize=9)
        ax.set_xlabel("Acquisition Step")
        ax.set_ylabel("Feature")
        ax.set_title(f"{dataset}: {title}")
        fig.colorbar(im, ax=ax, label="Selection Fraction")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Heatmap saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["syn1", "syn2", "syn3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--syn_seed", type=int, default=8139)
    args = parser.parse_args()

    selections_list, branch_masks_list, cov_stats_list = [], [], []
    for seed in args.seeds:
        ckpt = os.path.join(LOGS_ROOT, f"ec2_{args.dataset}_seed{seed}",
                            "ec2_train", "checkpoints", "best_val_perf_model.ckpt")
        if not os.path.exists(ckpt):
            print(f"WARNING: checkpoint not found, skipping: {ckpt}")
            continue
        print(f"Evaluating {ckpt}")
        selections, X_test_raw = run_greedy_selection(ckpt, args)
        selections_list.append(selections)
        branch_masks_list.append((X_test_raw[:, 10] < 0).numpy())
        cov_stats_list.append(coverage_stats(args.dataset, selections, X_test_raw))

    if not selections_list:
        raise SystemExit("No checkpoints found for the requested seeds")

    c_all, c_b1, c_b2, total, n_b1, n_b2 = per_step_counts(selections_list, branch_masks_list)
    print_tables(args.dataset, c_all, c_b1, c_b2, total, n_b1, n_b2)

    print(f"\n--- Coverage / gating statistics (aggregated over {len(selections_list)} seed(s)) ---")
    agg = _recursive_mean(cov_stats_list)
    print(json.dumps(agg, indent=2))

    out_json = os.path.join(LOGS_ROOT, f"{args.dataset}_feature_order.json")
    with open(out_json, "w") as f:
        json.dump({"dataset": args.dataset, "seeds": args.seeds, "n_samples_per_seed": total // len(selections_list),
                   "selection_counts": {"overall": c_all.tolist(), "branch1": c_b1.tolist(), "branch2": c_b2.tolist()},
                   "stats": agg}, f, indent=2)
    print(f"JSON saved to {out_json}")

    save_heatmap(args.dataset, c_all, c_b1, c_b2,
                 os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f"{args.dataset}_feature_order.png"))


if __name__ == "__main__":
    main()
