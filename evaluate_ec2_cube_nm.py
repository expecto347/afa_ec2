"""
Evaluate a trained CUBE-NM EC2 kNN checkpoint.

Reports accuracy at every budget from 0 to D_IN (55) — acquisition only stops
after all features have been used — for both EC2 greedy acquisition and a
random baseline. Test set is generated with seed = train_seed + 100 (separate
from training data).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from afabench.common.datasets.datasets import AFAContextDataset
from ec2_core import Predictor, EC2Estimator, MaskLayer
from ec2_cube_nm import CubeNMDataset, CUBE_NM_KWARGS
from eval_utils import run_dir_from_checkpoint, save_eval_results, auc_under_curve, append_to_results_log

D_IN    = 55
D_OUT   = 8
HIDDEN  = 512
DROPOUT = 0.3

# Context one-hot features occupy indices [0, n_contexts): observing any one of
# them determines the context and thus which 10-dim block is informative.
# Optimal policy: pick 1 context feature first, then block features.
N_CONTEXTS = CUBE_NM_KWARGS["n_contexts"]
CONTEXT_FEATURES = list(range(N_CONTEXTS))


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

    # Infer hidden sizes from the checkpoint so hyperparameter variants
    # (e.g. --hidden 1024) evaluate correctly.
    hidden_vn = ckpt["state_dict"]["value_network.0.weight"].shape[0]
    hidden_p  = ckpt["state_dict"]["predictor.net.0.weight"].shape[0]

    # Test set: separate seed from training
    test_afa = AFAContextDataset(n_samples=n_test, seed=seed + 100, **CUBE_NM_KWARGS)
    test_data = CubeNMDataset(test_afa, mean=feat_mean, std=feat_std)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Test set: {len(test_data)} samples")

    predictor  = Predictor(D_IN, D_OUT, hidden_p, DROPOUT)
    mask_layer = MaskLayer(mask_size=D_IN, append=True)
    value_network = nn.Sequential(
        nn.Linear(D_IN * 2, hidden_vn), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(hidden_vn, hidden_vn), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(hidden_vn, D_IN),
    )

    # strict=False: global_ec2_cache and left_float_train are registered as buffers
    # in on_fit_start() (not __init__), so they exist in the saved state_dict but
    # are not pre-registered at load time.
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
        strict=False,
    ).to(device)
    model.eval()

    # Acquire features until all D_IN features have been used
    max_budget = D_IN
    all_preds_ec2  = [[] for _ in range(max_budget + 1)]
    all_preds_rand = [[] for _ in range(max_budget + 1)]
    all_y = []
    # selection_counts[feat, step-1] = number of times feat was chosen at that step
    selection_counts = np.zeros((D_IN, max_budget), dtype=np.int64)
    # context_first_step[i] = acquisition step (1-based) at which sample i first
    # selected a context feature (NaN if never, within max_budget)
    context_first_step = np.full(len(test_data), np.nan)

    print("Evaluating...")
    with torch.no_grad():
        for batch_start in range(0, len(test_data), batch_size):
            x = test_data.features[batch_start:batch_start + batch_size].to(device)
            y = test_data.labels[batch_start:batch_start + batch_size].to(device)
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
                # record selection counts
                best_np = best.cpu().numpy()
                for feat_idx in best_np:
                    selection_counts[feat_idx, step - 1] += 1
                # record first step at which each sample picks a context feature
                is_context = np.isin(best_np, CONTEXT_FEATURES)
                local_sel = batch_start + np.arange(len(x))[is_context]
                step_filled = np.isnan(context_first_step[local_sel])
                context_first_step[local_sel[step_filled]] = step
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
    results = []
    for budget in range(max_budget + 1):
        p_ec2  = torch.argmax(torch.cat(all_preds_ec2[budget]),  dim=1).numpy()
        p_rand = torch.argmax(torch.cat(all_preds_rand[budget]), dim=1).numpy()
        acc_ec2  = accuracy_score(y_true, p_ec2)
        acc_rand = accuracy_score(y_true, p_rand)
        ec2_accs.append(acc_ec2)
        rand_accs.append(acc_rand)
        results.append((budget, acc_ec2, acc_rand))
        print(f"{budget:<10} | {acc_ec2:<12.4f} | {acc_rand:<12.4f}")

    # Accuracy plot
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

    # Feature selection heatmap
    # Normalize each step column so values are fractions [0, 1]
    step_totals = selection_counts.sum(axis=0, keepdims=True).clip(min=1)
    sel_freq = selection_counts / step_totals  # shape: (D_IN, max_budget)

    # Only keep features that are ever selected (non-zero rows) for readability
    active_rows = np.where(sel_freq.sum(axis=1) > 0)[0]
    sel_freq_active = sel_freq[active_rows]

    fig, ax = plt.subplots(figsize=(max(8, max_budget * 0.6), max(6, len(active_rows) * 0.22)))
    im = ax.imshow(sel_freq_active, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=sel_freq_active.max())
    ax.set_xticks(range(max_budget))
    ax.set_xticklabels([f"Step {i+1}" for i in range(max_budget)], rotation=45, ha="right")
    ax.set_yticks(range(len(active_rows)))
    ax.set_yticklabels([f"Feat {i}" for i in active_rows], fontsize=7)
    ax.set_xlabel("Acquisition Step")
    ax.set_ylabel("Feature Index")
    ax.set_title("CUBE-NM: Feature Selection Frequency per Step (EC2 kNN)")
    fig.colorbar(im, ax=ax, label="Selection Frequency")
    plt.tight_layout()
    heatmap_path = output_plot.replace(".png", "_feature_order.png")
    fig.savefig(heatmap_path, dpi=150)
    print(f"Feature order heatmap saved to {heatmap_path}")

    # ── Persist results ──────────────────────────────────────────────────────
    run_dir  = run_dir_from_checkpoint(checkpoint_path)
    run_name = os.path.basename(run_dir)

    save_eval_results(run_dir, results, ["budget", "ec2_acc", "rand_acc"])

    budgets = [r[0] for r in results]
    summary = {
        "auc_acc_curve":   auc_under_curve(budgets, ec2_accs),
        "final_acc":       ec2_accs[-1],
        "rand_final_acc":  rand_accs[-1],
        "max_budget":      max_budget,
    }
    append_to_results_log(run_name, "cube_nm", summary)

    # Also print top-5 most selected features per step
    print(f"\n{'Step':<6} | Top-5 features selected")
    print("=" * 45)
    for s in range(max_budget):
        top5 = np.argsort(selection_counts[:, s])[::-1][:5]
        counts = selection_counts[top5, s]
        items = ", ".join(f"feat{f}({c})" for f, c in zip(top5, counts))
        print(f"{s+1:<6} | {items}")

    # ── Context-first statistics ─────────────────────────────────────────────
    # Context features [0, n_contexts) are one-hot: one observation determines
    # the informative block. Optimal behavior: pick a context feature at step 1.
    n_test_samples = len(test_data)
    first_sel = np.argmax(selection_counts, axis=0)          # most-picked feat per step
    ctx_frac_per_step = (selection_counts[CONTEXT_FEATURES].sum(axis=0) /
                         np.maximum(1, selection_counts.sum(axis=0)))
    ctx_first_valid = context_first_step[~np.isnan(context_first_step)]
    stats = {
        "n_contexts": N_CONTEXTS,
        "context_features": CONTEXT_FEATURES,
        "first_step_most_picked_feature": int(first_sel[0]),
        "first_step_picks_context_frac": float(ctx_frac_per_step[0]),
        "first_3_steps_most_picked_are_context": float(np.mean(np.isin(first_sel[:3], CONTEXT_FEATURES))),
        "ctx_frac_per_step": [float(v) for v in ctx_frac_per_step],
        "mean_step_of_first_context_selection": float(np.mean(ctx_first_valid)) if len(ctx_first_valid) else None,
        "median_step_of_first_context_selection": float(np.median(ctx_first_valid)) if len(ctx_first_valid) else None,
        "budget_of_first_perfect_acc": int(next((b for b, a in enumerate(ec2_accs) if a >= 1.0), -1)),
    }
    # frac of samples that picked a context feature within the first 10 steps
    first_ctx_within_10 = context_first_step <= 10   # NaN → False
    stats["frac_context_within_first_10_steps"] = float(np.mean(first_ctx_within_10)) if n_test_samples else 0.0
    stats["frac_never_selected_context_in_first_10_steps"] = float(np.mean(~first_ctx_within_10))

    print(f"\n{'='*60}")
    print("Context-first analysis (context features = indices "
          f"{CONTEXT_FEATURES}, one-hot)")
    print("="*60)
    print(f"Step 1 most-picked feature:           feat{first_sel[0]} "
          f"(context fraction {ctx_frac_per_step[0]:.3f})")
    print(f"Mean 1st context selection at step:   {stats['mean_step_of_first_context_selection']:.2f} "
          f"(median {stats['median_step_of_first_context_selection']:.0f})")
    print(f"Frac context selected within 10 steps:{stats['frac_context_within_first_10_steps']:.3f}")
    print(f"Budget of first perfect accuracy:     {stats['budget_of_first_perfect_acc']}")

    ctx_stats_path = os.path.join(run_dir, "context_stats.json")
    with open(ctx_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Context stats saved → {ctx_stats_path}")


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
