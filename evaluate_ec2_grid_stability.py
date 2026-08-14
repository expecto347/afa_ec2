import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from ec2_grid_stability import GridStabilityDataset
from ec2_core import Predictor, EC2Estimator, MaskLayer
from eval_utils import run_dir_from_checkpoint, save_eval_results, auc_under_curve, append_to_results_log


def evaluate_ec2(checkpoint_path, data_dir, gpu):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    d_in = 12
    d_out = 2
    hidden = 512
    dropout = 0.3
    max_features = d_in

    print(f"Loading checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    temp_ckpt = torch.load(checkpoint_path, map_location='cpu')
    feat_mean       = temp_ckpt['state_dict']['feat_mean'].numpy()
    feat_std        = temp_ckpt['state_dict']['feat_std'].numpy()
    X_train         = temp_ckpt['state_dict']['X_train']
    y_train         = temp_ckpt['state_dict']['y_train']
    feat_thresholds = temp_ckpt['state_dict']['feat_thresholds']

    test_csv = os.path.join(data_dir, "test.csv")
    print("Loading datasets...")
    test_dataset = GridStabilityDataset(test_csv, mean=feat_mean, std=feat_std)
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
        strict=False,  # tolerate kNN cache buffers (global_ec2_cache, X_train_knn, ...) registered in on_fit_start
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

            mask_ec2 = torch.zeros(len(x), d_in, dtype=x.dtype, device=x.device)
            mask_rand = torch.zeros(len(x), d_in, dtype=x.dtype, device=x.device)

            for feature_step in range(1, max_features + 1):
                x_masked_ec2 = model.mask_layer(x, mask_ec2)
                pred_ec2 = model.value_network(x_masked_ec2)
                pred_ec2 += 1e6 * mask_ec2
                best_feature_index = torch.argmin(pred_ec2, dim=1)

                mask_update = torch.zeros_like(mask_ec2)
                mask_update.scatter_(1, best_feature_index.unsqueeze(1), 1.0)
                mask_ec2 = torch.max(mask_ec2, mask_update)

                x_masked_ec2 = model.mask_layer(x, mask_ec2)
                pred_main = model.estimate_class_frequency_mc(x, mask_ec2, x_masked_ec2)
                all_preds[feature_step].append(pred_main.cpu())

                rand_scores = torch.rand(len(x), d_in, device=device)
                rand_scores -= 1e6 * mask_rand
                best_feature_index_rand = torch.argmax(rand_scores, dim=1)

                mask_update_rand = torch.zeros_like(mask_rand)
                mask_update_rand.scatter_(1, best_feature_index_rand.unsqueeze(1), 1.0)
                mask_rand = torch.max(mask_rand, mask_update_rand)

                x_masked_rand = model.mask_layer(x, mask_rand)
                pred_main_rand = model.estimate_class_frequency_mc(x, mask_rand, x_masked_rand)
                all_preds_random[feature_step].append(pred_main_rand.cpu())

    y_true = torch.cat(all_y).numpy()

    print("\n" + "="*80)
    print(f"{'Num Features':<15} | {'EC2 Accuracy':<15} | {'EC2 AUROC':<15} | {'Rand Accuracy':<15} | {'Rand AUROC':<15}")
    print("="*80)

    results = []

    for step in range(max_features + 1):
        preds = torch.cat(all_preds[step])
        probs = preds[:, 1].numpy()
        pred_labels = torch.argmax(preds, dim=1).numpy()

        preds_rand = torch.cat(all_preds_random[step])
        probs_rand = preds_rand[:, 1].numpy()
        pred_labels_rand = torch.argmax(preds_rand, dim=1).numpy()

        acc = accuracy_score(y_true, pred_labels)
        acc_rand = accuracy_score(y_true, pred_labels_rand)
        try:
            auc_score = roc_auc_score(y_true, probs)
        except ValueError:
            auc_score = float('nan')

        try:
            auc_score_rand = roc_auc_score(y_true, probs_rand)
        except ValueError:
            auc_score_rand = float('nan')

        results.append((step, acc, auc_score, acc_rand, auc_score_rand))
        print(f"{step:<15} | {acc:<15.4f} | {auc_score:<15.4f} | {acc_rand:<15.4f} | {auc_score_rand:<15.4f}")

    print("="*80)

    # ── Persist results ──────────────────────────────────────────────────────
    run_dir  = run_dir_from_checkpoint(checkpoint_path)
    run_name = os.path.basename(run_dir)

    columns = ["budget", "ec2_acc", "ec2_auroc", "rand_acc", "rand_auroc"]
    save_eval_results(run_dir, results, columns)

    budgets   = [r[0] for r in results]
    ec2_accs  = [r[1] for r in results]
    ec2_aurocs = [r[2] for r in results]
    summary = {
        "auc_acc_curve":   auc_under_curve(budgets, ec2_accs),
        "auc_auroc_curve": auc_under_curve(budgets, ec2_aurocs),
        "final_acc":       results[-1][1],
        "final_auroc":     results[-1][2],
        "rand_final_acc":  results[-1][3],
        "rand_final_auroc":results[-1][4],
        "max_budget":      results[-1][0],
    }
    append_to_results_log(run_name, "grid_stability", summary)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='logs/ec2_grid_stability_trial_0/ec2_train/checkpoints/best_val_perf_model.ckpt')
    parser.add_argument('--data_dir', type=str, default='/data/users/haozhes/DIME/experiments/grid_stability/data')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    evaluate_ec2(args.checkpoint, args.data_dir, args.gpu)
