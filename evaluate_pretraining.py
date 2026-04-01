import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from afabench.common.datasets.datasets import AFAContextDataset
from ec2_cube import GaussianNLLLoss, MaskLayer, MaskingPretrainerEC2, TwoHeadMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the pretrain checkpoint")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument(
        "--retention_rates",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.8, 1.0],
        help="Feature retention rates used for robustness evaluation",
    )
    parser.add_argument("--tolerance", type=float, default=0.1, help="Tolerance for feature accuracy")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrain_eval_outputs",
        help="Directory for plots and JSON summaries",
    )
    parser.add_argument(
        "--calibration_max_points",
        type=int,
        default=20000,
        help="Maximum number of masked feature points to draw in the uncertainty scatter plot",
    )
    return parser.parse_args()


def make_test_dataloader(batch_size, seed, num_workers, n_samples):
    test_dataset = AFAContextDataset(
        n_samples=n_samples,
        seed=seed + 100,
        context_feature_std=0.0,
        informative_feature_std=0.0,
        non_informative_feature_mean=0.5,
        non_informative_feature_std=0.0,
        use_cheap_context_features=True,
        n_contexts=5,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return test_dataset, test_dataloader


def build_pretrain_module(checkpoint_path):
    d_in = 55
    d_out = 8
    hidden = 512
    dropout = 0.3

    predictor = TwoHeadMLP(d_in, d_out, hidden, dropout)
    mask_layer = MaskLayer(mask_size=d_in, append=True)

    module = MaskingPretrainerEC2.load_from_checkpoint(
        checkpoint_path,
        model=predictor,
        mask_layer=mask_layer,
        lr=1e-3,
        loss_fn=nn.CrossEntropyLoss(),
        val_loss_fn=lambda pred, y: (pred.argmax(dim=1) == y).float().mean(),
        aux_loss_fn=GaussianNLLLoss(reduction="none", var_penalty_weight=0.01),
        strict=False,
    )
    return module


def generate_fixed_retention_mask(batch_size, d_in, retention_rate, device):
    if retention_rate <= 0:
        return torch.zeros(batch_size, d_in, device=device)
    if retention_rate >= 1:
        return torch.ones(batch_size, d_in, device=device)
    return (torch.rand(batch_size, d_in, device=device) < retention_rate).float()


def update_confusion_matrix(confusion, preds, targets, num_classes):
    indices = targets * num_classes + preds
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes).cpu().numpy()


def compute_classification_metrics(confusion):
    confusion = confusion.astype(np.int64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    total = confusion.sum()

    accuracy = float(true_positive.sum() / total) if total > 0 else 0.0

    precision_per_class = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted > 0,
    )
    recall_per_class = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(precision_per_class + recall_per_class) > 0,
    )

    macro_precision = float(precision_per_class.mean())
    macro_recall = float(recall_per_class.mean())
    macro_f1 = float(f1_per_class.mean())

    weighted_precision = float(np.average(precision_per_class, weights=support)) if support.sum() > 0 else 0.0
    weighted_recall = float(np.average(recall_per_class, weights=support)) if support.sum() > 0 else 0.0
    weighted_f1 = float(np.average(f1_per_class, weights=support)) if support.sum() > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "support_per_class": support.tolist(),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def save_confusion_matrix_plot(confusion, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(confusion.shape[1]))
    ax.set_yticks(np.arange(confusion.shape[0]))

    threshold = confusion.max() / 2 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_curve(retention_rates, accuracies, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(retention_rates, accuracies, marker="o", linewidth=2)
    ax.set_title("Robustness to Feature Sparsity")
    ax.set_xlabel("Feature Retention Rate")
    ax.set_ylabel("Classification Accuracy")
    ax.set_xticks(retention_rates)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_uncertainty_scatter(variances, abs_errors, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(variances, abs_errors, s=8, alpha=0.2)
    ax.set_title("Uncertainty Calibration")
    ax.set_xlabel("Predicted Variance exp(logvar)")
    ax.set_ylabel("Absolute Error |mu - x|")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_true_vs_pred_scatter(true_values, pred_values, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_values, pred_values, s=8, alpha=0.2)

    vmin = min(float(np.min(true_values)), float(np.min(pred_values)))
    vmax = max(float(np.max(true_values)), float(np.max(pred_values)))
    if np.isclose(vmin, vmax):
        pad = 0.05 if vmin == 0 else abs(vmin) * 0.05
        vmin -= pad
        vmax += pad

    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.5, color="tab:red")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_title("True Feature x vs Predicted Mean mu")
    ax.set_xlabel("True Feature x")
    ax.set_ylabel("Predicted Feature mu")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_group_summary(true_values, pred_values):
    groups = [0.0, 0.5, 1.0]
    summary = {}
    for g in groups:
        group_mask = np.isclose(true_values, g, atol=1e-8)
        group_preds = pred_values[group_mask]
        if group_preds.size == 0:
            summary[str(g)] = {
                "count": 0,
                "pred_mean": None,
                "pred_std": None,
                "mae_mean": None,
            }
            continue
        mae = np.abs(group_preds - g)
        summary[str(g)] = {
            "count": int(group_preds.size),
            "pred_mean": float(group_preds.mean()),
            "pred_std": float(group_preds.std()),
            "mae_mean": float(mae.mean()),
        }
    return summary


def save_grouped_violin_strip(true_values, pred_values, output_path):
    groups = [0.0, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(9, 6))

    data = [pred_values[np.isclose(true_values, g, atol=1e-8)] for g in groups]
    parts = ax.violinplot(data, positions=np.arange(len(groups)), showmeans=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.35)
    if "cmeans" in parts:
        parts["cmeans"].set_color("tab:red")
        parts["cmeans"].set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        if vals.size == 0:
            continue
        jitter = rng.normal(loc=0.0, scale=0.04, size=vals.size)
        x = np.full(vals.size, i) + jitter
        ax.scatter(x, vals, s=10, alpha=0.18, color="tab:blue")

    for y_ref in groups:
        ax.axhline(y=y_ref, linestyle="--", linewidth=1.0, alpha=0.6, color="gray")

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("True Feature x (grouped)")
    ax.set_ylabel("Predicted Feature mu")
    ax.set_title("Predicted mu Distribution by True Feature Group")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_grouped_violin_strip_multi(retention_plot_data, output_path, max_panels=4):
    if not retention_plot_data:
        return

    selected = retention_plot_data[:max_panels]
    n_panels = len(selected)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    groups = [0.0, 0.5, 1.0]
    rng = np.random.default_rng(42)

    for i, (rate, true_values, pred_values) in enumerate(selected):
        ax = axes[i]
        data = [pred_values[np.isclose(true_values, g, atol=1e-8)] for g in groups]

        parts = ax.violinplot(data, positions=np.arange(len(groups)), showmeans=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_alpha(0.35)
        if "cmeans" in parts:
            parts["cmeans"].set_color("tab:red")
            parts["cmeans"].set_linewidth(1.2)

        for j, vals in enumerate(data):
            if vals.size == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=0.04, size=vals.size)
            x = np.full(vals.size, j) + jitter
            ax.scatter(x, vals, s=9, alpha=0.18, color="tab:blue")

        for y_ref in groups:
            ax.axhline(y=y_ref, linestyle="--", linewidth=0.9, alpha=0.55, color="gray")

        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Retention={rate:.2f}")
        ax.set_xlabel("True x")
        ax.grid(True, linestyle="--", alpha=0.3)

    for i in range(n_panels, nrows * ncols):
        fig.delaxes(axes[i])

    axes[0].set_ylabel("Predicted mu")
    if n_panels > 2:
        axes[2].set_ylabel("Predicted mu")

    fig.suptitle("Predicted mu by True-Value Group across Retentions", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_jittered_true_vs_pred_scatter(true_values, pred_values, output_path):
    rng = np.random.default_rng(42)
    jittered_true = true_values + rng.normal(loc=0.0, scale=0.02, size=true_values.shape[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(jittered_true, pred_values, s=10, alpha=0.2)

    vmin = min(float(np.min(true_values)), float(np.min(pred_values)))
    vmax = max(float(np.max(true_values)), float(np.max(pred_values)))
    if np.isclose(vmin, vmax):
        pad = 0.05 if vmin == 0 else abs(vmin) * 0.05
        vmin -= pad
        vmax += pad

    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.5, color="tab:red")
    ax.set_xlim(vmin - 0.05, vmax + 0.05)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("True Feature x (jittered)")
    ax.set_ylabel("Predicted Feature mu")
    ax.set_title("True x vs Predicted mu (Jittered Scatter)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_rate(module, dataloader, retention_rate, tolerance, calibration_max_points, device):
    predictor = module.model
    mask_layer = module.mask_layer
    predictor.eval()

    num_classes = predictor.main_head.out_features
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_samples = 0
    total_correct = 0

    masked_feature_count = 0.0
    masked_image_count = 0.0
    masked_image_hit_count = 0.0
    mse_sum = 0.0
    nll_sum = 0.0
    feature_hit_sum = 0.0

    variance_points = []
    abs_error_points = []
    true_feature_points = []
    pred_feature_points = []
    points_collected = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)

            mask = generate_fixed_retention_mask(len(x), mask_layer.mask_size, retention_rate, device)
            x_masked = mask_layer(x, mask)
            pred_main, pred_aux = predictor(x_masked)
            mu, logvar = pred_aux
            # 【修改点】：与训练保持一致，将下限收紧至 -5.0
            logvar = torch.clamp(logvar, min=-5.0, max=10.0)

            preds = torch.argmax(pred_main, dim=1)
            total_samples += len(x)
            total_correct += (preds == y).sum().item()
            update_confusion_matrix(confusion, preds.cpu(), y.cpu(), num_classes)

            missing_mask = 1.0 - mask
            missing_per_sample = missing_mask.sum(dim=1)
            valid_samples = missing_per_sample > 0
            missing_count_batch = missing_mask.sum().item()

            if missing_count_batch > 0:
                sq_error = (mu - x) ** 2
                abs_error = (mu - x).abs()
                
                # 【修改点】：直接调用 module.aux_loss_fn 确保包含 var_penalty
                gaussian_nll = module.aux_loss_fn(pred_aux, x)

                mse_sum += (sq_error * missing_mask).sum().item()
                nll_sum += (gaussian_nll * missing_mask).sum().item()
                feature_hit_sum += ((abs_error < tolerance).float() * missing_mask).sum().item()
                masked_feature_count += missing_count_batch

                if valid_samples.any():
                    sample_hits = ((abs_error < tolerance) | (missing_mask == 0)).all(dim=1)
                    masked_image_hit_count += sample_hits[valid_samples].float().sum().item()
                    masked_image_count += valid_samples.float().sum().item()

                if points_collected < calibration_max_points:
                    flat_missing = missing_mask.bool().reshape(-1)
                    flat_variance = torch.exp(logvar).reshape(-1)[flat_missing]
                    flat_abs_error = abs_error.reshape(-1)[flat_missing]

                    remaining = calibration_max_points - points_collected
                    if flat_variance.numel() > remaining:
                        choice = torch.randperm(flat_variance.numel(), device=device)[:remaining]
                        flat_variance = flat_variance[choice]
                        flat_abs_error = flat_abs_error[choice]

                    variance_points.append(flat_variance.cpu().numpy())
                    abs_error_points.append(flat_abs_error.cpu().numpy())

                    flat_true = x.reshape(-1)[flat_missing]
                    flat_pred = mu.reshape(-1)[flat_missing]
                    if flat_true.numel() > remaining:
                        flat_true = flat_true[choice]
                        flat_pred = flat_pred[choice]

                    true_feature_points.append(flat_true.cpu().numpy())
                    pred_feature_points.append(flat_pred.cpu().numpy())
                    points_collected += flat_variance.numel()

    class_metrics = compute_classification_metrics(confusion)

    if masked_feature_count > 0:
        mse = mse_sum / masked_feature_count
        rmse = float(np.sqrt(mse))
        nll = nll_sum / masked_feature_count
        feature_accuracy = feature_hit_sum / masked_feature_count
    else:
        mse = None
        rmse = None
        nll = None
        feature_accuracy = None

    image_accuracy = masked_image_hit_count / masked_image_count if masked_image_count > 0 else None

    if variance_points:
        variances = np.concatenate(variance_points)
        abs_errors = np.concatenate(abs_error_points)
        if len(variances) > 1:
            calibration_corr = float(np.corrcoef(variances, abs_errors)[0, 1])
            if np.isnan(calibration_corr):
                calibration_corr = None
        else:
            calibration_corr = None
    else:
        variances = np.array([])
        abs_errors = np.array([])
        calibration_corr = None

    if true_feature_points:
        true_values = np.concatenate(true_feature_points)
        pred_values = np.concatenate(pred_feature_points)
    else:
        true_values = np.array([])
        pred_values = np.array([])

    group_summary = compute_group_summary(true_values, pred_values)

    metrics = {
        "retention_rate": retention_rate,
        "num_samples": total_samples,
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "confusion_matrix": confusion.tolist(),
        "classification": class_metrics,
        "auxiliary": {
            "masked_feature_count": int(masked_feature_count),
            "mse": mse,
            "rmse": rmse,
            "gaussian_nll": nll,
            "feature_accuracy": feature_accuracy,
            "image_accuracy": image_accuracy,
            "tolerance": tolerance,
            "uncertainty_error_pearson": calibration_corr,
        },
        "true_value_groups": group_summary,
    }
    return metrics, variances, abs_errors, true_values, pred_values


def sanitize_rate(retention_rate):
    return str(retention_rate).replace(".", "p")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading pretrain checkpoint from: {args.ckpt_path}")

    _, dataloader = make_test_dataloader(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        n_samples=args.n_samples,
    )
    module = build_pretrain_module(args.ckpt_path)
    module.to(device)
    module.eval()

    all_metrics = []
    retention_rates = []
    accuracies = []
    retention_plot_data = []

    for retention_rate in args.retention_rates:
        print(f"Evaluating retention rate = {retention_rate:.2f}")
        metrics, variances, abs_errors, true_values, pred_values = evaluate_rate(
            module,
            dataloader,
            retention_rate,
            args.tolerance,
            args.calibration_max_points,
            device,
        )
        all_metrics.append(metrics)
        retention_rates.append(retention_rate)
        accuracies.append(metrics["accuracy"])

        rate_tag = sanitize_rate(retention_rate)
        confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        save_confusion_matrix_plot(confusion, output_dir / f"confusion_matrix_retention_{rate_tag}.png")

        if len(variances) > 0:
            save_uncertainty_scatter(
                variances,
                abs_errors,
                output_dir / f"uncertainty_scatter_retention_{rate_tag}.png",
            )

        if len(true_values) > 0:
            save_true_vs_pred_scatter(
                true_values,
                pred_values,
                output_dir / f"true_vs_pred_scatter_retention_{rate_tag}.png",
            )
            save_jittered_true_vs_pred_scatter(
                true_values,
                pred_values,
                output_dir / f"jittered_true_vs_pred_scatter_retention_{rate_tag}.png",
            )
            retention_plot_data.append((retention_rate, true_values, pred_values))

        print(
            "  "
            f"acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['classification']['macro_f1']:.4f}, "
            f"aux_mse={metrics['auxiliary']['mse']}, "
            f"aux_nll={metrics['auxiliary']['gaussian_nll']}"
        )

    save_accuracy_curve(retention_rates, accuracies, output_dir / "accuracy_vs_retention.png")

    save_grouped_violin_strip_multi(
        retention_plot_data,
        output_dir / "grouped_violin_strip_multi_retentions.png",
        max_panels=4,
    )

    summary = {
        "checkpoint_path": args.ckpt_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "retention_rates": args.retention_rates,
        "results": all_metrics,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary")
    print("-" * 60)
    for metrics in all_metrics:
        aux = metrics["auxiliary"]
        print(
            f"retention={metrics['retention_rate']:.2f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"macro_precision={metrics['classification']['macro_precision']:.4f} | "
            f"macro_recall={metrics['classification']['macro_recall']:.4f} | "
            f"macro_f1={metrics['classification']['macro_f1']:.4f} | "
            f"mse={aux['mse']} | rmse={aux['rmse']} | nll={aux['gaussian_nll']} | "
            f"feature_acc={aux['feature_accuracy']} | image_acc={aux['image_accuracy']} | "
            f"uncertainty_corr={aux['uncertainty_error_pearson']}"
        )

    print("-" * 60)
    print(f"Saved summary and plots to: {output_dir}")


if __name__ == "__main__":
    main()