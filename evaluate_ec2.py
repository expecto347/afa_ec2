import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from afabench.common.datasets.datasets import AFAContextDataset
from ec2_cube import GaussianNLLLoss, MaskLayer, TwoHeadMLP, WeightedEC2Estimator, multiclass_accuracy


def evaluate_model(checkpoint_path, batch_size=512, max_features=15, seed=42, output_plot="test_accuracy_curve.png"):
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = AFAContextDataset(
        n_samples=5000,
        seed=seed + 100,
        context_feature_std=0.0,
        informative_feature_std=0.0,
        non_informative_feature_mean=0.5,
        non_informative_feature_std=0.0,
        use_cheap_context_features=True,
        n_contexts=5,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    feature_costs = test_dataset.get_feature_acquisition_costs().to(device)

    d_in = 55
    d_out = 8
    hidden = 512
    dropout = 0.3

    predictor = TwoHeadMLP(d_in, d_out, hidden, dropout)
    value_network = nn.Sequential(
        nn.Linear(d_in * 2, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_in),
    )
    mask_layer = MaskLayer(mask_size=d_in, append=True)

    model = WeightedEC2Estimator.load_from_checkpoint(
        checkpoint_path,
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        lr=1e-3,
        max_features=max_features,
        eps=0.0,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        val_loss_fn=multiclass_accuracy,
        aux_loss_fn=GaussianNLLLoss(reduction="none", var_penalty_weight=0.01),
        feature_costs=feature_costs,
        strict=False,
    )
    model.to(device)
    model.eval()

    accuracies_per_step = {k: [] for k in range(max_features + 1)}

    print("Starting evaluation...")
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)

            mask = torch.zeros(len(x), model.mask_size, dtype=x.dtype, device=device)

            for step in range(max_features + 1):
                x_masked = model.mask_layer(x, mask)
                pred_main, _ = model.predictor(x_masked)

                pred_labels = torch.argmax(pred_main, dim=1)
                acc = (pred_labels == y).float().mean().item()
                accuracies_per_step[step].append(acc)

                if step < max_features:
                    pred_ec2 = model.value_network(x_masked)
                    pred_ec2 -= 1e6 * mask
                    best_features = torch.argmax(pred_ec2 / model.feature_costs, dim=1)

                    new_mask_addition = torch.zeros_like(mask)
                    new_mask_addition[torch.arange(len(x)), best_features] = 1
                    mask = torch.max(mask, new_mask_addition)

    steps = list(range(max_features + 1))
    avg_accuracies = [np.mean(accuracies_per_step[step]) for step in steps]

    print("\n--- Test Results ---")
    for step, acc in zip(steps, avg_accuracies):
        print(f"Features Acquired: {step:2d} | Accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, avg_accuracies, marker="o", linestyle="-", color="b")
    plt.title("Active Feature Acquisition: Accuracy vs. Budget", fontsize=14)
    plt.xlabel("Number of Features Acquired", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(steps)
    plt.ylim(0, 1.05)
    plt.axhline(y=0.125, color="r", linestyle="--", label="Random Guess (12.5%)")
    plt.legend()

    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"\nPlot successfully saved as {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the saved model .ckpt file")
    parser.add_argument("--max_features", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_plot", type=str, default="test_accuracy_curve.png")
    args = parser.parse_args()

    evaluate_model(
        args.ckpt_path,
        batch_size=args.batch_size,
        max_features=args.max_features,
        seed=args.seed,
        output_plot=args.output_plot,
    )
