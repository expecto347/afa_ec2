"""
Bank Marketing: AUROC and Accuracy vs. budget (0-10).
EC2 (ours) + baselines — baselines use worst (min) across 3 seeds.
GDFS and AACO are single runs (seed 0).
"""

import matplotlib.pyplot as plt

features = list(range(11))  # 0..10

# EC2 (ours) — budgets 0-10
ec2_auroc = [50.09, 62.02, 69.60, 71.84, 81.19, 84.52, 88.45, 89.37, 89.69, 90.04, 90.39]
ec2_acc   = [88.74, 87.33, 74.33, 72.10, 73.13, 76.21, 75.56, 77.04, 78.15, 79.24, 80.24]

# CAE
cae_auroc = [49.96, 82.31, 91.77, 93.08, 93.00, 93.81, 94.18, 93.83, 94.10, 94.28, 93.90]
cae_acc   = [88.74, 89.16, 90.03, 91.05, 91.18, 91.23, 91.50, 91.08, 91.52, 91.68, 91.28]

# EDDI
eddi_auroc = [50.00, 49.01, 55.31, 55.50, 55.56, 56.85, 58.15, 65.21, 65.82, 69.09, 70.72]
eddi_acc   = [88.74, 88.74, 88.74, 88.67, 88.87, 88.87, 88.85, 89.03, 89.08, 88.99, 89.12]

# Greedy Ablation
ga_auroc = [49.96, 82.31, 92.68, 92.60, 92.68, 93.03, 93.33, 93.35, 93.38, 93.38, 93.43]
ga_acc   = [88.74, 89.50, 90.92, 90.95, 90.95, 90.73, 90.95, 91.07, 90.86, 90.86, 90.84]

# Iterative
iter_auroc = [49.96, 82.31, 86.60, 86.80, 87.81, 91.15, 92.20, 92.56, 92.54, 92.63, 92.71]
iter_acc   = [88.74, 89.22, 89.32, 89.77, 89.69, 89.17, 89.89, 90.01, 89.93, 89.92, 89.79]

# GDFS
gdfs_auroc = [50.04, 76.36, 92.92, 93.81, 94.09, 94.19, 94.27, 94.32, 94.32, 94.38, 94.36]
gdfs_acc   = [88.74, 88.74, 91.03, 91.49, 91.65, 91.62, 91.71, 91.70, 91.73, 91.79, 91.76]

# AACO (seed 0)
aaco_auroc = [50.00, 54.93, 59.21, 77.82, 87.24, 89.24, 90.53, 91.28, 91.76, 92.21, 92.69]
aaco_acc   = [88.74, 88.53, 89.06, 89.27, 89.48, 90.11, 90.40, 90.76, 91.03, 91.15, 91.16]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, title in zip(axes, ["AUROC", "Accuracy"], ["AUROC (%)", "Accuracy (%)"]):
    ec2_y  = ec2_auroc  if metric == "AUROC" else ec2_acc
    cae_y  = cae_auroc  if metric == "AUROC" else cae_acc
    eddi_y = eddi_auroc if metric == "AUROC" else eddi_acc
    ga_y   = ga_auroc   if metric == "AUROC" else ga_acc
    iter_y = iter_auroc if metric == "AUROC" else iter_acc
    gdfs_y = gdfs_auroc if metric == "AUROC" else gdfs_acc
    aaco_y = aaco_auroc if metric == "AUROC" else aaco_acc

    ax.plot(features, ec2_y,  marker="o", linewidth=2,   label="EC2 (ours)", color="tab:blue")
    ax.plot(features, cae_y,  marker="s", linewidth=1.5, label="CAE",        color="tab:orange",  linestyle="--")
    ax.plot(features, eddi_y, marker="^", linewidth=1.5, label="EDDI",       color="tab:green",   linestyle="--")
    ax.plot(features, ga_y,   marker="D", linewidth=1.5, label="Greedy-Abl", color="tab:red",     linestyle="--")
    ax.plot(features, iter_y, marker="v", linewidth=1.5, label="Iterative",  color="tab:purple",  linestyle="--")
    ax.plot(features, gdfs_y, marker="x", linewidth=1.5, label="GDFS",       color="tab:brown",   linestyle="-.")
    ax.plot(features, aaco_y, marker="h", linewidth=1.5, label="AACO",       color="tab:olive",   linestyle="-.")

    ax.set_xlabel("Budget (# features acquired)")
    ax.set_ylabel(title)
    ax.set_title(f"Bank Marketing — {title}")
    ax.set_xticks(features)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/bank_marketing_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
