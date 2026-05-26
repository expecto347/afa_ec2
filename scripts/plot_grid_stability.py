"""
Grid Stability: AUROC and Accuracy vs. budget.
EC2 (ours) vs. baselines — baselines use worst (min) across 3 seeds.
"""

import matplotlib.pyplot as plt

features = list(range(13))

# EC2 (ours) — from existing figure.py, converted to %
ec2_auroc = [50.00, 67.63, 74.72, 80.78, 83.85, 87.34, 88.91, 92.73, 96.52, 97.93, 98.92, 99.32, 99.37]
ec2_acc   = [65.30, 70.50, 73.00, 76.50, 77.80, 79.70, 81.20, 86.70, 89.70, 92.80, 94.00, 94.50, 94.40]

# DIME
dime_auroc = [50.00, 63.78, 71.18, 78.62, 85.87, 91.54, 96.93, 99.00, 99.47, 99.25, 99.07, 98.82, 98.46]
dime_acc   = [65.30, 68.10, 70.10, 74.30, 79.20, 85.60, 91.10, 94.40, 95.90, 95.10, 94.30, 93.40, 93.10]

# CAE
cae_auroc = [50.00, 62.02, 66.90, 76.21, 81.14, 85.14, 89.93, 92.63, 99.72, 99.71, 99.68, 99.62, 99.66]
cae_acc   = [65.30, 68.20, 67.70, 72.20, 74.10, 78.30, 81.10, 86.20, 96.60, 96.80, 96.70, 96.70, 96.50]

# EDDI
eddi_auroc = [50.00, 64.43, 73.04, 77.42, 80.57, 84.71, 87.73, 91.12, 93.59, 95.77, 97.03, 97.96, 98.35]
eddi_acc   = [65.30, 68.90, 71.20, 75.00, 74.70, 78.30, 80.60, 83.60, 86.30, 89.90, 91.30, 93.30, 92.80]

# Greedy Ablation
ga_auroc = [50.00, 62.11, 71.86, 79.96, 84.61, 90.12, 95.38, 98.44, 99.27, 99.19, 99.20, 99.22, 99.23]
ga_acc   = [65.30, 67.40, 71.30, 75.20, 78.70, 82.60, 88.90, 93.30, 95.70, 94.90, 94.90, 94.70, 94.40]

# Iterative
iter_auroc = [50.00, 63.22, 70.16, 77.96, 83.86, 90.54, 95.51, 97.55, 97.59, 97.44, 97.46, 97.37, 97.61]
iter_acc   = [65.30, 67.50, 69.60, 73.90, 78.10, 83.00, 87.50, 91.10, 91.10, 90.60, 90.30, 89.70, 91.10]

# GDFS
gdfs_auroc = [50.00, 64.03, 72.28, 78.72, 83.17, 90.99, 96.71, 99.02, 99.69, 99.72, 99.72, 99.69, 99.71]
gdfs_acc   = [65.30, 67.60, 71.10, 73.50, 76.70, 83.30, 90.00, 94.40, 97.40, 96.90, 97.20, 96.90, 97.20]

# AACO (seed 0)
aaco_auroc = [50.00, 67.86, 74.03, 81.04, 86.15, 91.66, 95.75, 98.26, 99.23, 99.46, 99.47, 99.51, 99.35]
aaco_acc   = [65.30, 71.30, 71.00, 74.80, 79.10, 84.00, 90.20, 93.30, 95.70, 96.00, 95.70, 96.20, 95.30]

fig, ax = plt.subplots(figsize=(7, 5))

for ax, metric, title in zip([ax], ["AUROC"], ["AUROC (%)"]):
    ec2_y   = ec2_auroc   if metric == "AUROC" else ec2_acc
    dime_y  = dime_auroc  if metric == "AUROC" else dime_acc
    cae_y   = cae_auroc   if metric == "AUROC" else cae_acc
    eddi_y  = eddi_auroc  if metric == "AUROC" else eddi_acc
    ga_y    = ga_auroc    if metric == "AUROC" else ga_acc
    iter_y  = iter_auroc  if metric == "AUROC" else iter_acc
    gdfs_y  = gdfs_auroc  if metric == "AUROC" else gdfs_acc
    aaco_y  = aaco_auroc  if metric == "AUROC" else aaco_acc

    ax.plot(features, ec2_y,  marker="o", linewidth=2, label="EC2 (ours)",       color="tab:blue")
    ax.plot(features, dime_y, marker="P", linewidth=1.5, label="DIME",           color="tab:cyan",    linestyle="--")
    ax.plot(features, cae_y,  marker="s", linewidth=1.5, label="CAE ",     color="tab:orange",  linestyle="--")
    ax.plot(features, eddi_y, marker="^", linewidth=1.5, label="EDDI",    color="tab:green",   linestyle="--")
    ax.plot(features, ga_y,   marker="D", linewidth=1.5, label="Greedy-Abl", color="tab:red", linestyle="--")
    ax.plot(features, iter_y, marker="v", linewidth=1.5, label="Iterative", color="tab:purple", linestyle="--")
    ax.plot(features, gdfs_y, marker="x", linewidth=1.5, label="GDFS",           color="tab:brown",   linestyle="-.")
    ax.plot(features, aaco_y, marker="h", linewidth=1.5, label="AACO",           color="tab:olive",   linestyle="-.")

    ax.set_xlabel("Budget (# features acquired)")
    ax.set_ylabel(title)
    ax.set_title(f"Grid Stability — {title}")
    ax.set_xticks(features)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/grid_stability_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
