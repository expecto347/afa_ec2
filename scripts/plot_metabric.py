"""
Metabric: AUROC and Accuracy vs. budget.
EC2 (ours) vs. baselines — baselines use worst (min) across 3 seeds.
"""

import matplotlib.pyplot as plt

features = list(range(13))

# EC2 (ours) — converted to %
ec2_auroc = [50.00, 67.91, 82.92, 86.31, 89.19, 89.55, 89.77, 90.30, 89.78, 90.27, 90.46, 90.80, 91.04]
ec2_acc   = [33.51, 46.07, 58.64, 62.30, 64.40, 68.59, 65.97, 69.11, 70.16, 70.16, 71.20, 71.73, 69.11]

# CAE
cae_auroc = [48.21, 69.83, 72.79, 78.30, 82.17, 86.72, 84.54, 89.08, 88.70, 89.46, 89.91, 90.58, 90.60]
cae_acc   = [33.51, 41.36, 48.69, 50.79, 60.73, 59.69, 60.73, 67.02, 67.02, 68.59, 70.16, 70.68, 70.16]

# EDDI
eddi_auroc = [48.21, 71.83, 79.78, 83.00, 85.84, 86.24, 88.41, 89.32, 90.07, 89.94, 90.10, 90.56, 90.62]
eddi_acc   = [33.51, 37.70, 55.50, 57.07, 60.21, 64.40, 68.06, 70.68, 72.25, 69.63, 69.11, 69.11, 67.02]

# Greedy Ablation
ga_auroc = [49.65, 73.65, 80.55, 86.25, 87.81, 89.86, 89.29, 90.17, 90.64, 90.76, 90.92, 91.44, 91.47]
ga_acc   = [33.51, 40.84, 55.50, 62.30, 69.11, 67.54, 65.97, 66.49, 68.59, 67.54, 71.73, 69.11, 69.63]

# Iterative
iter_auroc = [49.05, 72.64, 80.21, 84.59, 86.50, 87.43, 88.40, 89.07, 90.18, 90.11, 90.15, 90.47, 90.82]
iter_acc   = [33.51, 44.50, 53.40, 57.59, 61.78, 63.35, 67.02, 68.59, 68.06, 70.16, 69.63, 69.63, 68.06]

# GDFS — single run
gdfs_auroc = [51.29, 74.02, 81.94, 84.93, 88.05, 89.90, 90.73, 90.38, 90.52, 90.59, 90.80, 90.98, 91.17]
gdfs_acc   = [33.51, 46.07, 56.02, 63.87, 68.06, 74.87, 71.73, 70.68, 69.11, 69.63, 70.68, 71.20, 71.73]

# DIME
dime_auroc = [50.00, 74.23, 83.03, 88.08, 88.23, 88.26, 88.59, 89.43, 89.16, 90.27, 90.03, 89.95, 91.05]
dime_acc   = [33.51, 41.88, 60.21, 67.02, 65.97, 65.97, 62.30, 63.87, 62.83, 67.54, 69.63, 68.59, 68.06]

# AACO (seed 0)
aaco_auroc = [50.00, 66.01, 77.26, 84.33, 87.04, 89.30, 90.02, 90.47, 90.53, 90.47, 90.40, 90.80, 91.06]
aaco_acc   = [33.51, 44.50, 52.36, 54.97, 59.69, 68.06, 71.20, 72.77, 71.20, 70.68, 69.63, 72.25, 71.73]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(features, ec2_auroc,  marker="o", linewidth=2, label="EC2 (ours)",          color="tab:blue")
ax.plot(features, cae_auroc,  marker="s", linewidth=1.5, label="CAE",        color="tab:orange",  linestyle="--")
ax.plot(features, eddi_auroc, marker="^", linewidth=1.5, label="EDDI",       color="tab:green",   linestyle="--")
ax.plot(features, ga_auroc,   marker="D", linewidth=1.5, label="Greedy-Abl", color="tab:red",     linestyle="--")
ax.plot(features, iter_auroc, marker="v", linewidth=1.5, label="Iterative",  color="tab:purple",  linestyle="--")
ax.plot(features, gdfs_auroc, marker="x", linewidth=1.5, label="GDFS",              color="tab:brown",   linestyle="-.")
ax.plot(features, dime_auroc, marker="*", linewidth=1.5, label="DIME",              color="tab:pink",    linestyle=":")
ax.plot(features, aaco_auroc, marker="h", linewidth=1.5, label="AACO",             color="tab:olive",   linestyle="-.")

ax.set_xlabel("Budget (# features acquired)")
ax.set_ylabel("AUROC (%)")
ax.set_title("Metabric — AUROC (%)")
ax.set_xticks(features)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/metabric_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
