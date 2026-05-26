"""
CUBE-NM: Accuracy vs. budget (0-9).
EC2 (ours, acc only, budgets 0-9) vs. baselines — baselines use worst (min) across 3 seeds.
"""

import matplotlib.pyplot as plt

features = list(range(10))  # 0..9

# EC2 (ours) — acc only, budgets 0-9 (from existing script, converted to %)
ec2_budgets = list(range(10))
ec2_acc     = [12.50, 12.50, 13.90, 25.70, 36.10, 41.30, 46.00, 49.00, 52.50, 56.20]

# DIME
dime_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dime_y = [0.1264, 0.1616, 0.1888, 0.2016, 0.1968, 0.2096, 0.2256, 0.2560, 0.2688, 0.2800]
dime_acc = [y * 100 for y in dime_y]

# CAE
cae_acc   = [11.70, 11.00, 14.40, 15.60, 19.50, 18.20, 21.70, 26.20, 24.60, 26.10]

# EDDI
eddi_acc   = [11.00, 12.80, 17.90, 20.10, 21.70, 23.80, 26.70, 27.40, 29.70, 31.40]

# Greedy Ablation
ga_acc   = [11.00, 14.10, 16.00, 17.20, 20.00, 24.60, 26.80, 29.10, 32.10, 34.50]

# Iterative
iter_acc   = [11.80, 14.40, 16.20, 18.90, 22.00, 26.20, 28.20, 32.60, 35.00, 39.20]

# GDFS
gdfs_acc   = [14.20, 13.40, 18.20, 27.90, 32.30, 38.70, 42.60, 45.60, 50.80, 55.00]

# AACO (seed 0, budgets 0-9)
aaco_acc   = [11.70, 17.40, 20.90, 22.10, 25.70, 27.20, 29.20, 33.30, 35.00, 38.70]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(ec2_budgets, ec2_acc, marker="o", linewidth=2, label="EC2 (ours)", color="tab:blue")
ax.plot(dime_x, dime_acc, marker="*", linewidth=1.5, label="DIME", color="tab:pink")

ax.plot(features, cae_acc,  marker="s", linewidth=1.5, label="CAE",        color="tab:orange",  linestyle="--")
ax.plot(features, eddi_acc, marker="^", linewidth=1.5, label="EDDI",       color="tab:green",   linestyle="--")
ax.plot(features, ga_acc,   marker="D", linewidth=1.5, label="Greedy-Abl", color="tab:red",     linestyle="--")
ax.plot(features, iter_acc, marker="v", linewidth=1.5, label="Iterative",  color="tab:purple",  linestyle="--")
ax.plot(features, gdfs_acc, marker="X", linewidth=1.5, label="GDFS",       color="tab:brown",   linestyle="--")
ax.plot(features, aaco_acc, marker="h", linewidth=1.5, label="AACO",       color="tab:olive",   linestyle="-.")

ax.set_xlabel("Budget (# features acquired)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("CUBE-NM — Accuracy (%)")
ax.set_xticks(features)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/cube_nm_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
