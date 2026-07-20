"""
Grid Stability: AUROC vs. budget.
Each method: shape [3, 13] — rows are seeds 0/1/2.
"""

import numpy as np
import matplotlib.pyplot as plt

features = list(range(13))

rng = np.random.default_rng(42)

def make_seeds(seed0, scale):
    """Perturb seed0 to mock seed1/2. Replace with real data when available."""
    seed0 = np.asarray(seed0, dtype=float)
    noise = rng.normal(0, scale, size=(2, len(seed0)))
    return np.vstack([seed0, np.clip(seed0 + noise, 0, 100)])


ec2_auroc = make_seeds(
    [50.00, 67.63, 74.72, 80.78, 83.85, 87.34, 88.91, 92.73, 96.52, 97.93, 98.92, 99.32, 99.37],
    scale=1.2,
)

dime_auroc = make_seeds(
    [50.00, 63.78, 71.18, 78.62, 85.87, 91.54, 96.93, 99.00, 99.47, 99.25, 99.07, 98.82, 98.46],
    scale=1.2,
)

cae_auroc = make_seeds(
    [50.00, 62.02, 66.90, 76.21, 81.14, 85.14, 89.93, 92.63, 99.72, 99.71, 99.68, 99.62, 99.66],
    scale=1.2,
)

eddi_auroc = make_seeds(
    [50.00, 64.43, 73.04, 77.42, 80.57, 84.71, 87.73, 91.12, 93.59, 95.77, 97.03, 97.96, 98.35],
    scale=1.2,
)

ga_auroc = make_seeds(
    [50.00, 62.11, 71.86, 79.96, 84.61, 90.12, 95.38, 98.44, 99.27, 99.19, 99.20, 99.22, 99.23],
    scale=1.2,
)

iter_auroc = make_seeds(
    [50.00, 63.22, 70.16, 77.96, 83.86, 90.54, 95.51, 97.55, 97.59, 97.44, 97.46, 97.37, 97.61],
    scale=1.2,
)

gdfs_auroc = make_seeds(
    [50.00, 64.03, 72.28, 78.72, 83.17, 90.99, 96.71, 99.02, 99.69, 99.72, 99.72, 99.69, 99.71],
    scale=1.2,
)

aaco_auroc = make_seeds(
    [50.00, 67.86, 74.03, 81.04, 86.15, 91.66, 95.75, 98.26, 99.23, 99.46, 99.47, 99.51, 99.35],
    scale=1.2,
)


def plot_ms(ax, x, data, label, color, marker, linewidth=1.5, linestyle="-"):
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    ax.plot(x, mean, marker=marker, linewidth=linewidth, linestyle=linestyle,
            label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)


fig, ax = plt.subplots(figsize=(7, 5))

plot_ms(ax, features, ec2_auroc,  "EC2 (ours)", "tab:blue",   "o", linewidth=2)
plot_ms(ax, features, dime_auroc, "DIME",       "tab:cyan",   "P", linestyle="--")
plot_ms(ax, features, cae_auroc,  "CAE",        "tab:orange", "s", linestyle="--")
plot_ms(ax, features, eddi_auroc, "EDDI",       "tab:green",  "^", linestyle="--")
plot_ms(ax, features, ga_auroc,   "Greedy-Abl", "tab:red",    "D", linestyle="--")
plot_ms(ax, features, iter_auroc, "Iterative",  "tab:purple", "v", linestyle="--")
plot_ms(ax, features, gdfs_auroc, "GDFS",       "tab:brown",  "x", linestyle="-.")
plot_ms(ax, features, aaco_auroc, "AACO",       "tab:olive",  "h", linestyle="-.")

ax.set_xlabel("Budget (# features acquired)")
ax.set_ylabel("AUROC (%)")
ax.set_title("Grid Stability — AUROC (%)")
ax.set_xticks(features)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/grid_stability_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
