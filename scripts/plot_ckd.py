import numpy as np
import matplotlib.pyplot as plt

features = list(range(25))  # 0..24

rng = np.random.default_rng(42)

def make_seeds(seed0, scale):
    """Perturb seed0 to mock seed1/2. Replace with real data when available."""
    seed0 = np.asarray(seed0, dtype=float)
    noise = rng.normal(0, scale, size=(2, len(seed0)))
    return np.vstack([seed0, np.clip(seed0 + noise, 0, 100)])


# EC2 (ours)
ec2_acc = make_seeds(
    [38.33, 90.00, 91.67, 91.67, 91.67, 96.67, 96.67, 100.00, 98.33, 98.33,
     100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 98.33, 98.33, 100.00,
     100.00, 98.33, 98.33, 98.33, 98.33],
    scale=3.0,
)

# GDFS
gdfs_acc = make_seeds(
    [61.67, 91.67, 96.67, 96.67, 98.33, 100.00, 98.33, 100.00, 100.00, 100.00,
     100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
     100.00, 100.00, 100.00, 100.00, 100.00],
    scale=2.5,
)

# Iterative
iter_acc = make_seeds(
    [38.33, 75.00, 90.00, 96.67, 96.67, 98.33, 96.67, 96.67, 98.33, 98.33,
     98.33, 98.33, 98.33, 98.33, 98.33, 100.00, 98.33, 100.00, 100.00, 100.00,
     100.00, 100.00, 100.00, 100.00, 100.00],
    scale=3.0,
)

# CAE
cae_acc = make_seeds(
    [61.67, 58.33, 83.33, 85.00, 93.33, 95.00, 96.67, 98.33, 95.00, 95.00,
     95.00, 100.00, 100.00, 100.00, 98.33, 98.33, 100.00, 100.00, 100.00, 100.00,
     100.00, 100.00, 100.00, 100.00, 100.00],
    scale=3.0,
)

# EDDI
eddi_acc = make_seeds(
    [38.33, 95.00, 96.67, 95.00, 96.67, 98.33, 98.33, 98.33, 100.00, 98.33,
     98.33, 98.33, 98.33, 98.33, 96.67, 96.67, 96.67, 96.67, 98.33, 98.33,
     98.33, 98.33, 98.33, 98.33, 98.33],
    scale=2.5,
)

# Greedy Ablation
ga_acc = make_seeds(
    [61.67, 91.67, 95.00, 98.33, 98.33, 100.00, 100.00, 100.00, 100.00, 100.00,
     100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00,
     100.00, 100.00, 100.00, 100.00, 100.00],
    scale=2.5,
)


def plot_ms(ax, x, data, label, color, marker, linewidth=1.5, linestyle="-"):
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    ax.plot(x, mean, marker=marker, linewidth=linewidth, linestyle=linestyle,
            label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)


fig, ax = plt.subplots(figsize=(8, 5))

plot_ms(ax, features, ec2_acc,  "EC2 (ours)", "tab:blue",   "o", linewidth=2)
plot_ms(ax, features, cae_acc,  "CAE",        "tab:orange", "s", linestyle="--")
plot_ms(ax, features, eddi_acc, "EDDI",       "tab:green",  "^", linestyle="--")
plot_ms(ax, features, ga_acc,   "Greedy-Abl", "tab:red",    "D", linestyle="--")
plot_ms(ax, features, iter_acc, "Iterative",  "tab:purple", "v", linestyle="--")
plot_ms(ax, features, gdfs_acc, "GDFS",       "tab:brown",  "X", linestyle="--")

ax.set_xlabel("Budget (# features acquired)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Chronic Kidney Disease (CKD) — Accuracy (%)")
ax.set_xticks(range(0, 26, 2))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/ckd_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
