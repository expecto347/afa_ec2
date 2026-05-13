"""
Plot Accuracy and AUROC vs. number of acquired features for Metabric,
comparing EC2 (ours), Random baseline, and DIME.
"""

import matplotlib.pyplot as plt
import numpy as np

features = list(range(13))

ec2_acc   = [0.3351, 0.3979, 0.5969, 0.6283, 0.6859, 0.6806, 0.6754, 0.6806, 0.7068, 0.7173, 0.7225, 0.7016, 0.6859]
ec2_auroc = [0.5000, 0.7395, 0.8403, 0.8627, 0.8728, 0.8922, 0.8985, 0.8849, 0.8960, 0.9045, 0.9065, 0.9142, 0.9112]

rand_acc   = [0.3351, 0.4084, 0.4764, 0.5445, 0.6178, 0.6545, 0.6702, 0.6859, 0.6754, 0.6545, 0.6754, 0.7120, 0.6859]
rand_auroc = [0.5000, 0.6612, 0.7583, 0.8033, 0.8394, 0.8666, 0.8857, 0.8916, 0.8987, 0.8957, 0.9016, 0.9069, 0.9112]

dime_acc   = [0.3351, 0.4293, 0.5707, 0.6335, 0.6440, 0.6859, 0.6911, 0.6859, 0.6859, 0.7068, 0.7068, 0.6911, 0.7016]
dime_auroc = [0.5416, 0.7431, 0.8489, 0.8628, 0.8901, 0.9047, 0.8981, 0.8969, 0.9011, 0.9111, 0.9140, 0.9109, 0.9137]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, ec2_y, rand_y, dime_y, ylabel in zip(
    axes,
    [ec2_acc, ec2_auroc],
    [rand_acc, rand_auroc],
    [dime_acc, dime_auroc],
    ["Accuracy", "AUROC"],
):
    ax.plot(features, ec2_y,   marker="o", label="EC2 (ours)", color="tab:blue")
    ax.plot(features, dime_y,  marker="s", label="DIME",       color="tab:orange")
    ax.plot(features, rand_y,  marker="^", label="Random",     color="tab:gray", linestyle="--")

    ax.set_xlabel("Number of Acquired Features")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Metabric — {ylabel}")
    ax.set_xticks(features)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = "scripts/metabric_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
