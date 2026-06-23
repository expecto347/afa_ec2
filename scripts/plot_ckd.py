# import matplotlib.pyplot as plt

features = list(range(25))  # 0..24

# EC2 (ours)
ec2_acc = [38.33, 90.00, 91.67, 91.67, 91.67, 96.67, 96.67, 100.00, 98.33, 98.33, 
           100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 98.33, 98.33, 100.00, 
           100.00, 98.33, 98.33, 98.33, 98.33]

# GDFS
gdfs_acc = [61.67, 91.67, 96.67, 96.67, 98.33, 100.00, 98.33, 100.00, 100.00, 100.00, 
            100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 
            100.00, 100.00, 100.00, 100.00, 100.00]

# Iterative
iter_acc = [38.33, 75.00, 90.00, 96.67, 96.67, 98.33, 96.67, 96.67, 98.33, 98.33, 
            98.33, 98.33, 98.33, 98.33, 98.33, 100.00, 98.33, 100.00, 100.00, 100.00, 
            100.00, 100.00, 100.00, 100.00, 100.00]

# CAE
cae_acc = [61.67, 58.33, 83.33, 85.00, 93.33, 95.00, 96.67, 98.33, 95.00, 95.00, 
           95.00, 100.00, 100.00, 100.00, 98.33, 98.33, 100.00, 100.00, 100.00, 100.00, 
           100.00, 100.00, 100.00, 100.00, 100.00]

# EDDI
eddi_acc = [38.33, 95.00, 96.67, 95.00, 96.67, 98.33, 98.33, 98.33, 100.00, 98.33, 
            98.33, 98.33, 98.33, 98.33, 96.67, 96.67, 96.67, 96.67, 98.33, 98.33, 
            98.33, 98.33, 98.33, 98.33, 98.33]

# Greedy Ablation
ga_acc = [61.67, 91.67, 95.00, 98.33, 98.33, 100.00, 100.00, 100.00, 100.00, 100.00, 
          100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 
          100.00, 100.00, 100.00, 100.00, 100.00]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(features, ec2_acc,  marker="o", linewidth=2,   label="EC2 (ours)", color="tab:blue")
ax.plot(features, cae_acc,  marker="s", linewidth=1.5, label="CAE",        color="tab:orange",  linestyle="--")
ax.plot(features, eddi_acc, marker="^", linewidth=1.5, label="EDDI",       color="tab:green",   linestyle="--")
ax.plot(features, ga_acc,   marker="D", linewidth=1.5, label="Greedy-Abl", color="tab:red",     linestyle="--")
ax.plot(features, iter_acc, marker="v", linewidth=1.5, label="Iterative",  color="tab:purple",  linestyle="--")
ax.plot(features, gdfs_acc, marker="X", linewidth=1.5, label="GDFS",       color="tab:brown",   linestyle="--")

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
