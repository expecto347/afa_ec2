import matplotlib.pyplot as plt

# ================= 数据准备 =================
# 特征数量 (X轴)
num_features = list(range(13))

# Accuracy 数据 (Y轴)
# 已更新为最新提供的 EC2 和 Rand 数据
ec2_acc =  [0.6530, 0.7050, 0.7300, 0.7650, 0.7780, 0.7970, 0.8120, 0.8670, 0.8970, 0.9280, 0.9400, 0.9450, 0.9440]
rand_acc = [0.6530, 0.6590, 0.6670, 0.7120, 0.7410, 0.7400, 0.7510, 0.7680, 0.8000, 0.8310, 0.8600, 0.9050, 0.9440]

# 保持不变的 DIME 和 EC2 with PVAE 数据
dime_acc = [0.6530, 0.6810, 0.7010, 0.7430, 0.7920, 0.8560, 0.9110, 0.9440, 0.9590, 0.9510, 0.9430, 0.9340, 0.9310]
ec2_pvae_acc = [0.6490, 0.6650, 0.6800, 0.7030, 0.7310, 0.7980, 0.8500, 0.8980, 0.9300, 0.9500, 0.9600, 0.9600, 0.9590]

# AUROC 数据 (Y轴)
# 已更新为最新提供的 EC2 和 Rand 数据
ec2_auroc =  [0.5000, 0.6763, 0.7472, 0.8078, 0.8385, 0.8734, 0.8891, 0.9273, 0.9652, 0.9793, 0.9892, 0.9932, 0.9937]
rand_auroc = [0.5000, 0.5947, 0.6557, 0.7171, 0.7579, 0.7792, 0.8114, 0.8450, 0.8719, 0.9081, 0.9346, 0.9663, 0.9937]

# 保持不变的 DIME 和 EC2 with PVAE 数据
dime_auroc = [0.5000, 0.6378, 0.7118, 0.7862, 0.8587, 0.9154, 0.9693, 0.9900, 0.9947, 0.9925, 0.9907, 0.9882, 0.9846]
ec2_pvae_auroc = [0.5176, 0.6256, 0.6239, 0.6952, 0.7613, 0.8311, 0.9016, 0.9421, 0.9776, 0.9878, 0.9813, 0.9715, 0.9531]


# ================= 绘图设置 =================
# 设置全局字体大小和图表样式
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# ----- 图 1: Accuracy 对比 -----
ax1.plot(num_features, ec2_acc, marker='o', linewidth=2, label='EC2 Accuracy')
ax1.plot(num_features, dime_acc, marker='s', linewidth=2, label='DIME Accuracy')
ax1.plot(num_features, ec2_pvae_acc, marker='d', linewidth=2, label='EC2 with PVAE Accuracy', color='purple')
ax1.plot(num_features, rand_acc, marker='^', linewidth=2, label='Rand Accuracy', linestyle='--')

ax1.set_title('Test Accuracy vs. Number of Features Selected', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Features Selected', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_xticks(num_features)
ax1.legend(fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.7)

# ----- 图 2: AUROC 对比 -----
ax2.plot(num_features, ec2_auroc, marker='o', linewidth=2, label='EC2 AUROC')
ax2.plot(num_features, dime_auroc, marker='s', linewidth=2, label='DIME AUROC')
ax2.plot(num_features, ec2_pvae_auroc, marker='d', linewidth=2, label='EC2 with PVAE AUROC', color='purple')
ax2.plot(num_features, rand_auroc, marker='^', linewidth=2, label='Rand AUROC', linestyle='--')

ax2.set_title('Test AUROC vs. Number of Features Selected', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Features Selected', fontsize=12)
ax2.set_ylabel('AUROC', fontsize=12)
ax2.set_xticks(num_features)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.7)

# 调整布局并显示图表
plt.tight_layout()
plt.savefig('feature_selection_comparison.png', dpi=300)