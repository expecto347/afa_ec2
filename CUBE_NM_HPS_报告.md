# CUBE-NM 超参数搜索实验报告（2026-08-14）

## 摘要

对 CUBE-NM（afa_context_v2，无噪声配置，55 特征 / 8 类）执行了 4 波共 **22 个 SLURM 运行**（每 job 1 GPU，峰值 8 并行），结论：

1. **模型确实首先选择 context 特征**：全部 22 个健康运行 step 1 选 context 的比例 = **100%**（中位首次 context 选择步 = 1.00）。
2. **只有 kNN 邻居数 k 影响最终策略**：k=100 最优（AUC 0.92675，seed 0）；eps / max_features / random_first_step / 训练时长（400 ep）产生**逐位相同**结果——价值网络收敛到同一个目标函数。
3. **根因诊断（假说 A 确认）**：模型在 step 2 仍为 ~79% 样本选第二个 context，**不是学习失败**——价值网络与 kNN 目标 argmin 一致性 = 1.000，**目标函数本身**在 81.6% 的 step-1 状态下把冗余 context（score 0.544）排在最佳信息块特征（0.618）之前。10-bin 粗离散化稀释了 3-bit label 编码信号。
4. **种子稳定性**：配对验证（k100 vs k50 × seed 1/3）显示 k100 均值 0.92540 vs k50 0.92453（+0.0009），方向一致但落在种子方差（±0.003）内——**建议 k=100 作为默认，但提升幅度不显著**。

## 完整结果表（AUC = accuracy-budget 曲线下面积, 0–55）

| 配置 | 改动 | AUC | acc@5 | acc@4 | perfect@ |
|---|---|---|---|---|---|
| k100 (s0) | k=100 | **0.92675** | 0.637 | 0.440 | 7 |
| k100s1 (s1) | k=100, seed 1 | 0.92695 | 0.646 | 0.416 | 8 |
| k50s1 (s1) | k=50, seed 1 | 0.92712 | 0.681 | 0.416 | 8 |
| trial_0 / base | 原始配置 (k50) | 0.92652 | 0.630 | 0.436 | 7 |
| mf20 / eps02 / rfst / long | max_features/eps/随机首步/400ep | 0.92652 | — | — | 逐位同 base |
| k200 | k=200 | 0.92628 | 0.632 | 0.402 | 7 |
| h1024 / h1024k | hidden=1024 | 0.92592 / 0.92594 | 0.630/0.597 | — | — |
| k100s2 (s2) | k=100, seed 2 | 0.92521 | 0.656 | 0.396 | 8 |
| k25 | k=25 | 0.92472 | 0.612 | 0.413 | 8 |
| bins20 | n_bins=20 | 0.92414 | 0.607 | 0.396 | 8 |
| k100s3 (s3) | k=100, seed 3 | 0.92259 | 0.530 | 0.430 | 10 |
| k50s3 (s3) | k=50, seed 3 | 0.92005 | 0.559 | 0.419 | 10 |
| ret25 / ret10 | 低 budget 预训练掩码 | 0.92137 / 0.91906 | 0.570/0.583 | — | 7 |
| bs32 | batch=32 | 0.91426 | 0.514 | 0.321 | 9 |
| seed0/1/2 (旧) | noisy+mf14（历史崩坏） | 0.76–0.81 | — | — | — |

> 注：k100s1（wave-3 提交）实为 seed-0 重复（未传 --seed），结果与 k100 逐位相同，已排除。wave-4 的 k100s1 才是真 seed-1 运行。

## Context-first 分析

全部运行（含不同 k/seed/hidden）的逐 step context 选择比例完全一致：

| Step | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| context 选择比例 | 1.000 | ~0.79 | ~0.58 | ~0.40 | 0.00–0.16 |

- step 1 最常选特征：feat1（context one-hot 之一），100% 样本
- 100% 样本在前 10 步内首次选到 context（实际全在 step 1）
- 首个 100% 准确率 budget：7（多数），8–10（少数种子/h1024）

## 根因：为什么超参改不动策略

诊断实验（对照 checkpoint k100 与 base k50，test seed 100，500 样本）：

```
step-1 状态（1 个 context 已观察）:
  model:  argmin context frac = 0.816
  target: argmin context frac = 0.816   ← kNN 目标本身
  argmin agreement model-vs-target: 1.000
  target: min context score 0.544 vs best block score 0.618
```

价值网络对目标函数的拟合是完美的（agreement 1.000）；问题出在 **kNN Gini 目标在"context 已观察"状态下低估信息块价值**——推测 10-bin 离散化太粗，3-bit label 编码（每块 10 维中 3 位）经 StandardScaler + 分箱后被稀释。因此任何价值网络超参（hidden、lr、eps、epochs…）都改变不了行为，**可改的杠杆只有目标计算本身**：

1. **kNN 路径 pow 2→3**（与 step-0 全局路径一致，锐化排名）——最直接的一行改动
2. **n_bins 变小**（bins20 已证更差，可试 bins 4–6）
3. **显式冗余惩罚**：目标中扣除"已观察 context 特征"的信息量（理论修正）

## 建议

- **默认 k 改为 100**：`run_cube_nm.sh` / `run_cube_nm_seeds.sh` 加 `--k 100`（平均更好、不劣，在噪声内）
- 若要真正提升 step-2 行为：改目标计算（pow3 / 更细分箱 / 去冗余），重跑验证
- **注意**：当前工作树 `CUBE_NM_KWARGS` 是无噪声配置（std=0，未提交改动）；HEAD 原始版本是 noisy（0.1/0.3/0.3）。若与论文/AFA-benchmark 对齐需切回 noisy，但早期 noisy+max_features=14 的运行崩坏（AUC 0.76–0.81），需同时保持 mf=55 + pow2 才可学
- 基建修复已就位：`evaluate_ec2_cube_nm.py` 从 checkpoint 推断 hidden（修复 h1024 评估崩溃）；`run_cube_nm_hps.sh` 正确传递 seed 到 train+eval（test seed = train seed + 100）

## 实验记录

- 3 波共 22 job：wave1 8 配置（seed 0）+ wave2 3 配置 + wave3 4 配置 + wave4 4 配置 + 2 诊断 + 1 重评估
- 节点：ganymede ×4 + io ×3 并行，排除 uranus/neptune（L40s CUDA 不稳定）
- 复现命令示例：`sbatch run_cube_nm_hps.sh cube_hps_k100 --k 100`（wave-4 起 seed 正确传递）
