# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All Python scripts must run inside the pre-built Apptainer container:
```bash
apptainer exec --nv afa_env.sif python -u <script.py> [args]
```

The container (`afa_env.sif`) is built from `AFA_EC2.def` and bundles Python 3.12, PyTorch (CUDA 11.8), PyTorch Lightning, and the [AFA-Benchmark](https://github.com/Linusaronsson/AFA-Benchmark) library (imported as `afabench`).

Jobs are submitted to SLURM via shell scripts (`sbatch <script>.sh`). **All training and evaluation must be submitted to SLURM — do not run experiments interactively.**

### SLURM partition rules

| Expected runtime | Partition flag |
|-----------------|---------------|
| ≤ 30 minutes | `-p short` |
| > 30 minutes | `-p long` |

### SLURM scripts

| Script | Dataset | Partition |
|--------|---------|-----------|
| `run_afa.sh` | Grid Stability | `long` |
| `run_cube_nm.sh` | CUBE-NM (afa_context_v2) | `long` |
| `run_syn1.sh` | Syn1 (synthetic) | `short` |
| `run_syn2.sh` | Syn2 (synthetic) | `short` |
| `run_syn3.sh` | Syn3 (synthetic) | `short` |
| `run_metabric.sh` | Metabric | `short` |
| `run_mnist_seeds.sh` | MNIST (20 STG pixels) | `long` |
| `run_california_housing_seeds.sh` | California Housing (4-class) | `long` |
| `run_miniboone_seeds.sh` | MiniBooNE (balanced binary) | `long` |
| `run_fashion_mnist_seeds.sh` | Fashion MNIST (20 STG pixels) | `long` |

Each script trains the model and then auto-evaluates the best checkpoint.

## Running experiments

**Grid Stability (primary active experiment):**
```bash
# Train
apptainer exec --nv afa_env.sif python -u ec2_grid_stability.py --k 50 --patience 20

# Evaluate (replace path with actual checkpoint)
apptainer exec --nv afa_env.sif python -u evaluate_ec2_grid_stability.py \
    --checkpoint logs/ec2_grid_stability_trial_0/ec2_train/checkpoints/best_val_perf_model.ckpt
```

**CUBE-NM (afa_context_v2, kNN EC2):**
```bash
# Train + auto-evaluate via SLURM
sbatch run_cube_nm.sh

# Or run manually:
apptainer exec --nv afa_env.sif python -u ec2_cube_nm.py \
    --k 50 --patience 20 --max_features 14 --batch_size 64

# Evaluate
apptainer exec --nv afa_env.sif python -u evaluate_ec2_cube_nm.py \
    --checkpoint logs/ec2_cube_nm_trial_0/ec2_train/checkpoints/best_val_perf_model.ckpt
```

**AFAContextDataset (Cube, synthetic, neural EC2):**
```bash
# Train
apptainer exec --nv afa_env.sif python -u ec2_cube.py

# Evaluate
apptainer exec --nv afa_env.sif python -u evaluate_ec2.py \
    --ckpt_path <path/to/checkpoint.ckpt>
```

**SEFA synthetic datasets (Syn1, Syn2, Syn3):**
```bash
sbatch run_syn1.sh   # also runs evaluation automatically after training
sbatch run_syn2.sh
sbatch run_syn3.sh
```
Data is generated inline — no external downloads needed.

**SEFA Metabric dataset:**
```bash
sbatch run_metabric.sh
```
Requires preprocessing first (one-time):
1. Download `METABRIC_RNA_Mutation.csv` from Kaggle and place at `/data/users/haozhes/SEFA/datasets/data/metabric/`.
2. `cd /data/users/haozhes/SEFA && PYTHONPATH=. apptainer exec --nv ~/AFA_EC2/afa_env.sif python datasets/create_metabric.py`

**Pretrain only (skip EC2 estimator training):**
```bash
apptainer exec --nv afa_env.sif python -u ec2_grid_stability.py --pretrain_only
```

Logs and checkpoints land in `logs/` (gitignored). TensorBoard logs are stored there alongside SLURM `.out` files.

## Architecture

### Core concepts

**Active Feature Acquisition (AFA):** At inference time, features are acquired sequentially with a budget. The system must decide *which* feature to acquire next to maximize classification accuracy most efficiently.

**EC2 (Expected Cost-Conditional Class purity reduction):** The acquisition criterion. For each candidate feature, EC2 estimates how much class uncertainty would be reduced if that feature were observed. The value network is trained to predict these EC2 scores.

**MaskLayer** (`ec2_cube.py:51`, `ec2_grid_stability.py:53`): Applies a binary mask to features, then concatenates the mask to the input — so all networks take `d_in * 2` dimensional input.

### Three experimental configurations

| | `ec2_cube.py` | `ec2_cube_nm.py` | `ec2_grid_stability.py` |
|---|---|---|---|
| Dataset | `AFAContextDataset` noiseless (CUBE-NM-noiseless, `n_contexts=5`) | `AFAContextDataset` noisy (CUBE-NM / `afa_context_v2`, `n_contexts=5`) | Grid Stability CSV |
| `d_in` / `d_out` | 55 / 8 | 55 / 8 | 12 / 2 |
| Predictor | `TwoHeadMLP` (classification + Gaussian aux head) | `Predictor` (single MLP, classification only) | `Predictor` |
| EC2 targets | Neural: uses predictor aux head + surprise weight `w` | **kNN + Gini binarization** (`knn_ec2_target()`) | kNN + Gini binarization |
| Pretraining | `MaskingPretrainerEC2` (classification + GaussianNLL) | `PredictorPretrainer` (classification only) | `PredictorPretrainer` |
| Label format | one-hot (handled internally) | one-hot → argmax wrapper `CubeNMDataset` | integer (from CSV) |
| Train size | 16000 (20000 × 0.8) | 5000 (6250 × 0.8) | from train.csv |
| Checkpoint monitor | `Perf Val/Final` | `Acc Val/Predictor Mean` | `Acc Val/Predictor Mean` |

### Training pipeline (two-phase)

1. **Predictor pretraining**: The predictor is trained with random masks so it can classify from any subset of features. This phase freezes the predictor's weights before EC2 training.

2. **EC2 Estimator training** (`EC2Estimator`): Only the value network is optimized. In each step, the value network predicts EC2 scores for all features; epsilon-greedy selects actions; targets come from either the neural aux head (cube) or kNN Gini computation (grid stability). The epsilon decays in steps controlled by `eps_steps` and `eps_decay`.

### Key implementation details

- `automatic_optimization = False` in `EC2Estimator` — gradients are accumulated over all `max_features` steps per batch before calling `opt.step()`.
- In the grid stability kNN approach (`knn_ec2_target`), rows with an all-zero mask use the global training distribution (not kNN) to get a stable starting target.
- Checkpoints monitor `Acc Val/Predictor Mean` (grid stability) or `Perf Val/Final` (cube).
- Data for grid stability lives at `/data/users/haozhes/DIME/experiments/grid_stability/data` (train/val/test CSVs). Features are StandardScaler-normalized; the scaler parameters are stored in the Lightning checkpoint as registered buffers.

### Other scripts

- `evaluate_pretraining.py` / `evaluate_predictor.py`: Standalone evaluations of the pretrained predictor at various feature retention fractions.
- `ec2_minst.py`: An older MNIST-based variant (not the active experiment).
- `train_predictor.py`: Standalone MNIST predictor pretraining script (historical).
- `check_cube_balance.py`: Quick dataset inspection utility for `AFAContextDataset`.
- `visualization.py` / `scripts/figure.py`: Plotting utilities for evaluation outputs.

## Bank Marketing 数据集的特殊处理

Bank Marketing 含有大量类别特征（10 个），通过 `prepare_bank_marketing.py` one-hot 展开后共约 63 列。这导致三个问题需要特殊处理，均已在 `ec2_tabular.py` 中实现，对其他数据集**无影响**。

### 问题与修复

**Fix 1 — 不使用类别权重**

Bank Marketing 类别极度不平衡（"yes" 约 11%）。若在 `PredictorPretrainer` 中使用逆频率 `class_weights`，预测器在低 budget 时会过度预测少数类，导致准确率崩溃（88% → 74%），而 AUROC 因为是排名指标反而上升，两者严重分裂。`ec2_tabular.py` 已移除 `class_weights`，用均等损失训练预测器。

**Fix 2 — 仅对连续列做 StandardScaler**

稀有 one-hot 列（如某职业只占 5%）经 StandardScaler 后值可达 +4.36，在 kNN 欧氏距离中远远压制真正的连续特征，扭曲 kNN EC2 目标值。`ec2_tabular.py` 自动检测二值列（`np.isin(col, [0,1])`），只对连续列做标准化，one-hot 列保持原始 0/1。

**Fix 3 — 特征组级别的 acquisition**

若每个 dummy 列算作一个独立 acquisition 步骤，budget=1 只能看到"job_admin."而不是完整的"job"特征，语义错误且效率低。修复方案：

- `prepare_bank_marketing.py` 构建 `feature_groups.json`：10 个连续特征各为单列 group，10 个类别特征各为包含所有 dummy 列的 group，共 20 个 group；保存在数据目录下
- `ec2_tabular.py` 和 `evaluate_ec2_tabular.py` 自动从 train/test CSV 同目录加载 `feature_groups.json`，传入 `EC2Estimator`
- `EC2Estimator`（`ec2_core.py`）：注册 `group_col_matrix` 缓冲区 `[n_groups, d_in]`；`_col_scores_to_group_scores()` 将列级 EC2 分数聚合为 group 分数；`_apply_group_action()` 一次将整个 group 的所有列置为已观测；training/validation step 有 if/else 分支，无 groups 时代码路径与原来完全一致

### 运行方式

```bash
# 一键训练 + 评估（prepare_bank_marketing.py 在脚本内自动运行）
sbatch run_bank_marketing.sh
```

数据目录：`/data/users/haozhes/bank_marketing/`（含 train/val/test.csv 和 feature_groups.json）。

### 各数据集特征类型对比

| 数据集 | 连续特征 | 类别/二值特征 | 是否需要 Fix 1/2/3 |
|--------|---------|-------------|------------------|
| Grid Stability | 全部 12 列 | 无 | 否 |
| Metabric | 全部（z-score 基因表达） | 无（突变列未被选中） | 否 |
| CUBE-NM | 全部 55 列 | 无 | 否 |
| Bank Marketing | 10 列 | 53 个 one-hot dummy 列 | **是** |
| MNIST | 20 列（STG 选出的像素，[0,1] 浮点） | 无 | 否 |
| Fashion MNIST | 20 列（STG 选出的像素，[0,1] 浮点） | 无 | 否 |
| California Housing | 全部 8 列 | 无（label 为 4 等频 bin） | 否 |
| MiniBooNE | 20 列（STG 选出） | 无 | 否 |

## MNIST / Fashion MNIST / California Housing / MiniBooNE 数据集

四个数据集均走通用 CSV 路径（`ec2_tabular.py` + `evaluate_ec2_tabular.py`，与 CKD/Bank Marketing 相同），由对应 `prepare_*.py` 生成 `/data/users/haozhes/{mnist,fashion_mnist,california_housing,miniboone}/{train,val,test}.csv`，预处理与 SEFA/AFA-benchmark 保持一致：

- **MNIST**：展平为 784 维后取 STG 选出的 20 个像素（`prepare_mnist.py` 内 `BEST_FEATURES`），10 类，指标 accuracy，划分 50000/10000/10000（官方 test）。原始文件已在本仓库 `data/MNIST/raw`，不会触发下载。
- **Fashion MNIST**：与 MNIST 相同流程（`prepare_fashion_mnist.py` 内 `BEST_FEATURES`），10 类，指标 accuracy，划分 50000/10000/10000（官方 test）。原始文件由 torchvision 自动下载到 `data/FashionMNIST/raw`（首次运行 prepare 时）。
- **California Housing**：sklearn 自动下载，8 个连续特征；回归目标按分位数离散化为 4 个等频 bin，指标 accuracy，划分 16512/2064/2064。
- **MiniBooNE**：UCI 自动下载（`prepare_miniboone.py` 内 `UCI_ZIP_URL`）；负类降采样至 40000 强制类别平衡（36499 正 / 40000 负），取 STG 选出的 20 个特征，二分类，指标 AUROC，划分 56499/10000/10000。

运行方式（train + eval 一键完成，prepare 脚本由 array task 0 自动执行）：

```bash
sbatch run_mnist_seeds.sh
sbatch run_california_housing_seeds.sh
sbatch run_miniboone_seeds.sh
sbatch run_fashion_mnist_seeds.sh
```

四者也均已注册进 `run_all_seeds.sh` 和 `plot_results.py` 的 `datasets` 列表。
