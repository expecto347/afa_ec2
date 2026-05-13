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
