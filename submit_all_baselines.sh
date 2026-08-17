#!/bin/bash
# 提交 7 个表格数据集 (grid_stability/ckd/metabric/bank_marketing/mnist/california_housing/miniboone)
# × 3 个 baseline 仓库 (dynamic-selection/AACO/DIME) × 3 seeds 的全部运行。
# EC2 3-seed 结果已存在，无需重跑。
#
# Usage: bash submit_all_baselines.sh
# 每个 sbatch 都是 array 0-2（3 个 seed），并行利用集群多 GPU。

DATASETS="grid_stability ckd metabric bank_marketing mnist california_housing miniboone"
METHODS="cae eddi greedy_ablation iterative gdfs"

set -e
echo "=== DS baselines (5 methods x 7 datasets, seeds 0-2) ==="
cd /data/users/haozhes/dynamic-selection/experiments/tabular
for ds in $DATASETS; do
  for m in $METHODS; do
    JOB=$(DATASET=$ds METHOD=$m sbatch --parsable run_ds_seeds.sh)
    echo "  DS $ds/$m: ${JOB}"
  done
done

echo "=== AACO baselines (7 datasets, seeds 0-2) ==="
cd /data/users/haozhes/aaco
for ds in $DATASETS; do
  JOB=$(DATASET=$ds sbatch --parsable run_aaco_tabular_seeds.sh)
  echo "  AACO $ds: ${JOB}"
done

echo "=== DIME baselines (7 datasets, seeds 0-2) ==="
cd /data/users/haozhes/DIME
for ds in $DATASETS; do
  JOB=$(DATASET=$ds sbatch --parsable run_dime_seeds.sh)
  echo "  DIME $ds: ${JOB}"
done

echo ""
echo "All baseline jobs submitted. Monitor with: squeue -u \$USER"
