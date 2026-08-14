#!/bin/bash
#SBATCH -J AFA_HPS_EVAL
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -p long
#SBATCH -o logs/%j.out
#SBATCH --exclude=uranus,neptune   # L40s 节点 CUDA 不稳

# 仅评估已训练好的 HPS checkpoint（训练已完成的失败 job 重跑 eval）。
# 用法: sbatch run_hps_eval_only.sh <dataset> <k> <n_bins> <seed>

DATASET=$1
K=$2
BINS=$3
SEED=$4

RUN_NAME="ec2_${DATASET}_k${K}_b${BINS}_seed${SEED}"

cd ~/AFA_EC2

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "Evaluating checkpoint: $CKPT"

case "$DATASET" in
  grid_stability)
    apptainer exec --nv afa_env.sif python -u evaluate_ec2_grid_stability.py --checkpoint "$CKPT"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    exit 1
    ;;
esac

echo "Finished: ${RUN_NAME}"
