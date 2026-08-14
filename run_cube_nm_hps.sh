#!/bin/bash
#SBATCH -J AFA_CUBE_HPS
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%j.out
#SBATCH --exclude=uranus,neptune   # L40s 节点 CUDA 不稳

# CUBE-NM 超参数搜索单任务。
# 用法: sbatch run_cube_nm_hps.sh <run_name> [extra args...]
#   额外参数直接透传给 ec2_cube_nm.py（如 --k 25 --batch_size 32）

RUN_NAME=$1
shift

cd ~/AFA_EC2

# 从额外参数中提取 --seed（默认 0），训练与 eval 使用同一个 seed，
# 这样不同 seed 的运行会用各自的 test set（test seed = train seed + 100）。
SEED=$(echo "$*" | grep -oE -- '--seed[ =][0-9]+' | grep -oE '[0-9]+' | head -1)
SEED=${SEED:-0}

echo "=== ${RUN_NAME}: $* (seed ${SEED}) ==="

apptainer exec --nv afa_env.sif python -u ec2_cube_nm.py \
    --seed "$SEED" --run_name "$RUN_NAME" "$@"

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_cube_nm.py \
    --checkpoint "$CKPT" --seed "$SEED" \
    --output_plot "logs/${RUN_NAME}/eval_plot.png"

echo "Finished: ${RUN_NAME}"
