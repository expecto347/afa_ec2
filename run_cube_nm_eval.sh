#!/bin/bash
#SBATCH -J AFA_CUBE_NM_EVAL
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -p short
#SBATCH -o /data/users/haozhes/AFA_EC2/logs/%j.out

# 仅评估已训练好的 CUBE-NM checkpoint：从 budget 0 一直采到全部 55 个特征用完。
# 用法: sbatch run_cube_nm_eval.sh <seed>

SEED=$1
RUN_NAME="ec2_cube_nm_seed${SEED}"

cd ~/AFA_EC2

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
