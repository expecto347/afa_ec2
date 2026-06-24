#!/bin/bash
#SBATCH -J AFA_GS
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_grid_stability_seed${SEED}"

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u ec2_grid_stability.py \
    --k 50 --patience 20 --seed $SEED --run_name "$RUN_NAME"

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_grid_stability.py \
    --checkpoint "$CKPT"
