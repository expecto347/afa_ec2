#!/bin/bash
#SBATCH -J AFA_CAL_HOUSING
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

# California Housing: regression target discretized into 4 equally-sized
# bins (see prepare_california_housing.py; downloads via sklearn on first run).
# Task 0 runs prepare_california_housing.py; other tasks wait 120 s.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_california_housing_seed${SEED}"

cd ~/AFA_EC2

if [ "$SEED" -eq 0 ]; then
    apptainer exec --nv afa_env.sif python -u prepare_california_housing.py
else
    echo "Task ${SEED}: waiting 120 s for data preparation by task 0..."
    sleep 120
fi

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/california_housing/train.csv \
    --val_csv   /data/users/haozhes/california_housing/val.csv \
    --label_col label \
    --k 50 --patience 20 \
    --batch_size 128 \
    --seed $SEED --run_name "$RUN_NAME"

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_tabular.py \
    --checkpoint "$CKPT" \
    --test_csv   /data/users/haozhes/california_housing/test.csv \
    --label_col  label
