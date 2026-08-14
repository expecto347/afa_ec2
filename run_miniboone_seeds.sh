#!/bin/bash
#SBATCH -J AFA_MINIBOONE
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

# MiniBooNE: balanced binary classification on 20 STG-selected features
# (see prepare_miniboone.py; downloads from UCI on first run).
# Task 0 runs prepare_miniboone.py; other tasks wait 180 s.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_miniboone_seed${SEED}"

cd ~/AFA_EC2

if [ "$SEED" -eq 0 ]; then
    apptainer exec --nv afa_env.sif python -u prepare_miniboone.py
else
    echo "Task ${SEED}: waiting 180 s for data preparation by task 0..."
    sleep 180
fi

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/miniboone/train.csv \
    --val_csv   /data/users/haozhes/miniboone/val.csv \
    --label_col label \
    --k 50 --knn_subsample 10000 --patience 20 \
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
    --test_csv   /data/users/haozhes/miniboone/test.csv \
    --label_col  label
