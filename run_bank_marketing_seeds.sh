#!/bin/bash
#SBATCH -J AFA_BANK
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

# Prerequisite: place bank-additional-full.csv at
#   /data/users/haozhes/bank_marketing/bank-additional-full.csv
# Download from https://archive.ics.uci.edu/dataset/222/bank+marketing
#
# This script is submitted with --dependency=afterok on all other dataset jobs,
# so it only starts after Grid Stability, CUBE-NM, CKD, and Metabric all finish.
# Task 0 runs prepare_bank_marketing.py; other tasks wait 120 s.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_bank_marketing_seed${SEED}"

cd ~/AFA_EC2

if [ "$SEED" -eq 0 ]; then
    apptainer exec --nv afa_env.sif python -u prepare_bank_marketing.py
else
    echo "Task ${SEED}: waiting 120 s for data preparation by task 0..."
    sleep 120
fi

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/bank_marketing/train.csv \
    --val_csv   /data/users/haozhes/bank_marketing/val.csv \
    --label_col label \
    --max_features 10 \
    --k 50 --knn_subsample 5000 --patience 20 \
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
    --test_csv   /data/users/haozhes/bank_marketing/test.csv \
    --label_col  label
