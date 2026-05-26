#!/bin/bash
#SBATCH -J AFA_BANK
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%j.out

# Prerequisite: place bank-additional-full.csv at
#   /data/users/haozhes/bank_marketing/bank-additional-full.csv
# Download from https://archive.ics.uci.edu/dataset/222/bank+marketing

cd ~/AFA_EC2

# apptainer exec --nv afa_env.sif python -u prepare_bank_marketing.py

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/bank_marketing/train.csv \
    --val_csv   /data/users/haozhes/bank_marketing/val.csv \
    --label_col label \
    --max_features 10 \
    --k 50 --knn_subsample 5000 --patience 20 \
    --batch_size 128 \
    --run_name ec2_bank_marketing_trial_0

CKPT_DIR="logs/ec2_bank_marketing_trial_0/ec2_train/checkpoints"
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
