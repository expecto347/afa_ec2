#!/bin/bash
#SBATCH -J AFA_CKD
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%j.out

# Chronic Kidney Disease dataset (UCI ID 336, 400 samples, 24 features, binary)
# Data is downloaded automatically via ucimlrepo if not already present.

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u prepare_ckd.py

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/ckd/train.csv \
    --val_csv   /data/users/haozhes/ckd/val.csv \
    --label_col label \
    --k 20 \
    --patience 20 \
    --batch_size 64 \
    --run_name ec2_ckd_trial_0

CKPT_DIR="logs/ec2_ckd_trial_0/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_tabular.py \
    --checkpoint "$CKPT" \
    --test_csv   /data/users/haozhes/ckd/test.csv \
    --label_col  label
