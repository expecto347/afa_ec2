#!/bin/bash
#SBATCH -J AFA_CKD
#SBATCH --array=0-2
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%A_%a.out

# Chronic Kidney Disease dataset (UCI ID 336, 400 samples, 24 features, binary)
# prepare_ckd.py downloads data automatically; only task 0 runs it to avoid
# concurrent writes. Other tasks wait 60 s for the files to appear.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_ckd_seed${SEED}"

cd ~/AFA_EC2

if [ "$SEED" -eq 0 ]; then
    apptainer exec --nv afa_env.sif python -u prepare_ckd.py
else
    echo "Task ${SEED}: waiting 60 s for data preparation by task 0..."
    sleep 60
fi

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/ckd/train.csv \
    --val_csv   /data/users/haozhes/ckd/val.csv \
    --label_col label \
    --k 20 --patience 20 --batch_size 64 \
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
    --test_csv   /data/users/haozhes/ckd/test.csv \
    --label_col  label
