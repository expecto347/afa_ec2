#!/bin/bash
#SBATCH -J AFA_MNIST
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

# MNIST reduced to 20 STG-selected pixels (see prepare_mnist.py).
# Raw MNIST files already exist under data/MNIST/raw — no download needed.
# Task 0 runs prepare_mnist.py; other tasks wait 180 s.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_mnist_seed${SEED}"

cd ~/AFA_EC2

if [ "$SEED" -eq 0 ]; then
    apptainer exec --nv afa_env.sif python -u prepare_mnist.py
else
    echo "Task ${SEED}: waiting 180 s for data preparation by task 0..."
    sleep 180
fi

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/mnist/train.csv \
    --val_csv   /data/users/haozhes/mnist/val.csv \
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
    --test_csv   /data/users/haozhes/mnist/test.csv \
    --label_col  label
