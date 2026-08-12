#!/bin/bash
#SBATCH -J AFA_FMNIST
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

# Fashion MNIST reduced to 20 STG-selected pixels (see prepare_fashion_mnist.py).
# Raw files are downloaded once by torchvision into data/FashionMNIST/raw.
# Task 0 runs prepare_fashion_mnist.py (skips if the CSVs already exist);
# other tasks wait 180 s.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_fashion_mnist_seed${SEED}"

cd ~/AFA_EC2

if [ "$SEED" -eq 0 ]; then
    if [ -f /data/users/haozhes/fashion_mnist/train.csv ] && \
       [ -f /data/users/haozhes/fashion_mnist/val.csv ] && \
       [ -f /data/users/haozhes/fashion_mnist/test.csv ]; then
        echo "Fashion MNIST CSVs already exist — skipping prepare_fashion_mnist.py."
    else
        apptainer exec --nv afa_env.sif python -u prepare_fashion_mnist.py
    fi
else
    echo "Task ${SEED}: waiting 180 s for data preparation by task 0..."
    sleep 180
fi

apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
    --train_csv /data/users/haozhes/fashion_mnist/train.csv \
    --val_csv   /data/users/haozhes/fashion_mnist/val.csv \
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
    --test_csv   /data/users/haozhes/fashion_mnist/test.csv \
    --label_col  label
