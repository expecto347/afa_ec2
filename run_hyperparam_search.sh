#!/bin/bash
#SBATCH -J AFA_HPS
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 3:30:00
#SBATCH -p long
#SBATCH -o logs/%j.out

# EC2 超参数搜索单任务（k × n_bins）。
# 用法: sbatch run_hyperparam_search.sh <dataset> <k> <n_bins> <seed>
#   CSV 路径数据集: mnist / california_housing / miniboone / ckd  (ec2_tabular.py)
#   专用脚本数据集: grid_stability / metabric / syn1 / syn2 / syn3

DATASET=$1
K=$2
BINS=$3
SEED=$4

RUN_NAME="ec2_${DATASET}_k${K}_b${BINS}_seed${SEED}"

cd ~/AFA_EC2

case "$DATASET" in
  mnist|california_housing|miniboone|ckd)
    KNN_ARGS=""
    if [ "$DATASET" = "mnist" ] || [ "$DATASET" = "miniboone" ]; then
        KNN_ARGS="--knn_subsample 10000"
    fi
    BS=128
    if [ "$DATASET" = "ckd" ]; then
        BS=64   # 小数据集: 279 样本，batch 128 会因 drop_last 丢太多
    fi
    echo "=== ${RUN_NAME}: k=${K} n_bins=${BINS} seed=${SEED} (ec2_tabular) ==="
    apptainer exec --nv afa_env.sif python -u ec2_tabular.py \
        --train_csv /data/users/haozhes/${DATASET}/train.csv \
        --val_csv   /data/users/haozhes/${DATASET}/val.csv \
        --label_col label \
        --k ${K} --n_bins ${BINS} --patience 20 --batch_size ${BS} ${KNN_ARGS} \
        --seed ${SEED} --run_name "${RUN_NAME}"
    CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
    CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
    echo "Evaluating checkpoint: $CKPT"
    apptainer exec --nv afa_env.sif python -u evaluate_ec2_tabular.py \
        --checkpoint "$CKPT" \
        --test_csv  /data/users/haozhes/${DATASET}/test.csv \
        --label_col label
    ;;

  grid_stability)
    echo "=== ${RUN_NAME}: k=${K} n_bins=${BINS} seed=${SEED} (ec2_grid_stability) ==="
    apptainer exec --nv afa_env.sif python -u ec2_grid_stability.py \
        --k ${K} --n_bins ${BINS} --patience 20 --seed ${SEED} --run_name "${RUN_NAME}"
    CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
    CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
    echo "Evaluating checkpoint: $CKPT"
    apptainer exec --nv afa_env.sif python -u evaluate_ec2_grid_stability.py --checkpoint "$CKPT"
    ;;

  metabric|syn1|syn2|syn3)
    EXTRA=""
    if [ "$DATASET" = "metabric" ]; then
        EXTRA="--dropout 0.0 --random_first_step --train_epochs 300"
    fi
    echo "=== ${RUN_NAME}: k=${K} n_bins=${BINS} seed=${SEED} (ec2_sefa) ==="
    apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
        --dataset ${DATASET} --k ${K} --n_bins ${BINS} --patience 20 \
        --seed ${SEED} --run_name "${RUN_NAME}" ${EXTRA}
    CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
    CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found in ${CKPT_DIR}"
        exit 1
    fi
    echo "Evaluating checkpoint: $CKPT"
    apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
        --checkpoint "$CKPT" --dataset ${DATASET}
    ;;

  *)
    echo "Unknown dataset: ${DATASET}"
    exit 1
    ;;
esac

echo "Finished: ${RUN_NAME}"
