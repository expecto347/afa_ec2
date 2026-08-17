#!/bin/bash
#SBATCH -J AFA_SYN_HPS
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 4:30:00
#SBATCH -p long
#SBATCH --exclude=neptune,uranus
#SBATCH -o logs/%x_%A_%a.out

# syn2 EC2 超参搜索: 5 组配置 (HPS_NAME=hps1..hps5)。
# 数据 CSV 由 prepare_syn.py 生成(与 baseline 一致)。
# Usage: DATASET=syn2 HPS_NAME=hps1 sbatch run_syn_hps_seeds.sh
#        SEED_OFFSET=N 可将 array 任务号平移(补跑更多 seed 时用)

DATASET=${DATASET:?must be set to syn1/syn2/syn3}
HPS_NAME=${HPS_NAME:?must be set to hps1..hps5}
SEED=$((SLURM_ARRAY_TASK_ID + ${SEED_OFFSET:-0}))
RUN_NAME="ec2_${DATASET}_${HPS_NAME}_seed${SEED}"

case "$HPS_NAME" in
    hps1) K=100; NBINS=10; DROPOUT=0.3 ;;  # 更多 kNN 邻居
    hps2) K=50;  NBINS=30; DROPOUT=0.3 ;;  # 更细 Gini 分箱
    hps3) K=20;  NBINS=10; DROPOUT=0.3 ;;  # 更少邻居
    hps4) K=50;  NBINS=10; DROPOUT=0.1 ;;  # 更低 dropout (syn2 规则简单)
    hps5) K=150; NBINS=10; DROPOUT=0.3 ;;  # 继续增大 k
    *) echo "Unknown HPS_NAME=$HPS_NAME"; exit 1 ;;
esac

cd ~/AFA_EC2

echo "=== ${DATASET} ${HPS_NAME} seed${SEED}: k=${K} n_bins=${NBINS} dropout=${DROPOUT} ==="

apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
    --dataset "$DATASET" --run_name "$RUN_NAME" --patience 20 \
    --k $K --n_bins $NBINS --dropout $DROPOUT --seed $SEED

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
    --checkpoint "$CKPT" --dataset "$DATASET"
