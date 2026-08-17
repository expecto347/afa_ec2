#!/bin/bash
#SBATCH -J AFA_SYN
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 4:30:00
#SBATCH -p long
#SBATCH -o logs/%A_%a.out

# Syn1/2/3 (synthetic, generated inline by ec2_sefa.py) — 3 seeds per dataset.
# Usage: DATASET=syn1 bash run_syn_seeds.sh   (or set DATASET before sbatch)
#   DATASET=syn1 sbatch run_syn_seeds.sh

DATASET=${DATASET:?must be set to syn1/syn2/syn3}
SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_${DATASET}_seed${SEED}"

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
    --dataset "$DATASET" --run_name "$RUN_NAME" --patience 20 \
    --seed $SEED

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
    --checkpoint "$CKPT" --dataset "$DATASET"
