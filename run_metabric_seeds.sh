#!/bin/bash
#SBATCH -J AFA_METABRIC
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -p short
#SBATCH -o logs/%A_%a.out

# Prerequisite: Metabric .pt files must exist at
#   /data/users/haozhes/SEFA/datasets/data/metabric/
# See CLAUDE.md for download and preprocessing instructions.

SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_metabric_seed${SEED}"

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
    --dataset metabric --run_name "$RUN_NAME" --patience 20 \
    --k 15 --pretrain_epochs 100 --dropout 0.0 --random_first_step \
    --train_epochs 300 --seed $SEED

CKPT_DIR="logs/${RUN_NAME}/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
    --checkpoint "$CKPT" --dataset metabric --hidden 512
