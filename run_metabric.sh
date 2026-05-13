#!/bin/bash
#SBATCH -J AFA_METABRIC
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -p short
#SBATCH -o logs/%j.out

# Prerequisite: Metabric .pt files must exist at
#   /data/users/haozhes/SEFA/datasets/data/metabric/
# See CLAUDE.md for download and preprocessing instructions.

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
    --dataset metabric --run_name ec2_metabric_trial_6 --patience 20 \
    --k 15 --pretrain_epochs 100 --dropout 0.0 --random_first_step

CKPT_DIR="logs/ec2_metabric_trial_6/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
    --checkpoint "$CKPT" --dataset metabric --hidden 512
