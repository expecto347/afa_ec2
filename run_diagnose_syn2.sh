#!/bin/bash
#SBATCH -J DIAG_SYN2
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 10:00
#SBATCH -p short
#SBATCH -o logs/%j.out

cd ~/AFA_EC2

CKPT="logs/ec2_syn2_trial_0/ec2_train/checkpoints/best_val_perf_model-v1.ckpt"

apptainer exec --nv afa_env.sif python -u diagnose_syn2.py \
    --checkpoint "$CKPT" \
    --dataset syn2
