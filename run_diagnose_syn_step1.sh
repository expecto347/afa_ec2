#!/bin/bash
#SBATCH -J DIAG_SYN_S1
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0:15:00
#SBATCH -p short
#SBATCH -o logs/diag_syn_step1_%A_%a.out

# Dump step-1/step-2 acquisition choices + feature values for syn1/2/3 (seed 0).
#   sbatch run_diagnose_syn_step1.sh

DATASETS=(syn1 syn2 syn3)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u diagnose_syn_step1.py \
    --dataset "$DATASET" --seed 0 --n 3000
