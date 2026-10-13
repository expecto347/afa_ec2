#!/bin/bash
#SBATCH -J DIAG_FIRST
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%x_%j.out
cd ~/AFA_EC2
apptainer exec --nv afa_env.sif python -u diagnose_first_step.py "" _ft _uft _urft
