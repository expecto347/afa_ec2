#!/bin/bash
#SBATCH -J DIAG_CUBE
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%x_%j.out
cd ~/AFA_EC2
apptainer exec --nv afa_env.sif python -u diagnose_cube_step0.py
