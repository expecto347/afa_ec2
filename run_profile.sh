#!/bin/bash
#SBATCH -J PROFILE
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%x_%j.out
cd ~/AFA_EC2
nvidia-smi --query-gpu=name,utilization.gpu --format=csv
apptainer exec --nv afa_env.sif python -u profile_training.py
