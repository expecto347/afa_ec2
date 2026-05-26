#!/bin/bash
#SBATCH -J AFA_BANK_ANALYZE
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0:10:00
#SBATCH -p short
#SBATCH -o logs/%j.out

cd ~/AFA_EC2
apptainer exec --nv afa_env.sif python -u analyze_bank_feature_order.py
