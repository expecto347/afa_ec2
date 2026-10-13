#!/bin/bash
#SBATCH -J DIAG_NONMYOPIC
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%x_%j.out
cd ~/AFA_EC2
apptainer exec --nv afa_env.sif python -u diagnose_syn_nonmyopic.py
