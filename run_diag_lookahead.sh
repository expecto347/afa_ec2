#!/bin/bash
#SBATCH -J DIAG_LOOKAHEAD
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH -p long
#SBATCH -o logs/%x_%j.out
cd ~/AFA_EC2
apptainer exec afa_env.sif python -u diagnose_lookahead.py
