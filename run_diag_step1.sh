#!/bin/bash
#SBATCH -J AFA_DIAG
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 15:00
#SBATCH -p short
#SBATCH -o logs/%j.out

cd ~/AFA_EC2
CKPT=$1
K=${2:-100}
echo "=== diag step1-state: $CKPT (k=$K) ==="
apptainer exec --nv afa_env.sif python -u /data/users/haozhes/.claude/jobs/76a58c3e/tmp/diag_step1_state.py "$CKPT" "$K"
