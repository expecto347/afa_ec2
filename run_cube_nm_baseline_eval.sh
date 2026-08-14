#!/bin/bash
#SBATCH -J AFA_CUBE_EVAL
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 30:00
#SBATCH -p short
#SBATCH -o logs/%j.out

# 用新版 evaluate_ec2_cube_nm.py（含 context-first 统计）重评既有 checkpoint。

cd ~/AFA_EC2
CKPT=$1

echo "Re-evaluating: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_cube_nm.py \
    --checkpoint "$CKPT" --seed 0 \
    --output_plot logs/cube_nm_trial0_reeval.png
