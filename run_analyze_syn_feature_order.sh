#!/bin/bash
#SBATCH -J ANA_SYN_ORDER
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/analyze_syn_order_%A_%a.out

# Feature-order analysis for Syn1/2/3 (3 seeds each, greedy EC2 on test set).
# Pure evaluation — short partition. One job per dataset.
#   sbatch run_analyze_syn_feature_order.sh

DATASETS=(syn1 syn2 syn3)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u analyze_syn_feature_order.py \
    --dataset "$DATASET" --seeds 0 1 2
