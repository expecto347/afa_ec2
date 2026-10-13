#!/bin/bash
#SBATCH -J OL
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%A_%a.out
# OL (Opportunistic Learning) RL baseline via SEFA's implementation.
#   sweep : DATASET=syn2 MODE=sweep  sbatch --array=1-9 -p long -t 8:00:00 run_ol.sh   (array id = config)
#   final : DATASET=syn2 MODE=final  sbatch --array=0-2 -p long -t 8:00:00 run_ol.sh   (array id = seed)
cd ~/AFA_EC2
if [ "$MODE" = "sweep" ]; then ARG="--config $SLURM_ARRAY_TASK_ID"; else ARG="--seed $SLURM_ARRAY_TASK_ID"; fi
apptainer exec --nv afa_env.sif python -u ol_baseline.py --dataset "$DATASET" --mode "$MODE" $ARG $EXTRA
