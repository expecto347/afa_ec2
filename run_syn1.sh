#!/bin/bash
#SBATCH -J AFA_SYN1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 4:30:00
#SBATCH -p long
#SBATCH -o logs/%j.out

cd ~/AFA_EC2

apptainer exec --nv afa_env.sif python -u ec2_sefa.py \
    --dataset syn1 --run_name ec2_syn1_trial_0 --patience 20

CKPT_DIR="logs/ec2_syn1_trial_0/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_sefa.py \
    --checkpoint "$CKPT" --dataset syn1
