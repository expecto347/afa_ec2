#!/bin/bash
#SBATCH -J DIAG_S5
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -p short
#SBATCH -o logs/%x_%j.out
cd ~/AFA_EC2
R="apptainer exec --nv afa_env.sif python -u"
echo "=== first-step choices"; $R diagnose_first_step.py _s5 _s5ft _s5laft
echo "=== target agreement (two-phase S5)"; RUN_TAG=_tp $R diagnose_target_agreement.py
echo "=== Bayes oracle (two-phase S5, seed 0)"; RUN_TAG=_tp $R diagnose_syn_nonmyopic.py
