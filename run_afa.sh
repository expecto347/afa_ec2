#!/bin/bash
#SBATCH -J AFA
#SBATCH -c 8                  # 请求 8 个 CPU 核心
#SBATCH --mem=32G             # 请求 32GB 内存
#SBATCH --gres=gpu:1          # 请求 1 个 GPU
#SBATCH -t 30:00           # 运行 30 分钟
#SBATCH -p short              # 使用 short 分区
#SBATCH -o logs/%j.out        # 日志输出

# 1. 进入代码目录
cd ~/AFA_EC2

# 3. 运行你的训练脚本
apptainer exec --nv afa_env.sif python ec2_cube.py
# apptainer exec --nv afa_env.sif python evaluate_ec2.py --ckpt_path 'logs/ec2_max_features_15_trial_0/version_4/checkpoints/best_val_perf_model.ckpt' --alpha_list 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 --output_plot test_accuracy_curve_multi_alpha_1.png
# apptainer exec --nv afa_env.sif python evaluate_ec2.py --ckpt_path 'logs/ec2_max_features_15_trial_0/version_7/checkpoints/best_val_perf_model.ckpt'
# apptainer exec --nv afa_env.sif python check_cube_balance.py

# apptainer exec --nv afa_env.sif python evaluate_pretraining.py \
#   --ckpt_path logs/ec2_max_features_15_trial_0/pretrain/checkpoints/epoch=33-step=2108.ckpt \
#   --output_dir pretrain_eval_outputs