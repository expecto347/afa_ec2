#!/bin/bash
#SBATCH -J AFA
#SBATCH -c 8                  # 请求 8 个 CPU 核心
#SBATCH --mem=32G             # 请求 32GB 内存
#SBATCH --gres=gpu:1          # 请求 1 个 GPU
#SBATCH -t 60:00           # 运行 60 分钟
#SBATCH -p long              # 使用 long 分区
#SBATCH -o logs/%j.out        # 日志输出

# 1. 进入代码目录
cd ~/AFA_EC2

# 3. 训练（Predictor 预训练 + EC2 kNN 离散化训练，k=100 邻居，19 阈值）
apptainer exec --nv afa_env.sif python -u ec2_grid_stability.py --k 50 --patience 20

# 4. 自动找到最新的 checkpoint 并评估
CKPT_DIR="logs/ec2_grid_stability_trial_0/ec2_train/checkpoints"
CKPT=$(ls -t "${CKPT_DIR}"/best_val_perf_model*.ckpt 2>/dev/null | head -1)

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "Evaluating checkpoint: $CKPT"
apptainer exec --nv afa_env.sif python -u evaluate_ec2_grid_stability.py --checkpoint "$CKPT"
