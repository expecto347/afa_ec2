#!/bin/bash
# 提交 3×3 超参数网格（跳过默认组合 k=50/bins=10，已有 3-seed 结果作对照）
# 3 数据集 × 8 组合 × seed 0 = 24 个 job，每 job 1 GPU 并行

DATASETS="mnist california_housing miniboone"
K_VALUES="5 20 100"
BINS_VALUES="4 20 40"

JOBS=0
for DS in $DATASETS; do
    for K in $K_VALUES; do
        for BINS in $BINS_VALUES; do
            if [ "$K" = "50" ] && [ "$BINS" = "10" ]; then
                continue  # 默认组合已有结果
            fi
            JOBID=$(sbatch --parsable run_hyperparam_search.sh $DS $K $BINS 0)
            echo "Submitted $DS k=$K bins=$BINS seed=0 -> $JOBID"
            JOBS=$((JOBS + 1))
        done
    done
done
echo "Total submitted: $JOBS jobs"
