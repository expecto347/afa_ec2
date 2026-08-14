#!/bin/bash
# 提交快速数据集的 3×3 超参数网格（seed 0）
# ckd + metabric + grid_stability × (k∈{5,20,100} × bins∈{4,20,40}) = 27 个 job

DATASETS="ckd metabric grid_stability"
K_VALUES="5 20 100"
BINS_VALUES="4 20 40"

JOBS=0
for DS in $DATASETS; do
    for K in $K_VALUES; do
        for BINS in $BINS_VALUES; do
            JOBID=$(sbatch --parsable run_hyperparam_search.sh $DS $K $BINS 0)
            echo "Submitted $DS k=$K bins=$BINS seed=0 -> $JOBID"
            JOBS=$((JOBS + 1))
        done
    done
done
echo "Total submitted: $JOBS jobs"
