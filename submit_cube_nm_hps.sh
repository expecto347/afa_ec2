#!/bin/bash
# 提交 CUBE-NM 超参数搜索 wave 1（全部 seed 0，每 job 1 GPU，并行执行）。
# 基线对照: ec2_cube_nm_trial_0 (k=50, bs=64, mf=55, eps=0.1, bins=10) → AUC 0.9265

cd ~/AFA_EC2

declare -a JOBS=(
  # name                   extra args
  "cube_hps_base           --k 50   --batch_size 64 --max_features 55 --eps 0.1 --n_bins 10"
  "cube_hps_k25            --k 25   --batch_size 64 --max_features 55 --eps 0.1 --n_bins 10"
  "cube_hps_k100           --k 100  --batch_size 64 --max_features 55 --eps 0.1 --n_bins 10"
  "cube_hps_rfst           --k 50   --batch_size 64 --max_features 55 --eps 0.1 --n_bins 10 --random_first_step"
  "cube_hps_eps02          --k 50   --batch_size 64 --max_features 55 --eps 0.2 --n_bins 10 --random_first_step"
  "cube_hps_bs32           --k 50   --batch_size 32 --max_features 55 --eps 0.1 --n_bins 10"
  "cube_hps_mf20           --k 50   --batch_size 64 --max_features 20 --eps 0.1 --n_bins 10"
  "cube_hps_bins20         --k 50   --batch_size 64 --max_features 55 --eps 0.1 --n_bins 20"
)

N=0
for entry in "${JOBS[@]}"; do
    read -r NAME EXTRA <<< "$entry"
    JOBID=$(sbatch --parsable run_cube_nm_hps.sh "$NAME" $EXTRA)
    echo "Submitted $NAME ($EXTRA) -> $JOBID"
    N=$((N + 1))
done
echo "Total submitted: $N jobs"
