#!/bin/bash
# Submit all 5 datasets with 3 seeds each (array jobs), all in parallel.
#
# Usage:
#   bash run_all_seeds.sh
#
# Each dataset produces:
#   logs/<run_name>/eval_results.csv   (per-budget curve)
#   results_log.csv                    (one row per run, appended)

set -e

echo "=== Submitting Grid Stability (seeds 0-2) ==="
JOB_GS=$(sbatch --parsable run_afa_seeds.sh)
echo "  Job ID: ${JOB_GS}"

echo "=== Submitting CUBE-NM (seeds 0-2) ==="
JOB_CUBE=$(sbatch --parsable run_cube_nm_seeds.sh)
echo "  Job ID: ${JOB_CUBE}"

echo "=== Submitting CKD (seeds 0-2) ==="
JOB_CKD=$(sbatch --parsable run_ckd_seeds.sh)
echo "  Job ID: ${JOB_CKD}"

echo "=== Submitting Metabric (seeds 0-2) ==="
JOB_META=$(sbatch --parsable run_metabric_seeds.sh)
echo "  Job ID: ${JOB_META}"

echo "=== Submitting Bank Marketing (seeds 0-2) ==="
JOB_BANK=$(sbatch --parsable run_bank_marketing_seeds.sh)
echo "  Job ID: ${JOB_BANK}"

echo "=== Submitting MNIST (seeds 0-2) ==="
JOB_MNIST=$(sbatch --parsable run_mnist_seeds.sh)
echo "  Job ID: ${JOB_MNIST}"

echo "=== Submitting California Housing (seeds 0-2) ==="
JOB_CAL=$(sbatch --parsable run_california_housing_seeds.sh)
echo "  Job ID: ${JOB_CAL}"

echo "=== Submitting MiniBooNE (seeds 0-2) ==="
JOB_MINI=$(sbatch --parsable run_miniboone_seeds.sh)
echo "  Job ID: ${JOB_MINI}"

echo "=== Submitting Fashion MNIST (seeds 0-2) ==="
JOB_FMNIST=$(sbatch --parsable run_fashion_mnist_seeds.sh)
echo "  Job ID: ${JOB_FMNIST}"

echo ""
echo "All jobs submitted."
echo "  Grid Stability    : ${JOB_GS}   (array 0-2)"
echo "  CUBE-NM           : ${JOB_CUBE}  (array 0-2)"
echo "  CKD               : ${JOB_CKD}   (array 0-2)"
echo "  Metabric          : ${JOB_META}  (array 0-2)"
echo "  Bank Marketing    : ${JOB_BANK}  (array 0-2)"
echo "  MNIST             : ${JOB_MNIST}  (array 0-2)"
echo "  California Housing: ${JOB_CAL}  (array 0-2)"
echo "  MiniBooNE         : ${JOB_MINI}  (array 0-2)"
echo "  Fashion MNIST     : ${JOB_FMNIST}  (array 0-2)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will accumulate in: results_log.csv"
