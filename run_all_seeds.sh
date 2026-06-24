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

echo ""
echo "All jobs submitted."
echo "  Grid Stability : ${JOB_GS}   (array 0-2)"
echo "  CUBE-NM        : ${JOB_CUBE}  (array 0-2)"
echo "  CKD            : ${JOB_CKD}   (array 0-2)"
echo "  Metabric       : ${JOB_META}  (array 0-2)"
echo "  Bank Marketing : ${JOB_BANK}  (array 0-2)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will accumulate in: results_log.csv"
