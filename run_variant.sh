#!/bin/bash
#SBATCH -J AFA_VAR
#SBATCH --array=0-2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%A_%a.out

# Re-run one dataset (3 seeds) with extra training flags under a new run name,
# leaving the original ec2_<ds>_seed<s> runs untouched.
#
#   DATASET=<ds> VARIANT=<tag> EXTRA="<flags>" sbatch -p <part> -t <time> run_variant.sh
#
# Results: logs/ec2_<ds>_<tag>_seed<s>/eval_results.csv
# Partition / time limit are passed on the sbatch command line (see submit_variant.sh).

DATASET=${DATASET:?must be set}
VARIANT=${VARIANT:?must be set}
EXTRA=${EXTRA:-}
SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME="ec2_${DATASET}_${VARIANT}_seed${SEED}"
DATA=/data/users/haozhes

cd ~/AFA_EC2
RUN="apptainer exec --nv afa_env.sif python -u"

tabular() {  # $1 = data dir name, rest = dataset-specific flags
    local d=$1; shift
    $RUN ec2_tabular.py --train_csv $DATA/$d/train.csv --val_csv $DATA/$d/val.csv \
        --label_col label "$@" --seed $SEED --run_name "$RUN_NAME" $EXTRA
}

case "$DATASET" in
    syn1|syn2|syn3)
        $RUN ec2_sefa.py --dataset $DATASET --patience 20 \
            --seed $SEED --run_name "$RUN_NAME" $EXTRA ;;
    metabric)
        $RUN ec2_sefa.py --dataset metabric --patience 20 --k 15 --pretrain_epochs 100 \
            --dropout 0.0 --random_first_step --train_epochs 300 \
            --seed $SEED --run_name "$RUN_NAME" $EXTRA ;;
    cube_nm)
        $RUN ec2_cube_nm.py --k 50 --patience 20 --max_features 55 --batch_size 64 \
            --seed $SEED --run_name "$RUN_NAME" $EXTRA ;;
    grid_stability)
        $RUN ec2_grid_stability.py --k 50 --patience 20 \
            --seed $SEED --run_name "$RUN_NAME" $EXTRA ;;
    ckd)
        tabular ckd --k 20 --patience 20 --batch_size 64 ;;
    bank_marketing)
        tabular bank_marketing --max_features 10 --k 50 --knn_subsample 5000 \
            --patience 20 --batch_size 128 ;;
    mnist|miniboone)
        tabular $DATASET --k 50 --knn_subsample 10000 --patience 20 --batch_size 128 ;;
    california_housing)
        tabular california_housing --k 50 --patience 20 --batch_size 128 ;;
    *) echo "Unknown dataset $DATASET"; exit 1 ;;
esac

CKPT=$(ls -t logs/${RUN_NAME}/ec2_train/checkpoints/best_val_perf_model*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found for ${RUN_NAME}"
    exit 1
fi
echo "Evaluating checkpoint: $CKPT"

case "$DATASET" in
    syn1|syn2|syn3|metabric)
        $RUN evaluate_ec2_sefa.py --checkpoint "$CKPT" --dataset $DATASET --hidden 512 ;;
    cube_nm)
        $RUN evaluate_ec2_cube_nm.py --checkpoint "$CKPT" --seed $SEED \
            --output_plot "logs/${RUN_NAME}/eval_plot.png" ;;
    grid_stability)
        $RUN evaluate_ec2_grid_stability.py --checkpoint "$CKPT" ;;
    *)
        $RUN evaluate_ec2_tabular.py --checkpoint "$CKPT" \
            --test_csv $DATA/$DATASET/test.csv --label_col label ;;
esac
