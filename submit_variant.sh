#!/bin/bash
# Submit one training variant on all thesis datasets (3 seeds each) via run_variant.sh.
#
#   bash submit_variant.sh <tag> "<extra flags>" [sbatch dependency, e.g. afterok:123]
#
# Example:
#   bash submit_variant.sh pbest "--restore_best_predictor"

TAG=${1:?variant tag}
EXTRA=${2:-}
DEP=${3:+--dependency=$3}

cd ~/AFA_EC2

# Optional: pin all jobs to one node with NODE=<name>.
NODE=${NODE-}
NODEFLAG=${NODE:+-w $NODE}

# dataset : partition : time limit (≤30 min → short, else long)
for spec in \
    syn1:long:5:00:00 syn2:long:5:00:00 syn3:long:5:00:00 \
    cube_nm:long:5:00:00 grid_stability:long:5:00:00 \
    metabric:short:0:30:00 ckd:short:0:30:00 \
    bank_marketing:long:5:00:00 mnist:long:5:00:00 \
    miniboone:long:5:00:00 california_housing:long:5:00:00
do
    IFS=: read -r DS PART T <<< "$spec"
    if [ -n "$SKIP" ] && [[ " $SKIP " == *" $DS "* ]]; then continue; fi
    JID=$(DATASET=$DS VARIANT=$TAG EXTRA="$EXTRA" \
          sbatch --parsable -J "VAR_${DS}_${TAG}" -p "$PART" -t "$T" -c 2 --mem=16G $NODEFLAG $DEP run_variant.sh)
    echo "$DS ($TAG): $JID"
done
