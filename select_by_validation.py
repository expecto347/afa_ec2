"""Choose the final Deep-EC2 configuration per dataset on the VALIDATION set.

Candidates (all with the policy-adapted predictor):  ft, laft (+lookahead anchor),
tft (+validation-tuned k, M), tlaft (both).  The score of a candidate is the mean over
seeds of the trajectory-mean validation metric of the final model, as printed by
finetune_predictor_on_policy ("[finetune] kept predictor with val ... = v").
Test results are not read.  Writes symlinks logs/ec2_<ds>_final_seed<s> -> chosen run.
"""
import glob
import os
import re

DS = ["syn1", "syn2", "syn3", "cube_nm", "grid_stability", "ckd", "metabric",
      "bank_marketing", "mnist", "miniboone", "california_housing"]
CANDS = ["ft", "laft", "tft", "tlaft"]
# Synthetic benchmarks and CUBE-NM use the efficient training configuration (S5).
CANDS_DS = {ds: ["s5ft", "s5laft"] for ds in ["syn1", "syn2", "syn3", "cube_nm"]}
PAT = re.compile(r"\[finetune\] kept predictor with val \w+ mean = ([0-9.]+)")


def val_score(ds, tag):
    vals = []
    for seed in [0, 1, 2]:
        logs = [f for f in glob.glob(f"logs/VAR_{ds}_{tag}_*_{seed}.out")]
        best = None
        for f in sorted(logs, key=os.path.getmtime):          # newest run of this tag/seed wins
            m = PAT.findall(open(f, errors="ignore").read())
            if m:
                best = float(m[-1])
        if best is None:
            return None
        vals.append(best)
    return sum(vals) / len(vals)


for ds in DS:
    scores = {t: val_score(ds, t) for t in CANDS_DS.get(ds, CANDS)}
    scores = {t: v for t, v in scores.items() if v is not None}
    pick = max(scores, key=scores.get)
    print(f"{ds:20s} " + "  ".join(f"{t}:{v:.4f}" for t, v in scores.items()) + f"   -> {pick}")
    for s in [0, 1, 2]:
        link = f"logs/ec2_{ds}_final_seed{s}"
        if os.path.islink(link):
            os.remove(link)
        os.symlink(f"ec2_{ds}_{pick}_seed{s}", link)
