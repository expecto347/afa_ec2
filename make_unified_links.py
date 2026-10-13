"""Unified run names for the thesis tables/figures.

The synthetic benchmarks and CUBE-NM use the efficient training configuration (S5:
--fast_knn --knn_subsample 10000 --batch_size 256 --lr 1.4e-3); the real-world datasets
keep the original configuration.  This script creates

  logs/ec2_<ds>_tp_seed<s>     two-phase framework         (S5: ec2_<ds>_s5_*,     else ec2_<ds>_seed*)
  logs/ec2_<ds>_ftv_seed<s>    + policy-adapted predictor  (S5: ec2_<ds>_s5ft_*,   else ec2_<ds>_ft_*)
  logs/ec2_<ds>_laftv_seed<s>  + lookahead anchor          (S5: ec2_<ds>_s5laft_*, else ec2_<ds>_laft_*)

so that analyze_significance.py can be run with EC2_VARIANT=tp / ftv / laftv / final.
"""
import os

S5 = {"syn1", "syn2", "syn3", "cube_nm"}
DS = ["syn1", "syn2", "syn3", "cube_nm", "grid_stability", "ckd", "metabric",
      "bank_marketing", "mnist", "miniboone", "california_housing"]
MAP = {"tp": ("s5", ""), "ftv": ("s5ft", "ft"), "laftv": ("s5laft", "laft")}

for ds in DS:
    for name, (s5_tag, old_tag) in MAP.items():
        tag = s5_tag if ds in S5 else old_tag
        for s in [0, 1, 2]:
            target = f"ec2_{ds}_{tag}_seed{s}" if tag else f"ec2_{ds}_seed{s}"
            link = f"logs/ec2_{ds}_{name}_seed{s}"
            if not os.path.exists(os.path.join("logs", target)):
                continue                      # e.g. no laft run for Bank Marketing
            if os.path.islink(link):
                os.remove(link)
            os.symlink(target, link)
    print(ds, {n: (MAP[n][0] if ds in S5 else MAP[n][1]) or "orig" for n in MAP})
