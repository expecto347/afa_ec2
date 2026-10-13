"""Write the OL (RL baseline) curves into scripts/plot_<ds>.py as `ol_acc` / `ol_auroc`.

Reads /data/users/haozhes/ol_results/<ds>_ol_trial_0_seed{0,1,2}.pkl (written by ol_baseline.py)
and (re)places an `ol_acc = np.array([...])` / `ol_auroc = ...` block in the dataset's plot
script, before the plotting section, so analyze_significance.py / plot_results.py pick it up.
"""
import os
import pickle
import re
import sys

import numpy as np

OL_ROOT = "/data/users/haozhes/ol_results"
BEGIN, END = "# --- OL (RL baseline) begin ---", "# --- OL (RL baseline) end ---"


def fmt(a):
    return "np.array([\n" + ",\n".join("    [" + ", ".join(f"{v:.6g}" for v in r) + "]" for r in a) + "\n])"


def main(datasets):
    for ds in datasets:
        rows = {"acc": [], "auroc": []}
        for s in [0, 1, 2]:
            f = os.path.join(OL_ROOT, f"{ds}_ol_trial_0_seed{s}.pkl")
            if not os.path.exists(f):
                break
            d = pickle.load(open(f, "rb"))
            for m in rows:
                rows[m].append([d[m][b] for b in sorted(d[m])])
        if len(rows["acc"]) != 3:
            print(f"!! {ds}: OL results incomplete, skipped")
            continue
        block = f"{BEGIN}\nol_acc = {fmt(rows['acc'])}\nol_auroc = {fmt(rows['auroc'])}\n{END}\n"
        path = f"scripts/plot_{ds}.py"
        code = open(path).read()
        code = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", "", code, flags=re.S)
        cut = code.find("fig, ax = plt.subplots")
        code = code + "\n" + block if cut == -1 else code[:cut] + block + "\n" + code[cut:]
        open(path, "w").write(code)
        print(f"{ds}: OL acc-AUC per seed "
              f"{[round(float(np.trapezoid(r) / (len(r) - 1)), 3) for r in rows['acc']]} -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["syn1", "syn2", "syn3", "cube_nm"])
