"""
Opportunistic Learning (OL, Kachuee et al., ICLR 2019) as an RL baseline for Deep-EC2.

Uses the PyTorch implementation in the SEFA repository (models/opportunistic_rl.py,
unmodified), but feeds it exactly the data Deep-EC2 sees: same generator / splits /
standardisation, loaded through the EC2 data code.

Modes
  --mode sweep  : train one SEFA config (1..9) on seed 0, record the validation
                  trajectory AUC (SEFA's model-selection criterion).
  --mode final  : train the best sweep config on --seed, evaluate greedy acquisition on
                  the test set for budgets 0..d, save the curve in the baseline pkl
                  format, and record the training wall-clock time.

Usage (SLURM):
  apptainer exec --nv afa_env.sif python -u ol_baseline.py --dataset syn2 --mode sweep --config 3
  apptainer exec --nv afa_env.sif python -u ol_baseline.py --dataset syn2 --mode final --seed 0
"""
import argparse
import json
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import TensorDataset

SEFA_ROOT = "/data/users/haozhes/SEFA"
OUT_ROOT = "/data/users/haozhes/ol_results"


def sefa_imports():
    # SEFA modules use package-relative imports from the repository root.
    sys.path.insert(0, SEFA_ROOT)
    from experiments.metrics_dict import metrics_dict
    from models.opportunistic_rl import OpportunisticRL
    return metrics_dict, OpportunisticRL


def load_data(ds, seed):
    """Return standardised (X_tr, y_tr, X_val, y_val, X_te, y_te) tensors, as Deep-EC2 sees them."""
    if ds.startswith("syn") or ds == "metabric":
        from ec2_sefa import prepare_data

        class A:
            dataset, syn_seed = ds, 8139
            sefa_data_dir = "/data/users/haozhes/SEFA/datasets/data"
        Xtr, ytr, Xva, yva, Xte, yte, _, _ = prepare_data(A())
        return Xtr, ytr, Xva, yva, Xte, yte
    if ds == "cube_nm":
        from afabench.common.datasets.datasets import AFAContextDataset
        from sklearn.preprocessing import StandardScaler
        from ec2_cube_nm import CUBE_NM_KWARGS, CubeNMDataset
        full = AFAContextDataset(n_samples=6250, seed=seed, **CUBE_NM_KWARGS)
        np.random.seed(seed)
        val_inds = np.sort(np.random.choice(len(full), size=int(len(full) * 0.2), replace=False))
        tr_inds = np.setdiff1d(np.arange(len(full)), val_inds)
        tr, va = full.create_subset(tr_inds), full.create_subset(val_inds)
        sc = StandardScaler().fit(np.stack([tr[i][0].numpy() for i in range(len(tr))]))
        m, s = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
        te = AFAContextDataset(n_samples=1000, seed=seed + 100, **CUBE_NM_KWARGS)
        d = [CubeNMDataset(z, mean=m, std=s) for z in (tr, va, te)]
        return d[0].features, d[0].labels, d[1].features, d[1].labels, d[2].features, d[2].labels
    raise ValueError(f"dataset {ds} not wired up yet")


def build(ds, cfg_num, Xtr, ytr, OpportunisticRL):
    with open(osp.join(SEFA_ROOT, "experiments/hyperparameters/configs/opportunistic", f"config{cfg_num}.json")) as f:
        config = json.load(f)
    config.update({"num_con_features": Xtr.shape[1], "num_cat_features": 0, "most_categories": 0,
                   "out_dim": int(ytr.max()) + 1, "max_dim": None})
    return OpportunisticRL(config), config


@torch.no_grad()
def evaluate_curve(model, X, y, device, d_out):
    """Greedy acquisition on the test set; accuracy / AUROC at every budget 0..d."""
    model.eval()
    X, y = X.to(device), y.cpu().numpy()
    m_acq = torch.zeros_like(X)
    acc, auroc = {}, {}
    for b in range(X.shape[1] + 1):
        if b > 0:
            m_acq = model.acquire(X, m_acq)
        p = model.predict(X, m_acq).cpu().numpy()
        acc[b] = float(accuracy_score(y, p.argmax(1)))
        try:
            auroc[b] = float(roc_auc_score(y, p[:, 1]) if d_out == 2 else
                             roc_auc_score(y, p, multi_class="ovr", average="macro"))
        except ValueError:
            auroc[b] = float("nan")
    return acc, auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["sweep", "final"], required=True)
    ap.add_argument("--config", type=int, default=None, help="SEFA OL config 1..9 (sweep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_episodes", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--metric", default="accuracy",
                    help="validation metric for model selection (accuracy for synthetic/CUBE, as Deep-EC2)")
    args = ap.parse_args()

    metrics_dict, OpportunisticRL = sefa_imports()
    metric_f = metrics_dict[args.metric]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    Xtr, ytr, Xva, yva, Xte, yte = load_data(args.dataset, seed)
    ones = lambda X: torch.ones_like(X)
    train = TensorDataset(Xtr.float(), ytr.long(), ones(Xtr))
    val = TensorDataset(Xva.float(), yva.long(), ones(Xva))

    os.makedirs(OUT_ROOT, exist_ok=True)
    if args.mode == "sweep":
        cfg = args.config
        run_dir = osp.join(OUT_ROOT, "sweep", args.dataset, f"config{cfg}")
    else:
        sweep_dir = osp.join(OUT_ROOT, "sweep", args.dataset)
        scores = {}
        for c in range(1, 10):
            f = osp.join(sweep_dir, f"config{c}", "result.json")
            if osp.exists(f):
                scores[c] = json.load(open(f))["val_auc"]
        cfg = max(scores, key=scores.get)
        print(f"Sweep val AUCs: {scores} -> best config {cfg}")
        run_dir = osp.join(OUT_ROOT, "final", args.dataset, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    np.random.seed(8406 * (seed + 1) + 383)
    torch.manual_seed(8406 * (seed + 1) + 383)
    model, config = build(args.dataset, cfg, Xtr, ytr, OpportunisticRL)
    if args.num_episodes:
        model.num_episodes = args.num_episodes
    model = model.to(device)

    t0 = time.time()
    model.fit(train, val, run_dir, metric_f)          # SEFA: trains, reloads best, saves val_auc.pt
    train_time = time.time() - t0
    val_auc = float(torch.load(osp.join(run_dir, "val_auc.pt"), weights_only=False))
    result = {"dataset": args.dataset, "mode": args.mode, "config": cfg, "seed": seed,
              "val_auc": val_auc, "train_time_s": train_time, "sefa_config": config}

    if args.mode == "final":
        acc, auroc = evaluate_curve(model, Xte.float(), yte.long(), device, int(ytr.max()) + 1)
        result.update({"test_acc_final": acc[max(acc)], "test_acc_auc": float(np.mean(
            [(acc[b] + acc[b + 1]) / 2 for b in range(max(acc))]))})
        pkl = osp.join(OUT_ROOT, f"{args.dataset}_ol_trial_0_seed{seed}.pkl")
        pickle.dump({"acc": acc, "auroc": auroc, "train_time_s": train_time, "config": cfg}, open(pkl, "wb"))
        print(f"Saved curve -> {pkl}")
    json.dump(result, open(osp.join(run_dir, "result.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "sefa_config"}, indent=1, default=str))


if __name__ == "__main__":
    main()
