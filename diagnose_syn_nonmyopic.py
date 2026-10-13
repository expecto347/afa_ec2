"""
Why does Syn2 not show the gate-first (non-myopic) pattern?

Part A - step-0 kNN EC2 target per feature (global path, train set):
    * label-Gini target (what the code regresses on)
    * decision-region-Gini target (EC2 theory: Gini over Bayes regions R_y)
    for several bin counts M.

Part B - oracle evaluation of acquisition orders with the exact Bayes posterior
    p(y=1 | x_S) = E_{x_U ~ N(0,I)} sigmoid(-logit(x))      (Monte Carlo)
    Bayes accuracy is the best any predictor can reach on a given subset, so the
    comparison isolates the acquisition order from predictor quality.
    Orders: learned policy (seed-0 checkpoint), gate-first oracle, x7-first oracle.

Usage (SLURM):  apptainer exec --nv afa_env.sif python -u diagnose_syn_nonmyopic.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from ec2_core import EC2Estimator, MaskLayer, Predictor
from ec2_sefa import generate_syn_data

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RULES = {"syn1": 1, "syn2": 2, "syn3": 3}
# 0-based feature indices; x11 = 10
BRANCH = {
    "syn1": ([0, 1], [2, 3, 4, 5]),
    "syn2": ([0, 1], [6, 9, 8, 7]),
    "syn3": ([2, 3, 4, 5], [6, 9, 8, 7]),
}


def logit_np(X, rule):
    s1 = 4 * X[..., 0] * X[..., 1]
    s2 = 1.2 * np.sum(X[..., 2:6] ** 2, axis=-1) - 4.2
    s3 = -10 * np.sin(0.2 * X[..., 6]) + np.abs(X[..., 7]) + X[..., 8] + np.exp(-X[..., 9]) - 2.4
    a, b = {1: (s1, s2), 2: (s1, s3), 3: (s2, s3)}[rule]
    g = (X[..., 10] < 0)
    return np.where(g, a, b)


def logit_t(X, rule):
    s1 = 4 * X[..., 0] * X[..., 1]
    s2 = 1.2 * (X[..., 2:6] ** 2).sum(-1) - 4.2
    s3 = -10 * torch.sin(0.2 * X[..., 6]) + X[..., 7].abs() + X[..., 8] + torch.exp(-X[..., 9]) - 2.4
    a, b = {1: (s1, s2), 2: (s1, s3), 3: (s2, s3)}[rule]
    return torch.where(X[..., 10] < 0, a, b)


# ---------------------------------------------------------------- Part A
def step0_target(X, lab, M):
    """sum_b p_b^3 Gini(lab | bin_b) per feature, quantile bins (as compute_feature_thresholds)."""
    out = []
    for i in range(X.shape[1]):
        q = np.quantile(X[:, i], np.linspace(1 / M, 1 - 1 / M, M - 1))
        b = (X[:, i][:, None] > q[None, :]).sum(1)
        s = 0.0
        for k in range(M):
            m = b == k
            if m.sum() == 0:
                continue
            pb = m.mean()
            p1 = lab[m].mean()
            s += pb ** 3 * (1 - p1 ** 2 - (1 - p1) ** 2)
        out.append(s)
    return np.array(out)


# ---------------------------------------------------------------- Part B
@torch.no_grad()
def bayes_post(X, mask, rule, K=1024, chunk=500):
    """p(y=1 | x_S) by MC over unobserved coords. X raw [N,d] tensor, mask [N,d] bool."""
    out = []
    for s in range(0, len(X), chunk):
        x, m = X[s:s + chunk], mask[s:s + chunk]
        z = torch.randn(len(x), K, x.shape[1], device=DEV)
        xs = torch.where(m[:, None, :], x[:, None, :], z)
        out.append(torch.sigmoid(-logit_t(xs, rule)).mean(1))
    return torch.cat(out)


def bayes_acc_curve(X, y, orders, rule):
    """orders: [N, d] long tensor of acquisition order. Returns Bayes acc at budgets 0..d."""
    N, d = X.shape
    mask = torch.zeros(N, d, dtype=torch.bool, device=DEV)
    accs = []
    for t in range(d + 1):
        if t > 0:
            mask[torch.arange(N, device=DEV), orders[:, t - 1]] = True
        p = bayes_post(X, mask, rule)
        accs.append(((p > 0.5).long() == y).float().mean().item())
    return accs


def fixed_orders(X, first, ds):
    b1, b2 = BRANCH[ds]
    noise = lambda rel: [i for i in range(11) if i not in rel]
    rows = []
    for x in X.cpu().numpy():
        rel = b1 if x[10] < 0 else b2
        if first == "gate":
            seq = [10] + rel
        else:  # first = feature index, then gate, then branch features
            seq = [first, 10] + [i for i in rel if i != first]
        seq += [i for i in noise(seq) if i not in seq]
        rows.append(seq)
    return torch.tensor(rows, device=DEV)


@torch.no_grad()
def learned_orders(ds, Xn, seed=0, tag=None):
    import glob as _g
    tag = os.environ.get("RUN_TAG", "") if tag is None else tag
    ckpt = max(_g.glob(f"logs/ec2_{ds}{tag}_seed{seed}/ec2_train/checkpoints/best_val_perf_model*.ckpt"),
               key=os.path.getmtime)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    d = sd["X_train"].shape[1]
    vn = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.3),
                       nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, d))
    model = EC2Estimator.load_from_checkpoint(
        ckpt, value_network=vn, predictor=Predictor(d, 2, 512), mask_layer=MaskLayer(d),
        X_train=sd["X_train"], y_train=sd["y_train"], feat_thresholds=sd["feat_thresholds"],
        feat_mean=sd["feat_mean"].numpy(), feat_std=sd["feat_std"].numpy(),
        lr=1e-3, max_features=d, eps=0.1, strict=False).to(DEV).eval()
    mask = torch.zeros_like(Xn)
    seq = []
    for _ in range(d):
        s = model.value_network(model.mask_layer(Xn, mask)) + 1e6 * mask
        a = s.argmin(1)
        seq.append(a)
        mask[torch.arange(len(Xn), device=DEV), a] = 1.0
    return torch.stack(seq, 1), sd["feat_mean"].numpy(), sd["feat_std"].numpy()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    res = {}
    for ds, rule in RULES.items():
        Xtr, ytr, _, _, Xte, yte = generate_syn_data(rule, 8139)
        region = (logit_np(Xtr.astype(np.float64), rule) < 0).astype(float)  # Bayes y* = 1 iff p(y=1)>.5
        A = {}
        for M in [2, 4, 10, 20]:
            A[f"label_M{M}"] = step0_target(Xtr, ytr.astype(float), M).tolist()
            A[f"region_M{M}"] = step0_target(Xtr, region, M).tolist()

        N = 3000
        X = torch.tensor(Xte[:N], device=DEV)
        y = torch.tensor(yte[:N], device=DEV)
        B = {}
        # the learned policy sees inputs standardised with the checkpoint's own statistics
        import glob as _g
        _ck = max(_g.glob(f"logs/ec2_{ds}{os.environ.get('RUN_TAG', '')}_seed0/ec2_train/checkpoints/best_val_perf_model*.ckpt"),
                  key=os.path.getmtime)
        sd = torch.load(_ck, map_location="cpu")["state_dict"]
        Xn = (Xte[:N] - sd["feat_mean"].numpy()) / (sd["feat_std"].numpy() + 1e-8)
        lo, _, _ = learned_orders(ds, torch.tensor(Xn.astype(np.float32), device=DEV))
        B["learned"] = bayes_acc_curve(X, y, lo, rule)
        B["gate_first"] = bayes_acc_curve(X, y, fixed_orders(X, "gate", ds), rule)
        B["x7_first"] = bayes_acc_curve(X, y, fixed_orders(X, 6, ds), rule)
        if ds == "syn1":
            B["x3_first"] = bayes_acc_curve(X, y, fixed_orders(X, 2, ds), rule)
        B["learned_first_step"] = np.bincount(lo[:, 0].cpu().numpy(), minlength=11).tolist()
        res[ds] = {"A": A, "B": B}

        print(f"\n===== {ds} =====")
        names = [f"x{i+1}" for i in range(11)]
        for k in ["label_M10", "region_M10", "label_M4", "region_M4", "label_M2", "region_M2"]:
            v = np.array(A[k]); r = np.argsort(v)
            print(f"  step0 {k:11s} rank: " + " ".join(f"{names[i]}:{v[i]*1e3:.3f}" for i in r[:5])
                  + f"   | x11 rank {int(np.where(r == 10)[0][0]) + 1}")
        for k in ["learned", "gate_first", "x7_first"] + (["x3_first"] if ds == "syn1" else []):
            c = B[k]
            print(f"  Bayes acc {k:11s}: AUC {np.trapezoid(c) / (len(c) - 1):.4f}  "
                  + " ".join(f"{a:.3f}" for a in c[:7]))
        print(f"  learned first-step counts: {B['learned_first_step']}")
    json.dump(res, open(f"logs/syn_nonmyopic_diag{os.environ.get('RUN_TAG', '')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
