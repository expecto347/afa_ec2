"""
Does the learned value network reproduce the ranking of its kNN EC2 target?

For each checkpoint: compare value_network(zero mask) with the cached global
step-0 target (global_ec2_cache); then, on test instances, compare the policy's
argmin with the argmin of the kNN target at steps 1..3 along the policy's own
trajectory.  Also reports, on training states, the scale of the regression error
at step 0 versus the gaps between the step-0 targets.

Usage: apptainer exec --nv afa_env.sif python -u diagnose_target_agreement.py
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from ec2_core import EC2Estimator, MaskLayer, Predictor, knn_ec2_target
from ec2_sefa import prepare_data

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A:  # minimal args for prepare_data
    def __init__(self, ds):
        self.dataset, self.syn_seed, self.sefa_data_dir = ds, 8139, ""


TAG = os.environ.get("RUN_TAG", "")   # e.g. "_tp" -> logs/ec2_<ds>_tp_seed<s>


def load(ds, seed, tag=None):
    tag = TAG if tag is None else tag
    ckpt = max(glob.glob(f"logs/ec2_{ds}{tag}_seed{seed}/ec2_train/checkpoints/best_val_perf_model*.ckpt"),
               key=os.path.getmtime)
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    d = sd["X_train"].shape[1]
    vn = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.3),
                       nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, d))
    m = EC2Estimator.load_from_checkpoint(
        ckpt, value_network=vn, predictor=Predictor(d, 2, 512), mask_layer=MaskLayer(d),
        X_train=sd["X_train"], y_train=sd["y_train"], feat_thresholds=sd["feat_thresholds"],
        feat_mean=sd["feat_mean"].numpy(), feat_std=sd["feat_std"].numpy(),
        lr=1e-3, max_features=d, eps=0.1, strict=False).to(DEV).eval()
    return m, sd


@torch.no_grad()
def main():
    names = [f"x{i+1}" for i in range(11)]
    for ds in ["syn1", "syn2", "syn3"]:
        _, _, _, _, Xte, _, _, _ = prepare_data(A(ds))
        X = Xte[:2000].to(DEV)
        for seed in [0, 1, 2]:
            m, sd = load(ds, seed)
            g = sd["global_ec2_cache"].to(DEV)
            z = torch.zeros(1, 11, device=DEV)
            v0 = m.value_network(m.mask_layer(z, z))[0]
            rg, rv = g.argsort().tolist(), v0.argsort().tolist()
            print(f"{ds} s{seed}  step0 target top3 " + " ".join(f"{names[i]}:{g[i]*1e3:.3f}" for i in rg[:3])
                  + f" (x11 {g[10]*1e3:.3f}, rank {rg.index(10)+1})"
                  + " | net top3 " + " ".join(f"{names[i]}:{v0[i]*1e3:.3f}" for i in rv[:3])
                  + f" (x11 {v0[10]*1e3:.3f})  | max|net-target| {((v0-g).abs().max())*1e3:.3f}")
            # agreement along the policy's own trajectory
            mask = torch.zeros_like(X)
            for step in range(1, 4):
                v = m.value_network(m.mask_layer(X, mask)) + 1e6 * mask
                a = v.argmin(1)
                if step > 1:
                    T = knn_ec2_target(X, mask, m.X_train, m.y_train, m.feat_thresholds, k=50,
                                       global_ec2_cache=g) + 1e6 * mask
                    agree = (T.argmin(1) == a).float().mean().item()
                    # how often the network's choice is within the target's top-2
                    top2 = (T.argsort(1)[:, :2] == a[:, None]).any(1).float().mean().item()
                    print(f"      step {step}: policy==target argmin {agree:.3f}, in target top-2 {top2:.3f}")
                mask[torch.arange(len(X), device=DEV), a] = 1.0


if __name__ == "__main__":
    main()
