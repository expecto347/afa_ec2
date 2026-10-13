"""Profile one Deep-EC2 training epoch on Syn2 and test a faster, exact kNN.

Times (CUDA-synchronised): kNN target, value-network forward/backward, data loading,
and a validation pass; compares the current kNN distance (materialised [B, N, d]
tensor) with a matmul formulation, and checks that both give the same neighbours
and targets.

Usage: apptainer exec --nv afa_env.sif python -u profile_training.py
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ec2_core import compute_feature_thresholds, knn_ec2_target
from ec2_sefa import prepare_data

dev = torch.device("cuda")


class A:
    dataset, syn_seed, sefa_data_dir = "syn2", 8139, ""


def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def knn_idx_current(x, mask, X, k):
    diff = (X[None, :, :] - x[:, None, :]) * mask[:, None, :]
    return torch.topk(diff.pow(2).sum(-1), k, largest=False, dim=1)[1]


def knn_idx_matmul(x, mask, X, X2, k):
    # sum_j m_j (x_j - X_j)^2 = m.X^2 - 2 (m*x).X + sum_j m_j x_j^2   (mask is binary)
    d = mask @ X2.T - 2.0 * (mask * x) @ X.T + (mask * x * x).sum(1, keepdim=True)
    return torch.topk(d, k, largest=False, dim=1)[1]


def main():
    Xtr, ytr, Xva, yva, *_ = prepare_data(A())
    X, y = Xtr.to(dev), ytr.to(dev)
    th = compute_feature_thresholds(X, 10)
    B, d, k = 128, X.shape[1], 50
    x = X[:B].clone()
    print(f"N={len(X)} d={d} B={B} k={k}")

    for nobs in [1, 3, 6]:
        mask = torch.zeros(B, d, device=dev)
        for i in range(B):
            mask[i, torch.randperm(d)[:nobs]] = 1
        torch.backends.cuda.matmul.allow_tf32 = True  # as in training (set_float32_matmul_precision("high"))
        X2 = X * X
        t_cur = timeit(lambda: knn_idx_current(x, mask, X, k))
        t_mm = timeit(lambda: knn_idx_matmul(x, mask, X, X2, k))
        t_full = timeit(lambda: knn_ec2_target(x, mask, X, y, th, k=k))
        t_fast = timeit(lambda: knn_ec2_target(x, mask, X, y, th, k=k, X_train_sq=X2))
        Tc = knn_ec2_target(x, mask, X, y, th, k=k); Tf = knn_ec2_target(x, mask, X, y, th, k=k, X_train_sq=X2)
        same_arg = ((Tc + 1e6 * mask).argmin(1) == (Tf + 1e6 * mask).argmin(1)).float().mean().item()
        print(f"      full target: current {t_full*1e3:.2f} ms, fast {t_fast*1e3:.2f} ms (x{t_full/t_fast:.1f}); "
              f"max |dT| {(Tc - Tf).abs().max().item():.2e}, same arg-min {same_arg:.3f}")
        # neighbour-set agreement (ties among equal distances may be broken differently)
        a = knn_idx_current(x, mask, X, k).sort(1)[0]
        b = knn_idx_matmul(x, mask, X, X2, k).sort(1)[0]
        same = (a == b).all(1).float().mean().item()
        # agreement of the k-th distance (true set equivalence modulo ties)
        dc = ((X[None] - x[:, None]) * mask[:, None]).pow(2).sum(-1)
        kth_c = dc.gather(1, a).max(1)[0]; kth_m = dc.gather(1, b).max(1)[0]
        print(f"|S|={nobs}: kNN distance+topk current {t_cur*1e3:6.2f} ms  matmul {t_mm*1e3:6.2f} ms "
              f"(x{t_cur/t_mm:4.1f})  | full knn_ec2_target {t_full*1e3:6.2f} ms | "
              f"identical neighbour sets {same:.3f}, identical k-th distance {(kth_c == kth_m).float().mean():.3f}")

    # one-epoch breakdown with the current code
    vn = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 512), nn.ReLU(),
                       nn.Dropout(0.3), nn.Linear(512, d)).to(dev)
    opt = torch.optim.Adam(vn.parameters(), 1e-3)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=B, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    t_load = t_knn = t_net = 0.0
    nb = 60
    it = iter(dl)
    for _ in range(nb):
        t0 = time.time(); xb, yb = next(it); xb = xb.to(dev, non_blocking=True); torch.cuda.synchronize(); t_load += time.time() - t0
        mask = torch.zeros_like(xb)
        opt.zero_grad()
        for step in range(d):
            t0 = time.time()
            out = vn(torch.cat([xb * mask, mask], 1))
            torch.cuda.synchronize(); t_net += time.time() - t0
            t0 = time.time()
            T = knn_ec2_target(xb, mask, X, y, th, k=k)
            torch.cuda.synchronize(); t_knn += time.time() - t0
            t0 = time.time()
            ((out - T) ** 2 * (1 - mask)).mean().backward()
            a = torch.randint(d, (B,), device=dev)
            mask = torch.max(mask, torch.nn.functional.one_hot(a, d).float())
            torch.cuda.synchronize(); t_net += time.time() - t0
        t0 = time.time(); opt.step(); torch.cuda.synchronize(); t_net += time.time() - t0
    nbat = len(X) // B
    print(f"per batch: data {t_load/nb*1e3:.1f} ms, kNN targets {t_knn/nb*1e3:.1f} ms, value net {t_net/nb*1e3:.1f} ms"
          f"  -> projected epoch {nbat*(t_load+t_knn+t_net)/nb:.1f} s ({nbat} batches)")


if __name__ == "__main__":
    main()
