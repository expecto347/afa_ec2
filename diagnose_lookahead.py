"""
Would an L-step lookahead EC2 target rank the gate first?

Expected remaining edge weight with lookahead depth L on the full training set
(quantile bins as in compute_feature_thresholds):
    R_0(cell) = P(cell)^2 * Gini(labels in cell)
    R_L(cell) = min_j sum_b P(b | cell) * R_{L-1}(cell & bin_j = b)
    Q_L(i)    = sum_b P(b) * R_{L-1}(bin_i = b)        (Q_1 = current step-0 target)
Lower is better.

Usage: apptainer exec afa_env.sif python -u diagnose_lookahead.py
"""
import sys

import numpy as np

from ec2_sefa import generate_syn_data


def bins_of(X, M):
    out = np.empty(X.shape, dtype=np.int64)
    for i in range(X.shape[1]):
        q = np.quantile(X[:, i], np.linspace(1 / M, 1 - 1 / M, M - 1))
        out[:, i] = (X[:, i][:, None] > q[None, :]).sum(1)
    return out


def make_R(B, y, N, M):
    d = B.shape[1]

    def gini(idx):
        if len(idx) == 0:
            return 0.0
        p = y[idx].mean()
        return 1 - p * p - (1 - p) * (1 - p)

    def R(idx, used, L):
        if L == 0 or len(idx) < 2:
            return (len(idx) / N) ** 2 * gini(idx)
        best = np.inf
        for j in range(d):
            if j in used:
                continue
            bj = B[idx, j]
            s = 0.0
            for b in range(M):
                sub = idx[bj == b]
                if len(sub):
                    s += len(sub) / len(idx) * R(sub, used | {j}, L - 1)
            best = min(best, s)
        return best

    def Q(i, L):
        s = 0.0
        all_idx = np.arange(N)
        for b in range(M):
            sub = all_idx[B[:, i] == b]
            s += len(sub) / N * R(sub, {i}, L - 1)
        return s

    return Q


def main():
    names = [f"x{i+1}" for i in range(11)]
    for ds, rule in [(a, int(a[-1])) for a in (sys.argv[1:] or ["syn1", "syn2", "syn3"])]:
        X, y, *_ = generate_syn_data(rule, 8139)
        y = y.astype(float)
        print(f"===== {ds}")
        for M, Ls in [(10, [1, 2]), (4, [1, 2, 3])]:
            B = bins_of(X, M)
            Q = make_R(B, y, len(X), M)
            for L in Ls:
                q = np.array([Q(i, L) for i in range(11)])
                r = np.argsort(q)
                print(f"  M={M:2d} L={L}: " + " ".join(f"{names[i]}:{q[i]*1e3:.3f}" for i in r[:5])
                      + f"   | gate rank {int(np.where(r == 10)[0][0]) + 1}", flush=True)


if __name__ == "__main__":
    main()
