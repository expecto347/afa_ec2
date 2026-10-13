"""x11's kNN target and network output in states with one observed feature (mechanism check)."""
import torch
from diagnose_target_agreement import load, A, DEV
from ec2_core import knn_ec2_target
from ec2_sefa import prepare_data

names = [f"x{i+1}" for i in range(11)]
with torch.no_grad():
    for ds in ["syn1", "syn2", "syn3"]:
        _, _, _, _, Xte, _, _, _ = prepare_data(A(ds))
        X = Xte[:2000].to(DEV)
        m, sd = load(ds, 0)
        rows = []
        for j in range(11):
            mask = torch.zeros_like(X); mask[:, j] = 1
            T = knn_ec2_target(X, mask, m.X_train, m.y_train, m.feat_thresholds, k=50) + 1e6 * mask
            V = m.value_network(m.mask_layer(X, mask)) + 1e6 * mask
            rT = (T.argsort(1) == 10).float().argmax(1).float().mean().item() + 1
            rV = (V.argsort(1) == 10).float().argmax(1).float().mean().item() + 1 if j != 10 else float('nan')
            rows.append(f"{names[j]}: T_x11 {T[:,10].mean()*1e3 if j!=10 else float('nan'):6.2f} (rank {rT if j!=10 else float('nan'):4.1f}) "
                        f"T_x7 {T[:,6].mean()*1e3 if j!=6 else float('nan'):6.2f}  net x11 rank {rV:4.1f}")
        print(f"===== {ds}  (observed feature -> mean target x1e3, mean rank of x11 among unobserved)")
        print("\n".join("  " + r for r in rows))
