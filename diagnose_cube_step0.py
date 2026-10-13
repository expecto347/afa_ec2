"""CUBE-NM: step-0 kNN target (global cache) vs value-network output at the empty state."""
import torch, torch.nn as nn
for seed in [0, 1, 2]:
    import glob, os
    ck = max(glob.glob(f"logs/ec2_cube_nm_seed{seed}/ec2_train/checkpoints/best_val_perf_model*.ckpt"), key=os.path.getmtime)
    sd = torch.load(ck,
                    map_location="cpu", weights_only=False)["state_dict"]
    d = sd["X_train"].shape[1]
    vn = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Dropout(0.3),
                       nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, d))
    vn.load_state_dict({k[len("value_network."):]: v for k, v in sd.items() if k.startswith("value_network.")})
    vn.eval()
    g = sd["global_ec2_cache"]
    with torch.no_grad():
        v = vn(torch.zeros(1, 2 * d))[0]
    tg, tv = g.argsort()[:6].tolist(), v.argsort()[:6].tolist()
    print(f"seed{seed}: target top6 {tg} vals {[round(float(g[i])*1e4,3) for i in tg]}")
    print(f"        net    top6 {tv} vals {[round(float(v[i])*1e4,3) for i in tv]}  net argmin rank in target: {g.argsort().tolist().index(tv[0])+1}")
