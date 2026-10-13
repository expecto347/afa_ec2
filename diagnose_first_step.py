"""First-step choice (value net at the empty state) vs. step-0 target, for a given run tag."""
import glob, os, sys, torch, torch.nn as nn
for tag in sys.argv[1:]:
    for ds in ["syn1", "syn2", "syn3"]:
        out = []
        for seed in [0, 1, 2]:
            ck = max(glob.glob(f"logs/ec2_{ds}{tag}_seed{seed}/ec2_train/checkpoints/best_val_perf_model*.ckpt"), key=os.path.getmtime)
            sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
            d = sd["X_train"].shape[1]
            vn = nn.Sequential(nn.Linear(2*d, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, d))
            vn.load_state_dict({k[14:]: v for k, v in sd.items() if k.startswith("value_network.")}); vn.eval()
            with torch.no_grad(): v = vn(torch.zeros(1, 2*d))[0]
            g = sd["global_ec2_cache"]
            out.append(f"s{seed}: net x{int(v.argmin())+1} (x11 rank {v.argsort().tolist().index(10)+1}), target x{int(g.argmin())+1}")
        print(f"{ds}{tag or ' (orig)'}: " + " | ".join(out))
