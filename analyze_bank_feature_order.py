"""
Analyze which features EC2 selects at step 1 and step 2 on the bank marketing test set.

Step 1: mask=0 for all samples → value network sees same input → same feature chosen for all.
Step 2: mask has step-1 feature revealed → varies per sample.
"""

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from ec2_tabular import TabularDataset, _parse_label_col
from ec2_core import Predictor, EC2Estimator, MaskLayer

CKPT = "logs/ec2_bank_marketing_trial_0/ec2_train/checkpoints/best_val_perf_model-v3.ckpt"
TEST_CSV = "/data/users/haozhes/bank_marketing/test.csv"
LABEL_COL = "label"
HIDDEN = 512
DROPOUT = 0.3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

temp = torch.load(CKPT, map_location="cpu")
feat_mean       = temp["state_dict"]["feat_mean"].numpy()
feat_std        = temp["state_dict"]["feat_std"].numpy()
X_train         = temp["state_dict"]["X_train"]
y_train         = temp["state_dict"]["y_train"]
feat_thresholds = temp["state_dict"]["feat_thresholds"]

d_in  = X_train.shape[1]
d_out = int(y_train.max().item()) + 1

# Read column names (feature names, excluding label)
import pandas as pd
cols = list(pd.read_csv(TEST_CSV, nrows=0).columns)
feat_names = [c for c in cols if c != LABEL_COL]
assert len(feat_names) == d_in, f"Expected {d_in} features, got {len(feat_names)}"

test_dataset = TabularDataset(TEST_CSV, label_col=LABEL_COL, mean=feat_mean, std=feat_std)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

predictor     = Predictor(d_in, d_out, HIDDEN, DROPOUT)
mask_layer    = MaskLayer(mask_size=d_in, append=True)
value_network = nn.Sequential(
    nn.Linear(d_in * 2, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
    nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(), nn.Dropout(DROPOUT),
    nn.Linear(HIDDEN, d_in),
)

model = EC2Estimator.load_from_checkpoint(
    CKPT,
    value_network=value_network,
    predictor=predictor,
    mask_layer=mask_layer,
    X_train=X_train, y_train=y_train,
    feat_thresholds=feat_thresholds,
    feat_mean=feat_mean, feat_std=feat_std,
    lr=1e-3, max_features=d_in, eps=0.1,
    strict=False,
).to(device)
model.eval()

step1_choices = []
step2_choices = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        B = len(x)

        # Step 1: all-zero mask
        mask1 = torch.zeros(B, d_in, dtype=x.dtype, device=device)
        scores1 = model.value_network(model.mask_layer(x, mask1))
        scores1 += 1e6 * mask1
        best1 = torch.argmin(scores1, dim=1)  # (B,)
        step1_choices.extend(best1.cpu().tolist())

        # Step 2: reveal step-1 feature
        mask2 = torch.zeros(B, d_in, dtype=x.dtype, device=device)
        mask2.scatter_(1, best1.unsqueeze(1), 1.0)
        scores2 = model.value_network(model.mask_layer(x, mask2))
        scores2 += 1e6 * mask2
        best2 = torch.argmin(scores2, dim=1)  # (B,)
        step2_choices.extend(best2.cpu().tolist())

print("\n=== Step 1 (most common first acquisition) ===")
c1 = Counter(step1_choices)
for idx, cnt in c1.most_common(5):
    print(f"  feat[{idx:2d}] {feat_names[idx]:<30s} : {cnt:5d} / {len(step1_choices)} samples ({100*cnt/len(step1_choices):.1f}%)")

print("\n=== Step 2 (most common second acquisition) ===")
c2 = Counter(step2_choices)
for idx, cnt in c2.most_common(10):
    print(f"  feat[{idx:2d}] {feat_names[idx]:<30s} : {cnt:5d} / {len(step2_choices)} samples ({100*cnt/len(step2_choices):.1f}%)")
