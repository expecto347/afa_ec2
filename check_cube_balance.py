import torch
from afabench.common.datasets.datasets import AFAContextDataset

dataset = AFAContextDataset(
    n_samples=20000,
    seed=0,
    context_feature_std=0.0,
    informative_feature_std=0.0,
    non_informative_feature_mean=0.5,
    non_informative_feature_std=0.0,
    use_cheap_context_features=True,
    n_contexts=5
)

X, y = dataset.data, dataset.targets
print("Data shape:", X.shape)
print("Data min:", X.min().item())
print("Data max:", X.max().item())
print("Data mean:", X.mean().item())

# If the data is 0 and 1, check the balance
if len(torch.unique(X)) <= 10:
    for val in torch.unique(X):
        print(f"Value {val.item()}: {(X == val).float().mean().item():.4f}")
else:
    zero_ratio = (X < 1e-5).float().mean()
    one_ratio = (X > 1 - 1e-5).float().mean()
    print(f"Ratio of 0s: {zero_ratio:.4f}")
    print(f"Ratio of 1s: {one_ratio:.4f}")
