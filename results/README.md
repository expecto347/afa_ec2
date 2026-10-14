# Deep-EC2: results and new mechanisms

Figures and tables reported in the thesis. All numbers are accuracy–budget AUC,
mean over three seeds (0, 1, 2), with seed-paired t-tests.

## New mechanisms

### 1. Phase 3 — policy-adapted predictor (`--finetune_predictor`)

The original pipeline pretrains the predictor on random masks, freezes it, and trains
the value network. Phase 3 then rolls out the trained greedy policy on the training
set and fine-tunes the predictor on the masks it acquires, mixed with ~25% random
masks (Adam 1e-4, ≤50 epochs, early stopping on the validation trajectory-mean
accuracy). The value network and the acquisition order are unchanged.
Implementation: `finetune_predictor_on_policy` in `ec2_core.py`; the fine-tuned
checkpoint is saved as `best_val_perf_model-ft.ckpt`.

### 2. Lookahead anchor for the first acquisition (`--lookahead_init L`, `--lookahead_bins M`)

With continuous, identically distributed features and equal-mass quantile bins,
every bin has p_b = 1/M, so the version-space factor of the one-step target
sum_b p_b^3 Gini_b is the same for every feature. The target then reduces to a
myopic per-bin Gini, and a gate that is uninformative on its own (Syn x11) is never
ranked first.

The anchor computes, once and before training, an exact L-step EC2 lookahead on
the training set (M quantile bins per feature):

    R_0(C) = P(C)^2 Gini(C)
    R_l(C) = min_j sum_b P(b | C) R_{l-1}(C ∩ {bin_j = b})
    Q_L(i) = sum_b P(bin_i = b) R_{L-1}({bin_i = b})

The resulting ranking replaces the value network's output at the empty state only.
Every later acquisition is chosen greedily by the value network. Implementation:
`lookahead_init_scores` and `_anchor_value_network` in `ec2_core.py`. It takes
seconds to about a minute on CPU (Syn: d = 11, L = 3, M = 4).

### 3. Exact fast kNN (`--fast_knn`)

Masked distances are computed as one matrix product. A shortlist of 2k candidates
is then re-ranked exactly, so the targets are identical to the original search and
about 7x faster. This is an implementation change only.

### Other additions

- `ol_baseline.py`, `run_ol.sh`, `collect_ol.py`: the Opportunistic Learning (RL)
  baseline (SEFA's PyTorch implementation; 9-config validation sweep, then 3 seeds).
- `analyze_significance.py`, `gen_*_tex.py`, `plot_diff.py`: tables and significance tests.
- `select_by_validation.py`, `make_unified_links.py`: select variants on validation
  and link the reported runs.
- `diagnose_*.py`: analyses of the one-step target, the gate value, and the Bayes-oracle
  evaluation used in the thesis.
- `run_variant.sh` / `submit_variant.sh`: run one flag set on all datasets × 3 seeds
  via SLURM.

## Reported configuration

| Datasets | Flags |
|---|---|
| Syn1/2/3 | `--finetune_predictor --lookahead_init 3 --fast_knn --knn_subsample 10000 --batch_size 256 --lr 1.4e-3` |
| CUBE-NM | `--finetune_predictor --lookahead_init 2 --fast_knn --batch_size 256 --lr 1.4e-3` |
| Real-world tabular (except Bank) | `--finetune_predictor --lookahead_init 3` |
| Bank Marketing | `--finetune_predictor` |

```bash
bash submit_variant.sh <tag> "<flags>"   # results in logs/ec2_<ds>_<tag>_seed<s>/
```

## Results

### Ablation (`tables/latex_ablation.tex`)

| Dataset | two-phase | + adapted predictor | + lookahead anchor (full) | OL (RL) |
|---|---|---|---|---|
| Syn1 | 0.713 | 0.717 | 0.728 | 0.731 (n.s.) |
| Syn2 | 0.730 | 0.735 | **0.740** | 0.739 (n.s.) |
| Syn3 | 0.746 | 0.749 | 0.754 | 0.756 (n.s.) |
| CUBE-NM | 0.926 | 0.927 | **0.927** | 0.921 (p = 0.017) |
| Grid Stability | 0.820 | 0.847 | 0.849 | – |
| CKD | 0.964 | 0.970 | 0.977 | – |
| Metabric | 0.618 | 0.625 | 0.621 | – |
| Bank Marketing | 0.895 | 0.908 | 0.908 | – |
| MNIST | 0.640 | 0.656 | 0.657 | – |
| MiniBooNE | 0.867 | 0.870 | 0.870 | – |
| California Housing | 0.558 | 0.587 | 0.586 | – |

With the anchor, the gate x11 is acquired first on 100% of test instances on
Syn1/2/3 for all seeds (`figures/syn*_feature_order.png`; without the anchor:
`figures/syn2_twophase_feature_order.png`).

### Training cost vs OL

| | Deep-EC2 | OL |
|---|---|---|
| Training time per model, synthetic | 19 min | 30 min |
| Training time per model, CUBE-NM | 8 min | 150 min |
| Total incl. OL's 9-config sweep, synthetic | 1.0 GPU-h | 5.9 GPU-h |
| Total incl. OL's 9-config sweep, CUBE-NM | 0.4 GPU-h | 34 GPU-h |

## Files

- `figures/*_acc_curve.png`, `*_auroc_curve.png`: accuracy / AUROC against budget for all methods.
- `figures/ec2_vs_baselines_acc_diff.png`: per-budget difference to each baseline.
- `figures/syn{1,2,3}_feature_order.png`, `cube_nm_eval_feature_order.png`: acquisition order of the full method.
- `tables/latex_main_table_{acc,auroc}.tex`, `latex_perbudget.tex`, `latex_ablation.tex`: thesis tables.
- `tables/significance_results_{main,noanchor,twophase}.json`: per-method AUCs and paired tests.

## Without the lookahead anchor (`figures_noanchor/`)

Same figures for the variant with Phases 1–3 but no anchor (`noanchor` links: S5 `s5ft`
runs on Syn/CUBE-NM, `ft` runs on the real-world data), used in the defence slides.
Regenerate with `EC2_VARIANT=noanchor python plot_results.py`,
`EC2_VARIANT=noanchor python plot_diff.py` and `sbatch run_analyze_syn_order_noanchor.sh`.
Gate-first rate without the anchor: 1/3, 2/3, 3/3 seeds on Syn1/2/3.
Training time (S5, SLURM wall-clock incl. evaluation): Syn 15–20 min (mean 17.5), CUBE-NM 7–9 min (mean 8.0).
