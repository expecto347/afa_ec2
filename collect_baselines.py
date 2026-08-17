"""从三个 baseline 仓库的 pkl 结果生成 scripts/plot_<dataset>.py 数据文件。

每个 pkl 结构: {'acc': {budget: float}, 'auroc': {budget: float}, 'features': ...}
plot_results.py 读取 scripts/plot_<dataset>.py 中形如 <method>_acc / <method>_auroc 的
[n_seeds, n_budgets] 数组并逐 seed 展开画均值±CI。

Usage: apptainer exec afa_env.sif python collect_baselines.py
"""
import os
import pickle

import numpy as np

DATASETS = ['grid_stability', 'ckd', 'metabric', 'bank_marketing',
            'mnist', 'california_housing', 'miniboone']

# (变量名前缀, 对应仓库, 结果文件模板, seed 维度)
SOURCES = [
    ('cae',   'ds',   '/data/users/haozhes/dynamic-selection/results/{ds}_cae_trial_0_seed{s}.pkl'),
    ('eddi',  'ds',   '/data/users/haozhes/dynamic-selection/results/{ds}_eddi_trial_0_seed{s}.pkl'),
    ('ga',    'ds',   '/data/users/haozhes/dynamic-selection/results/{ds}_greedy_ablation_trial_0_seed{s}.pkl'),
    ('iter',  'ds',   '/data/users/haozhes/dynamic-selection/results/{ds}_iterative_trial_0_seed{s}.pkl'),
    ('gdfs',  'ds',   '/data/users/haozhes/dynamic-selection/results/{ds}_greedy_trial_0_seed{s}.pkl'),
    ('aaco',  'aaco', '/data/users/haozhes/aaco/results/{ds}_aaco_trial_0_seed{s}.pkl'),
    ('dime',  'dime', '/data/users/haozhes/DIME/experiments/{ds}/results/{ds}_dime_trial_0_seed{s}.pkl'),
]

SEEDS = [0, 1, 2]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')


def load_curves(ds):
    """返回 {method_prefix: {'acc': [3, n], 'auroc': [3, n]}}，缺失的 seed 记入 missing。"""
    curves = {}
    missing = {}
    for prefix, repo, tmpl in SOURCES:
        acc_rows, auroc_rows, n_budgets = [], [], None
        missing[prefix] = []
        for s in SEEDS:
            p = tmpl.format(ds=ds, s=s)
            if not os.path.exists(p):
                missing[prefix].append(s)
                continue
            with open(p, 'rb') as f:
                data = pickle.load(f)
            acc = [data['acc'][b] for b in sorted(data['acc'])]
            auroc = [data['auroc'][b] for b in sorted(data['auroc'])]
            if n_budgets is None:
                n_budgets = len(acc)
            acc_rows.append(acc)
            auroc_rows.append(auroc)
        if n_budgets is not None:
            curves[prefix] = {
                'acc': np.array(acc_rows, dtype=float),
                'auroc': np.array(auroc_rows, dtype=float),
            }
    return curves, missing


def fmt_array(a):
    """numpy 数组 → python 源码中的嵌套列表字符串。"""
    rows = []
    for row in a:
        rows.append('[' + ', '.join(f'{v:.6g}' for v in row) + ']')
    return '[\n    ' + ',\n    '.join(rows) + '\n]'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_missing = {}
    for ds in DATASETS:
        curves, missing = load_curves(ds)
        if not curves:
            print(f'!! {ds}: no results yet, skip')
            all_missing[ds] = missing
            continue

        lines = [f'"""', f'{ds}: ACC/AUROC vs. budget, 3-seed (rows = seed 0/1/2).',
                 '每个方法 shape [3, n_budgets], plot_results.py 会逐 seed 展开画均值±CI.',
                 '由 collect_baselines.py 从各仓库 pkl 自动生成。', '"""', '',
                 'import numpy as np', '']
        for prefix, curve in curves.items():
            lines.append(f"{prefix}_acc = np.array({fmt_array(curve['acc'])})")
            lines.append('')
            lines.append(f"{prefix}_auroc = np.array({fmt_array(curve['auroc'])})")
            lines.append('')

        out = os.path.join(OUT_DIR, f'plot_{ds}.py')
        with open(out, 'w') as f:
            f.write('\n'.join(lines))
        print(f'OK   {out}')
        for prefix, seeds_missing in missing.items():
            if seeds_missing:
                print(f'  missing {ds}/{prefix}: seeds {seeds_missing}')
        all_missing[ds] = missing

    print('\nSummary of missing results:')
    for ds, missing in all_missing.items():
        for prefix, seeds_missing in missing.items():
            if seeds_missing:
                print(f'  {ds}/{prefix}: seeds {seeds_missing}')


if __name__ == '__main__':
    main()
