"""汇总 syn1/2/3 × EC2 vs baselines (DIME/DS/AACO) 对比。

EC2:  logs/ec2_syn{N}_seed{s}/eval_results.csv  (budget,ec2_acc,ec2_auroc,...)
Baselines: 各仓库 results/*_trial_0_seed{s}.pkl  {'acc': {budget: v}, 'auroc': {...}}

输出每个数据集:
1. 各方法在 budget 3/5/8/11 的 acc / auroc (3-seed 均值±std)
2. 曲线下面积 (AUC of curve, 用 trapezoid) 排名
3. EC2 相对每个 baseline 的差值
"""
import csv
import glob
import os
import pickle

import numpy as np

DATASETS = ['syn1', 'syn2', 'syn3']
SEEDS = [0, 1, 2]
BUDGETS = [3, 5, 8, 11]

# method -> (pkl 目录, pkl 文件名模板)
METHODS = {
    'ec2':      (None, None),  # 特殊处理: eval_results.csv
    'dime':     ('/data/users/haozhes/DIME/experiments/{ds}/results', '{ds}_dime_trial_0_seed{s}.pkl'),
    'gdfs':     ('/data/users/haozhes/dynamic-selection/results', '{ds}_greedy_trial_0_seed{s}.pkl'),
    'cae':      ('/data/users/haozhes/dynamic-selection/results', '{ds}_cae_trial_0_seed{s}.pkl'),
    'eddi':     ('/data/users/haozhes/dynamic-selection/results', '{ds}_eddi_trial_0_seed{s}.pkl'),
    'iter':     ('/data/users/haozhes/dynamic-selection/results', '{ds}_iterative_trial_0_seed{s}.pkl'),
    'ga':       ('/data/users/haozhes/dynamic-selection/results', '{ds}_greedy_ablation_trial_0_seed{s}.pkl'),
    'aaco':     ('/data/users/haozhes/aaco/results', '{ds}_aaco_trial_0_seed{s}.pkl'),
}

METHOD_LABEL = {
    'ec2': 'EC2', 'dime': 'DIME', 'gdfs': 'GDFS', 'cae': 'CAE',
    'eddi': 'EDDI', 'iter': 'Iterative', 'ga': 'Greedy-Abl', 'aaco': 'AACO',
}


def label(m):
    if m.startswith('ec2_hps'):
        return f"EC2-{m.split('_')[1].upper()}"
    return METHOD_LABEL[m]


def load_ec2(ds, seed):
    f = f'/data/users/haozhes/AFA_EC2/logs/ec2_{ds}_seed{seed}/eval_results.csv'
    if not os.path.exists(f):
        return None
    acc, auroc = {}, {}
    for row in csv.DictReader(open(f)):
        b = int(row['budget'])
        acc[b] = float(row['ec2_acc'])
        auroc[b] = float(row['ec2_auroc'])
    return {'acc': acc, 'auroc': auroc}


def load_pkl(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def load_hps(ds):
    """glob logs/ec2_{ds}_hps*_seed*/eval_results.csv(自动发现所有 seed),按 hps 名聚合"""
    res = {}
    for f in glob.glob(f'/data/users/haozhes/AFA_EC2/logs/ec2_{ds}_hps*_seed*/eval_results.csv'):
        base = os.path.basename(os.path.dirname(f))
        hps = base.replace(f'ec2_{ds}_', '').rsplit('_seed', 1)[0]
        acc, auroc = {}, {}
        for row in csv.DictReader(open(f)):
            b = int(row['budget'])
            acc[b] = float(row['ec2_acc'])
            auroc[b] = float(row['ec2_auroc'])
        res.setdefault(hps, {'acc': {}, 'auroc': {}})
        for m, d in [('acc', acc), ('auroc', auroc)]:
            for b, v in d.items():
                res[hps][m].setdefault(b, []).append(float(v))
    return res


def load_method(ds, method):
    """-> {'acc': {budget: [seed 值]}, 'auroc': ...}"""
    if method == 'ec2':
        raw = [load_ec2(ds, s) for s in SEEDS]
    else:
        rdir_t, tpl = METHODS[method]
        rdir = rdir_t.format(ds=ds)
        raw = []
        for s in SEEDS:
            f = os.path.join(rdir, tpl.format(ds=ds, s=s))
            if os.path.exists(f):
                raw.append(load_pkl(f))
    raw = [r for r in raw if r]
    if not raw:
        return None
    res = {m: {} for m in ['acc', 'auroc']}
    for r in raw:
        for m in ['acc', 'auroc']:
            for b, v in r[m].items():
                res[m].setdefault(b, []).append(float(v))
    return res


def curve_auc(series):
    """曲线下面积 (0..max_budget, 归一化到 0-1)"""
    budgets = sorted(series)
    vals = [np.mean(series[b]) for b in budgets]
    area = np.trapezoid(vals, budgets)
    return area / budgets[-1]


def fmt(vals, fmtstr='{:.4f}'):
    if len(vals) < 3:
        return '  —   '
    return f"{fmtstr.format(np.mean(vals))}±{fmtstr.format(np.std(vals))}"


def main():
    for ds in DATASETS:
        print('=' * 88)
        print(f'  {ds}')
        print('=' * 88)
        data = {m: load_method(ds, m) for m in METHODS}
        data = {m: d for m, d in data.items() if d}

        # syn2 的 HPS 配置并入排名 (EC2-hpsN)
        for hps, d in load_hps(ds).items():
            data[f'ec2_{hps}'] = d

        # 曲线 AUC 排名
        print(f'\n  {"方法":<14} {"acc-AUC":>10} {"auroc-AUC":>12}   acc/auroc @ budgets')
        ranking = sorted(
            data.items(),
            key=lambda kv: -curve_auc(kv[1]['auroc']),
        )
        for m, d in ranking:
            auc_a = curve_auc(d['acc'])
            auc_r = curve_auc(d['auroc'])
            cells = []
            for b in BUDGETS:
                cells.append(
                    f"b{b}:{fmt(d['acc'][b], '{:.3f}')}/{fmt(d['auroc'][b], '{:.3f}')}"
                )
            print(f"  {label(m):<14} {auc_a:>8.3f}  {auc_r:>10.3f}   {'  '.join(cells)}")

        # EC2 vs 每个 baseline 的差值 (均值)
        print(f'\n  EC2 − baseline 差值 (均值, 正=EC2 好):')
        ec2 = data['ec2']
        print(f'  {"baseline":<14} {"acc-AUC":>9} {"auroc-AUC":>11}', end='')
        for b in BUDGETS:
            print(f'  {"acc@":>5}{b}', end='')
        print()
        for m in [k for k in data if k != 'ec2']:
            d = data[m]
            da = curve_auc(ec2['acc']) - curve_auc(d['acc'])
            dr = curve_auc(ec2['auroc']) - curve_auc(d['auroc'])
            print(f'  {label(m):<14} {da:>+9.3f} {dr:>+11.3f}', end='')
            for b in BUDGETS:
                diff = np.mean(ec2['acc'][b]) - np.mean(d['acc'][b])
                print(f'  {diff:>+8.3f}', end='')
            print()
        print()


if __name__ == '__main__':
    main()
