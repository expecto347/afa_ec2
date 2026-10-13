"""逐 seed AUC + 配对 t 检验: EC2 vs baselines, 所有数据集。

数据源:
  EC2/Random: logs/ec2_<ds>_seed{0,1,2}/eval_results.csv
  Baselines:  scripts/plot_<ds>.py 中 <prefix>_acc/<prefix>_auroc 数组 [n_seeds, n_budgets],
              budget = 数组下标, >1.5 的值自动视为百分比并除以 100

AUC 归一化: 各方法在公共 budget 网格上做 trapezoid, 除以网格最大 budget。
配对 t 检验: 3 个 seed 逐对配对 (df=2), p 值用 t(2) 分布闭式解, 不依赖 scipy。

输出:
  logs/significance_report.txt          人类可读报告
  logs/significance_results.json        全部数字
  logs/latex_main_table_acc.tex         主表 (AUC acc)
  logs/latex_main_table_auroc.tex       主表 (AUC AUROC)
"""
import csv
import json
import math
import os

import numpy as np

SEEDS = [0, 1, 2]
DATASETS = ['syn1', 'syn2', 'syn3', 'cube_nm', 'grid_stability', 'ckd',
            'metabric', 'bank_marketing', 'mnist', 'miniboone',
            'california_housing', 'fashion_mnist']
BASELINES = ['rand', 'cae', 'eddi', 'ga', 'iter', 'gdfs', 'aaco', 'dime', 'ol']
LABEL = {'rand': 'Random', 'cae': 'CAE', 'eddi': 'EDDI', 'ga': 'Greedy-Abl',
         'iter': 'Iterative', 'gdfs': 'GDFS', 'aaco': 'AACO', 'dime': 'DIME', 'ol': 'OL (RL)'}
# 主表排序: EC2 在前, Random 其次, 其余按原顺序
TABLE_COLS = ['rand', 'cae', 'eddi', 'ga', 'iter', 'gdfs', 'aaco', 'dime', 'ol']
# 主表只要求这些 baseline 齐全; OL 只在部分数据集上运行, 缺失时填 '--'
REQUIRED_COLS = ['cae', 'eddi', 'ga', 'iter', 'gdfs', 'aaco', 'dime']
# 做逐 budget 配对检验的数据集 (非短视优势段)
PER_BUDGET_DS = {'syn1', 'syn2', 'syn3', 'cube_nm'}
PER_BUDGET_MAX = 8

ROOT = os.path.dirname(os.path.abspath(__file__))
# EC2_VARIANT=<tag> 读取 logs/ec2_<ds>_<tag>_seed<s>，输出文件名带 _<tag> 后缀
VARIANT = os.environ.get('EC2_VARIANT', '')
RUN_INFIX = f'_{VARIANT}' if VARIANT else ''
OUT_SUFFIX = f'_{VARIANT}' if VARIANT else ''


def ttest_rel_3(d):
    """3 对差值的配对 t 检验, 返回 (t, p)。t(2) CDF: F(x)=1/2+x/(2*sqrt(x^2+2))。"""
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = d.mean()
    s = d.std(ddof=1)
    if s == 0:
        if m == 0:
            return 0.0, 1.0
        return (math.inf if m > 0 else -math.inf), 0.0
    t = m / (s / math.sqrt(n))
    cdf = 0.5 + t / (2.0 * math.sqrt(t * t + 2.0))
    p = 2.0 * min(cdf, 1.0 - cdf)
    return t, min(p, 1.0)


def load_ec2(ds):
    """-> {metric: np.array [3, n_budgets]} 或 None"""
    out = {'acc': [], 'auroc': []}
    for s in SEEDS:
        f = os.path.join(ROOT, 'logs', f'ec2_{ds}{RUN_INFIX}_seed{s}', 'eval_results.csv')
        if not os.path.exists(f):
            return None
        acc, auroc = [], []
        for row in csv.DictReader(open(f)):
            acc.append(float(row['ec2_acc']))
            auroc.append(float(row['ec2_auroc']) if 'ec2_auroc' in row else None)
        out['acc'].append(acc)
        out['auroc'].append(auroc)
    res = {}
    for m in ['acc', 'auroc']:
        if any(v is None for row in out[m] for v in row):
            res[m] = None
        else:
            res[m] = np.array(out[m], dtype=float)
    return res


def load_baselines(ds):
    """exec scripts/plot_<ds>.py, 收集 <prefix>_{acc,auroc} -> {prefix: {metric: arr}}"""
    path = os.path.join(ROOT, 'scripts', f'plot_{ds}.py')
    if not os.path.exists(path):
        return {}
    code = open(path).read()
    cut = code.find('fig, ax = plt.subplots')
    if cut != -1:
        code = code[:cut]
    env = {}
    exec(code, env)
    out = {}
    for k, v in env.items():
        if not (k.endswith('_acc') or k.endswith('_auroc')):
            continue
        prefix, metric = k.rsplit('_', 1)
        a = np.array(v, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
        if np.nanmax(a) > 1.5:
            a = a / 100.0
        out.setdefault(prefix, {})[metric] = a
    return out


def auc_norm(curve, budgets):
    """归一化 AUC: trapezoid(curve, budgets) / budgets[-1]"""
    return np.trapezoid(curve, budgets) / budgets[-1]


def get_curves(ds):
    """公共 budget 网格上的全部方法曲线 -> (curves, budgets)。

    curves: {name: {metric: np.array [n_seeds, common]}}，name ∈ {ec2, rand} ∪ baselines
    """
    ec2 = load_ec2(ds)
    if ec2 is None:
        return None, None
    base = load_baselines(ds)
    n_ec2 = ec2['acc'].shape[1]
    common = min([n_ec2] + [d['acc'].shape[1] for d in base.values()])
    budgets = np.arange(common, dtype=float)

    curves = {'ec2': {m: ec2[m][:, :common].copy()
                      for m in ['acc', 'auroc'] if ec2[m] is not None}}
    for prefix, d in base.items():
        curves[prefix] = {m: d[m][:, :common] for m in ['acc', 'auroc'] if m in d}

    rand_acc, rand_auroc, has_auroc = [], [], True
    for s in SEEDS:
        f = os.path.join(ROOT, 'logs', f'ec2_{ds}{RUN_INFIX}_seed{s}', 'eval_results.csv')
        acc, auroc = [], []
        for row in csv.DictReader(open(f)):
            acc.append(float(row['rand_acc']))
            if 'rand_auroc' in row:
                auroc.append(float(row['rand_auroc']))
            else:
                has_auroc = False
        rand_acc.append(acc[:common])
        if has_auroc:
            rand_auroc.append(auroc[:common])
    curves['rand'] = {'acc': np.array(rand_acc, dtype=float)}
    if has_auroc and ec2['auroc'] is not None:
        curves['rand']['auroc'] = np.array(rand_auroc, dtype=float)
    return curves, budgets


def analyze(ds):
    curves, budgets = get_curves(ds)
    if curves is None:
        return None
    common = len(budgets)
    info = {'grid': {}, 'methods': {}, 'per_budget': {}, 'reach_full': None}
    info['grid'] = {'common': common}

    # 逐方法: AUC (逐 seed) + 与 EC2 的配对检验
    for name in ['ec2'] + BASELINES:
        if name not in curves:
            continue
        entry = {'n_seeds': curves[name]['acc'].shape[0], 'auc': {}, 't': {}, 'p': {}, 'diff': {}}
        for m in ['acc', 'auroc']:
            if m not in curves[name]:
                continue
            n = len(curves[name][m])
            aucs = [auc_norm(curves[name][m][i], budgets) for i in range(n)]
            entry['auc'][m] = {'mean': float(np.mean(aucs)),
                               'std': float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
                               'per_seed': [float(a) for a in aucs]}
            if name != 'ec2' and entry['n_seeds'] == 3 and curves['ec2'].get(m) is not None:
                e_aucs = [auc_norm(curves['ec2'][m][i], budgets) for i in range(3)]
                d = np.array(e_aucs) - np.array(aucs)
                t, p = ttest_rel_3(d)
                entry['t'][m], entry['p'][m] = t, p
                entry['diff'][m] = {'mean': float(d.mean()),
                                    'per_seed': [float(x) for x in d]}
        info['methods'][name] = entry

    # 逐 budget 检验 (只对非短视优势段数据集)
    if ds in PER_BUDGET_DS:
        for b in range(1, min(PER_BUDGET_MAX, common - 1) + 1):
            info['per_budget'][b] = {}
            for name in BASELINES:
                if name not in curves or 'acc' not in curves[name]:
                    continue
                if curves[name]['acc'].shape[0] != 3 or b >= curves[name]['acc'].shape[1]:
                    continue
                d = curves['ec2']['acc'][:, b] - curves[name]['acc'][:, b]
                t, p = ttest_rel_3(d)
                info['per_budget'][b][name] = {
                    'ec2': float(curves['ec2']['acc'][:, b].mean()),
                    'base': float(curves[name]['acc'][:, b].mean()),
                    'diff_mean': float(d.mean()), 't': t, 'p': p}

    # 追平 Random 满预算性能所需 budget (逐 seed, acc)
    if 'acc' in curves['rand'] and 'acc' in curves['ec2']:
        target = curves['rand']['acc'][:, common - 1]
        steps = []
        for i in range(3):
            hit = [b for b in range(common) if curves['ec2']['acc'][i, b] >= target[i] - 1e-9]
            if hit:
                steps.append(hit[0])
        if len(steps) == 3:
            info['reach_full'] = {'steps': steps, 'mean': float(np.mean(steps))}

    return info


def fmt_ms(e, m):
    if m not in e['auc']:
        return None
    return e['auc'][m]['mean'], e['auc'][m]['std']


def mark(p, diff, thr=(0.05, 0.01)):
    """EC2 显著更好: 上标 * / **;  显著更差: 下标 + / ++ (LaTeX 数学模式)"""
    if p is None:
        return ''
    if p < thr[1]:
        return '$^{**}$' if diff > 0 else '$_{++}$'
    if p < thr[0]:
        return '$^{*}$' if diff > 0 else '$_{+}$'
    return ''


def main():
    report, all_results = [], {}
    for ds in DATASETS:
        info = analyze(ds)
        if info is None:
            report.append(f'!! {ds}: 缺 EC2 结果, 跳过')
            continue
        all_results[ds] = info
        report.append('=' * 100)
        report.append(f'  {ds}   (budget 网格: {info["grid"]})')
        report.append('=' * 100)
        for m in ['acc', 'auroc']:
            rows = []
            for name in ['ec2'] + BASELINES:
                e = info['methods'].get(name)
                if not e or m not in e['auc']:
                    continue
                a = e['auc'][m]
                flag = '' if e['n_seeds'] == 3 else f'  [仅 {e["n_seeds"]} seed, 无检验]'
                if name == 'ec2':
                    rows.append(f'  {"EC2":<12} {a["mean"]:.3f}±{a["std"]:.3f}{flag}')
                else:
                    t, p = e['t'].get(m, (None, None)), e['p'].get(m)
                    d = e['diff'][m]['mean'] if m in e['diff'] else None
                    sig = ''
                    if p is not None:
                        sig = f'  t={e["t"][m]:+.2f} p={p:.3f}' + (' *' if p < .05 else ' **' if p < .01 else '')
                    d_str = f'{d:+.3f}' if d is not None else '  --  '
                    rows.append(f'  {LABEL[name]:<12} {a["mean"]:.3f}±{a["std"]:.3f}'
                                f'  diff={d_str}{sig}{flag}')
            report.append(f'  --- AUC({m}) ---')
            report.extend(rows)
        if info['reach_full']:
            r = info['reach_full']
            report.append(f'  EC2 acc 追平 Random@满budget 所需步数: {r["steps"]} (均值 {r["mean"]:.1f})')
        if info['per_budget']:
            report.append('  --- 逐 budget 配对检验 (acc) ---')
            for b, res in info['per_budget'].items():
                cells = []
                for name, r in res.items():
                    star = '*' if r['p'] < .05 else ('**' if r['p'] < .01 else '')
                    cells.append(f'{LABEL[name]} {r["diff_mean"]:+.3f} p={r["p"]:.3f}{star}')
                report.append(f'    b={b}: ' + ' | '.join(cells))
        report.append('')

    # LaTeX 主表: 只纳入 EC2 与全部 7 个 baseline 均有 3-seed 该指标数据的数据集
    for metric, title in [('acc', 'AUC(accuracy)'), ('auroc', 'AUC(AUROC)')]:
        lines = []
        for ds in DATASETS:
            info = all_results.get(ds)
            if info is None:
                continue
            ec2_has = 'ec2' in info['methods'] and metric in info['methods']['ec2']['auc']
            base_full = all(
                name in info['methods'] and metric in info['methods'][name]['auc']
                and info['methods'][name]['n_seeds'] == 3
                for name in REQUIRED_COLS)
            if not (ec2_has and base_full):
                continue
            cells = []
            vals = {}
            for name in ['ec2'] + TABLE_COLS:
                e = info['methods'].get(name)
                if not e or metric not in e['auc']:
                    cells.append('--')
                    continue
                vals[name] = e['auc'][metric]['mean']
            best = max(vals.values())
            row = []
            for name in ['ec2'] + TABLE_COLS:
                e = info['methods'].get(name)
                if not e or metric not in e['auc'] or e['n_seeds'] != 3:
                    row.append('--')
                    continue
                v, s = e['auc'][metric]['mean'], e['auc'][metric]['std']
                cell = f'{v:.3f}$\\pm${s:.3f}'
                if e['auc'][metric]['mean'] == best:
                    cell = f'\\textbf{{{cell}}}'
                if name != 'ec2':
                    p = e['p'].get(metric)
                    d = e['diff'][metric]['mean'] if metric in e['diff'] else 0
                    cell += mark(p, d)
                row.append(cell)
            dsname = {'cube_nm': 'CUBE-NM', 'syn1': 'Syn1', 'syn2': 'Syn2', 'syn3': 'Syn3',
                      'grid_stability': 'Grid Stability', 'ckd': 'CKD', 'metabric': 'Metabric',
                      'bank_marketing': 'Bank Marketing', 'mnist': 'MNIST',
                      'miniboone': 'MiniBooNE', 'california_housing': 'California Housing',
                      'fashion_mnist': 'Fashion-MNIST'}[ds]
            lines.append(f'{dsname} & ' + ' & '.join(row) + r' \\')
        out = os.path.join(ROOT, 'logs', f'latex_main_table_{metric}{OUT_SUFFIX}.tex')
        open(out, 'w').write('\n'.join(lines) + '\n')
        report.append(f'LaTeX 片段 ({title}): {out}')

    open(os.path.join(ROOT, 'logs', f'significance_report{OUT_SUFFIX}.txt'), 'w').write('\n'.join(report) + '\n')
    json.dump(all_results, open(os.path.join(ROOT, 'logs', f'significance_results{OUT_SUFFIX}.json'), 'w'),
              indent=1, ensure_ascii=False)
    print('\n'.join(report))


if __name__ == '__main__':
    main()
