"""Ablation table for the thesis: contribution of each Deep-EC2 component.

Columns (accuracy--AUC, mean +- sd over three seeds):
  two-phase (no adaptation, no anchor) | + policy-adapted predictor | full Deep-EC2 (+ anchor)
  | full + validation-tuned (k, M) (tabular datasets where validation departs from the default)
  | best baseline | #baselines significantly better / worse than the full method.
Reads logs/significance_results_{twophase,noanchor,main,tlaft}.json.
Output: logs/latex_ablation.tex
"""
import json
import os

DS = [('syn1', 'Syn1'), ('syn2', 'Syn2'), ('syn3', 'Syn3'), ('cube_nm', 'CUBE-NM'),
      ('grid_stability', 'Grid Stability'), ('ckd', 'CKD'), ('metabric', 'Metabric'),
      ('bank_marketing', 'Bank Marketing'), ('mnist', 'MNIST'), ('miniboone', 'MiniBooNE'),
      ('california_housing', 'California Housing')]
BASE = ['cae', 'eddi', 'ga', 'iter', 'gdfs', 'aaco', 'dime', 'ol']
R = {k: json.load(open(f'logs/significance_results_{k}.json')) for k in ['twophase', 'noanchor', 'main']}
T = json.load(open('logs/significance_results_tlaft.json')) if os.path.exists('logs/significance_results_tlaft.json') else {}


def cell(res, ds, dash='--'):
    if ds not in res:
        return dash
    a = res[ds]['methods']['ec2']['auc']['acc']
    return f"{a['mean']:.3f}$\\pm${a['std']:.3f}"


def count(info, sign):
    return sum(1 for b in BASE if b in info['methods'] and 'acc' in info['methods'][b]['p']
               and info['methods'][b]['p']['acc'] < 0.05 and sign * info['methods'][b]['diff']['acc']['mean'] > 0)


lines = []
for ds, name in DS:
    m = R['main'][ds]
    best = max(m['methods'][b]['auc']['acc']['mean'] for b in BASE if b in m['methods'])
    full = cell(R['main'], ds)
    if m['methods']['ec2']['auc']['acc']['mean'] > best:
        full = f"\\textbf{{{full}}}"
    anchor_na = ds == 'bank_marketing'
    lines.append(f"{name} & {cell(R['twophase'], ds)} & {cell(R['noanchor'], ds)} & "
                 f"{full}{'$^{\\dagger}$' if anchor_na else ''} & {cell(T, ds)} & {best:.3f} & "
                 f"{count(m, 1)} / {count(m, -1)} \\\\")
open('logs/latex_ablation.tex', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
