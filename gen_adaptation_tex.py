"""Extension table: two-phase Deep-EC2 vs. its optional extensions.

Columns: accuracy--AUC of the two-phase framework, + policy-adapted predictor (ft),
+ lookahead anchor (laft), and the final configuration chosen per dataset on the
validation set (select_by_validation.py -> logs/ec2_<ds>_final_seed*), its components,
the best baseline, and #baselines significantly better / worse (two-phase vs. final).

Needs logs/significance_results{,_ft,_laft,_final}.json from analyze_significance.py.
Output: logs/latex_adaptation.tex
"""
import json
import os

DS = [('syn1', 'Syn1'), ('syn2', 'Syn2'), ('syn3', 'Syn3'), ('cube_nm', 'CUBE-NM'),
      ('grid_stability', 'Grid Stability'), ('ckd', 'CKD'), ('metabric', 'Metabric'),
      ('bank_marketing', 'Bank Marketing'), ('mnist', 'MNIST'), ('miniboone', 'MiniBooNE'),
      ('california_housing', 'California Housing')]
BASE = ['cae', 'eddi', 'ga', 'iter', 'gdfs', 'aaco', 'dime', 'ol']
COMP = {'ft': 'A', 'laft': 'A+L', 'tft': 'A+T', 'tlaft': 'A+L+T', 's5ft': 'A', 's5laft': 'A+L'}

R = {k: json.load(open(f'logs/significance_results{v}.json'))
     for k, v in [('', '_tp'), ('_ft', '_ftv'), ('_laft', '_laftv'), ('_final', '_final')]}


def count(info, sign):
    return sum(1 for b in BASE
               if b in info['methods'] and 'acc' in info['methods'][b]['p']
               and info['methods'][b]['p']['acc'] < 0.05
               and sign * info['methods'][b]['diff']['acc']['mean'] > 0)


def cell(res, ds):
    if ds not in res:
        return '--'
    a = res[ds]['methods']['ec2']['auc']['acc']
    return f"{a['mean']:.3f}$\\pm${a['std']:.3f}"


lines = []
for ds, name in DS:
    o, f = R[''][ds], R['_final'][ds]
    chosen = os.readlink(f'logs/ec2_{ds}_final_seed0').split('_')[-2]
    best = max(o['methods'][b]['auc']['acc']['mean'] for b in BASE if b in o['methods'])
    fa = f['methods']['ec2']['auc']['acc']
    final = f"{fa['mean']:.3f}$\\pm${fa['std']:.3f}"
    if fa['mean'] > best:
        final = f"\\textbf{{{final}}}"
    lines.append(f"{name} & {cell(R[''], ds)} & {cell(R['_ft'], ds)} & {cell(R['_laft'], ds)} & "
                 f"{final} & {COMP[chosen]} & {best:.3f} & {count(o, 1)} / {count(o, -1)} & "
                 f"{count(f, 1)} / {count(f, -1)} \\\\")
open('logs/latex_adaptation.tex', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
