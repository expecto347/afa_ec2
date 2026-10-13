"""从 significance_results.json 生成附录用逐 budget 配对检验 LaTeX 片段。

输出: logs/latex_perbudget.tex (acc, budgets 1-8, 4 个非短视优势数据集)
"""
import json

DS = [('syn1', 'Syn1'), ('syn2', 'Syn2'), ('syn3', 'Syn3'), ('cube_nm', 'CUBE-NM')]
COLS = ['rand', 'cae', 'eddi', 'ga', 'iter', 'gdfs', 'aaco', 'dime', 'ol']
NAME = {'rand': 'Random', 'cae': 'CAE', 'eddi': 'EDDI', 'ga': 'Greedy-Abl',
        'iter': 'Iterative', 'gdfs': 'GDFS', 'aaco': 'AACO', 'dime': 'DIME', 'ol': 'OL'}

data = json.load(open('logs/significance_results_main.json'))
lines = []
for ds, disp in DS:
    lines.append(r'\emph{' + disp + r'}' + ' &' * len(COLS) + r' \\[2pt]')
    lines.append('Budget & ' + ' & '.join(NAME[c] for c in COLS) + r' \\')
    for b in sorted(int(k) for k in data[ds]['per_budget']):
        row = data[ds]['per_budget'][str(b)]
        cells = [str(b)]
        for c in COLS:
            r = row.get(c)
            if r is None:
                cells.append('--')
                continue
            mk = ''
            if r['p'] < 0.01:
                mk = '^{**}'
            elif r['p'] < 0.05:
                mk = '^{*}'
            cells.append('$%+.3f%s$' % (r['diff_mean'], mk))
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\addlinespace[4pt]')

open('logs/latex_perbudget.tex', 'w').write('\n'.join(lines) + '\n')
print('wrote logs/latex_perbudget.tex')
