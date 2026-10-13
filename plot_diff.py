"""EC2 − baselines 逐 budget 精度差值图 (cube_nm + syn1/2/3)。

展示"先投资后收割"模式: b1–2 因购置 gate/context 特征而落后, b4+ 反超并显著领先。
粗线 = EC2 − 该 budget 下最优 baseline; 细线 = EC2 − 各 baseline (3-seed 均值)。
粗线上的实心标记 = 该步 EC2 与全部 baseline 的差异均显著 (配对 t, p<0.05)。

输出: figure/ec2_vs_baselines_acc_diff.png / .pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analyze_significance import BASELINES, LABEL, OUT_SUFFIX, get_curves, ttest_rel_3

DATASETS = ['cube_nm', 'syn1', 'syn2', 'syn3']
TITLES = {'cube_nm': 'CUBE-NM', 'syn1': 'Syn1', 'syn2': 'Syn2', 'syn3': 'Syn3'}
XMAX = {'cube_nm': 12}
COLORS = {'rand': '#aaaaaa', 'cae': '#1f77b4', 'eddi': '#ff7f0e', 'ga': '#2ca02c',
          'iter': '#d62728', 'gdfs': '#9467bd', 'aaco': '#8c564b', 'dime': '#e377c2', 'ol': '#17becf'}


def main():
    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    axes = axes.ravel()

    for ax, ds in zip(axes, DATASETS):
        curves, budgets = get_curves(ds)
        ec2 = curves['ec2']['acc']
        b = int(len(budgets) if ds not in XMAX else XMAX[ds] + 1)

        diffs, accs = {}, {}
        for name in BASELINES:
            if name not in curves or 'acc' not in curves[name]:
                continue
            diffs[name] = (ec2[:, :b] - curves[name]['acc'][:, :b]).mean(axis=0)
            accs[name] = curves[name]['acc'][:, :b].mean(axis=0)

        for name, d in diffs.items():
            ax.plot(range(b), d, lw=1.0, color=COLORS[name], alpha=0.55,
                    label=LABEL[name] if ds == 'cube_nm' else None)

        best = np.max(list(accs.values()), axis=0)
        margin = ec2[:, :b].mean(axis=0) - best
        ax.plot(range(b), margin, lw=2.6, color='black', zorder=5,
                label='vs. best baseline' if ds == 'cube_nm' else None)
        for k in range(1, b):
            better = worse = 0
            for name in diffs:
                d = ec2[:, k] - curves[name]['acc'][:, k]
                t, p = ttest_rel_3(d)
                if p < 0.05:
                    better += d.mean() > 0
                    worse += d.mean() < 0
            if better == len(diffs):
                ax.plot(k, margin[k], 'o', color='black', ms=7, zorder=6)
            elif worse == len(diffs):
                ax.plot(k, margin[k], 'v', color='black', ms=7, zorder=6)
        ax.axhline(0, color='grey', ls='--', lw=1)
        ax.set_title(TITLES[ds], fontsize=13)
        ax.set_xlabel('Budget')

    axes[0].set_ylabel('$\\Delta$ accuracy vs. baselines')
    axes[2].set_ylabel('$\\Delta$ accuracy vs. baselines')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ['png', 'pdf']:
        fig.savefig(f'figure/ec2_vs_baselines_acc_diff{OUT_SUFFIX}.{ext}', dpi=300, bbox_inches='tight')
    print(f'saved figure/ec2_vs_baselines_acc_diff{OUT_SUFFIX}.png')


if __name__ == '__main__':
    main()
