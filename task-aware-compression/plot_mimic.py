"""MIMIC-CXR test split: rate vs clinical accuracy and rate vs agreement with the uncompressed report.
Reads results/mimic_full/fidelity_stats.json.  python plot_mimic.py [stats.json] [out.png]"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'results/mimic_full/fidelity_stats.json'))
OUT = sys.argv[2] if len(sys.argv) > 2 else 'mimic_rate_vs_quality_test.png'
BLUE, AQUA, INK, MUTED, SURFACE = '#2a78d6', '#1baf7a', '#0b0b0b', '#52514e', '#fcfcfb'
T = (500, 1000, 2000)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200, facecolor=SURFACE)
panels = [('abn_f1', 'Abnormal-finding F1 vs radiologist report', S['test_500']['A_f1'], 'Uncompressed R2Gen', lambda v: f'{v:.3f}', (0.16, 0.30)),
          ('findings', 'Same findings as the uncompressed report (%)', 100.0, 'Uncompressed R2Gen = 100%', lambda v: f'{v:.0f}%', (20, 106))]
for ax, (key, ylabel, ceil, ceil_label, fmt, ylim) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.axhline(ceil, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(450, ceil + (ylim[1] - ylim[0]) * 0.012, ceil_label + (f' = {ceil:.3f}' if key == 'abn_f1' else ''), ha='left', va='bottom', color=MUTED, fontsize=8.5)
    for cond, color, name, dy in (('Bft', BLUE, 'Feature-only codec', -15), ('F', AQUA, 'Task-aware codec (uncompressed-report loss)', 9)):
        xs = [S[f'test_{t}'][cond]['bits'] for t in T]; ys = [S[f'test_{t}'][cond][key] for t in T]
        ax.plot(xs, ys, color=color, lw=2, marker='o', ms=7, mec=SURFACE, mew=1.5, label=name)
        for x, y in zip(xs, ys):
            ax.annotate(fmt(y), (x, y), textcoords='offset points', xytext=(0, dy), ha='center', color=INK, fontsize=9)
    ax.set_xscale('log', base=2); ax.set_xticks(T); ax.set_xticklabels(['500', '1,000', '2,000']); ax.set_xlim(430, 2400)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Actual compressed feature bits per image', color=MUTED); ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color='#e6e5e1', lw=0.8); ax.tick_params(colors=MUTED)
    for sp in ax.spines.values(): sp.set_visible(False)
axes[1].legend(frameon=False, loc='lower right', labelcolor=INK)
fig.suptitle('MIMIC-CXR test (3,858 images): task-aware vs feature-only compression of R2Gen features', color=INK, fontsize=11.5, x=0.01, ha='left')
fig.text(0.01, 0.01, 'Frozen R2Gen (MIMIC-CXR checkpoint). Findings from the rule-based labeller (ta-spans-1.0). One training run per codec. Uncompressed feature: 3.2 million bits per image.', color=MUTED, fontsize=7.5, va='bottom')
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(OUT, facecolor=SURFACE); print('saved', OUT)
