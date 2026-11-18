"""Test-split version of plot_fidelity.py: reads results/ta_full/eval_fidelity_test_r<bits>/per_study.json.
Run from the medical-compression folder:  python plot_fidelity_test.py [results/ta_full] [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/ta_full')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('codec_rate_vs_fidelity_test.png')
BLUE, ORANGE, AQUA, INK, MUTED, SURFACE = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#fcfcfb'
SERIES = [('Bft', BLUE, 'Feature-only codec', -16), ('C', ORANGE, 'Task-aware codec (reference-report loss)', -15),
          ('F', AQUA, 'Task-aware codec (uncompressed-report loss)', 9)]

pts, n = {}, 0
for target in (500, 1000, 2000):
    d = json.loads((ROOT / f'eval_fidelity_test_r{target}' / 'per_study.json').read_text())
    a = {r['id']: r for r in d['A']}
    for cond, _, _, _ in SERIES:
        rows = d[cond]
        n = len(rows)
        pts[(target, cond)] = (sum(r['bits'] for r in rows) / n,
                               100.0 * sum(r['labels'] == a[r['id']]['labels'] for r in rows) / n)

fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(100, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(2000, 100.8, 'Uncompressed R2Gen = 100% (ceiling)', ha='right', va='bottom', color=MUTED, fontsize=9)
for cond, color, name, dy in SERIES:
    s = [pts[(t, cond)] for t in (500, 1000, 2000)]
    ax.plot([p[0] for p in s], [p[1] for p in s], color=color, lw=2, marker='o', ms=7, mec=SURFACE, mew=1.5, label=name)
    for j, p in enumerate(s):
        near = cond == 'Bft' and j == 0   # keep clear of the orange label at 500 bits
        ax.annotate(f'{p[1]:.0f}%', (p[0], p[1]), textcoords='offset points', xytext=(14, -3) if near else (0, dy),
                    ha='left' if near else 'center', color=INK, fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks([500, 1000, 2000])
ax.set_xticklabels(['500', '1,000', '2,000'])
ax.set_xlim(440, 2300)
ax.set_ylim(60, 106)
ax.set_xlabel('Actual compressed feature bits per study', color=MUTED)
ax.set_ylabel('Studies with the same findings as the\nuncompressed report (%)', color=MUTED)
ax.set_title('IU X-Ray test: how much of the uncompressed report survives compression', color=INK, fontsize=11, loc='left')
ax.grid(True, color='#e6e5e1', lw=0.8)
ax.tick_params(colors=MUTED)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.legend(frameon=False, loc='lower right', labelcolor=INK)
fig.text(0.01, 0.01, f'{n} test studies. Findings from the rule-based labeller (ta-spans-1.0), compared per study with '
         'the uncompressed report.', color=MUTED, fontsize=7.5, va='bottom')
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT, facecolor=SURFACE)
print('saved', OUT)
