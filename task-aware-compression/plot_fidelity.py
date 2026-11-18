"""Rate vs agreement with the uncompressed R2Gen report (IU X-Ray validation, 296 studies).

Reads results/ta_full/eval_*/per_study.json. Agreement is measured against condition A
(uncompressed), so 100% is a hard ceiling that no codec can exceed.
If eval_fidelity_<TAG>_r<bits>/ folders exist (run_fidelity.sh), the fidelity codec is added.
Run from the medical-compression folder:  python plot_fidelity.py [results/ta_full] [out.png] [TAG]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/ta_full')
RUNS = {500: 'eval_compare_r500', 1000: 'eval_compare_r1000', 2000: 'eval_compare_r2000',
        4000: 'eval_B_c64_r4000', 8000: 'eval_B_c64_r8000'}
TAG = sys.argv[3] if len(sys.argv) > 3 else 'F'
AQUA = '#1baf7a'
BLUE, ORANGE, INK, MUTED, SURFACE = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#fcfcfb'

pts = {}
for target, folder in RUNS.items():
    d = json.loads((ROOT / folder / 'per_study.json').read_text())
    a = {r['id']: r for r in d['A']}
    for cond, rows in d.items():
        if cond == 'A':
            continue
        n = len(rows)
        pts[(target, cond)] = (
            sum(r['bits'] for r in rows) / n,
            100.0 * sum(r['labels'] == a[r['id']]['labels'] for r in rows) / n,
            100.0 * sum(r['generated'] == a[r['id']]['generated'] for r in rows) / n)

for target in (500, 1000, 2000, 4000, 8000):
    f = ROOT / f'eval_fidelity_{TAG}_r{target}' / 'per_study.json'
    if f.exists():
        d = json.loads(f.read_text())
        a = {r['id']: r for r in d['A']}
        rows = d['F']
        n = len(rows)
        pts[(target, 'F')] = (
            sum(r['bits'] for r in rows) / n,
            100.0 * sum(r['labels'] == a[r['id']]['labels'] for r in rows) / n,
            100.0 * sum(r['generated'] == a[r['id']]['generated'] for r in rows) / n)

base = [pts[(t, 'Bft')] for t in (500, 1000, 2000)] + [pts[(t, 'B')] for t in (4000, 8000)]
task = [pts[(t, 'C')] for t in (500, 1000, 2000)]

fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axhline(100, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(8000, 100.8, 'Uncompressed R2Gen = 100% (ceiling)', ha='right', va='bottom', color=MUTED, fontsize=9)
fid = [pts[k] for k in sorted(pts) if k[1] == 'F']
series_list = [(base, BLUE, 'Feature-only codec', 9), (task, ORANGE, 'Task-aware codec (reference-report loss)', -15)]
if fid:
    series_list.append((fid, AQUA, 'Task-aware codec (uncompressed-report loss)', 9))
    series_list[0] = (base, BLUE, 'Feature-only codec', -15)
for series, color, name, dy in series_list:
    ax.plot([p[0] for p in series], [p[1] for p in series], color=color, lw=2, marker='o', ms=7,
            mec=SURFACE, mew=1.5, label=name)
    for p in series:
        ax.annotate(f'{p[1]:.0f}%', (p[0], p[1]), textcoords='offset points',
                    xytext=(0, dy), ha='center', color=INK, fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks([500, 1000, 2000, 4000, 8000])
ax.set_xticklabels(['500', '1,000', '2,000', '4,000', '8,000'])
ax.set_ylim(60, 106)
ax.set_xlabel('Actual compressed feature bits per study', color=MUTED)
ax.set_ylabel('Studies with the same findings as the\nuncompressed report (%)', color=MUTED)
ax.set_title('IU X-Ray validation: how much of the uncompressed report survives compression',
             color=INK, fontsize=11, loc='left')
ax.grid(True, color='#e6e5e1', lw=0.8)
ax.tick_params(colors=MUTED)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(frameon=False, loc='lower right', labelcolor=INK)
fig.text(0.01, 0.01, '296 studies. Findings from the rule-based labeller (ta-spans-1.0), compared per study with the '
         'uncompressed report.\n4,000 and 8,000-bit points are the feature-only codec before the extra fine-tuning.',
         color=MUTED, fontsize=7.5, va='bottom')
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('codec_rate_vs_fidelity.png')
fig.savefig(out, facecolor=SURFACE)
print('saved', out)
for k, v in sorted(pts.items()):
    print(k, 'bits %.0f  same findings %.1f%%  identical report %.1f%%' % v)
