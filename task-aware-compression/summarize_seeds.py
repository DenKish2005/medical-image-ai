"""Pool the fidelity comparison (F vs Bft) over random seeds.

Reads, for each split and target rate:
  seed 1234: results/ta_full/eval_fidelity_F_r<bits>/ (val) and eval_fidelity_test_r<bits>/ (test)
  new seeds: results/ta_full/eval_seed_<split>_r<bits>_s<seed>/
Agreement is with the uncompressed report of the same study. The pooled interval resamples
studies and averages each study's F-minus-Bft difference over seeds.
Run from the medical-compression folder:  python summarize_seeds.py [results/ta_full]
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/ta_full')
random.seed(0)


def load(folder):
    d = json.loads((folder / 'per_study.json').read_text())
    a = {r['id']: r for r in d['A']}
    out = {}
    for cond in ('Bft', 'F'):
        rows = {r['id']: r for r in d[cond]}
        ids = sorted(rows)
        out[cond] = {
            'ids': ids, 'bits': sum(rows[i]['bits'] for i in ids) / len(ids),
            'findings': [int(rows[i]['labels'] == a[i]['labels']) for i in ids],
            'identical': [int(rows[i]['generated'] == a[i]['generated']) for i in ids]}
    return out


def interval(per_study, n_boot=10000):
    n = len(per_study)
    means = sorted(sum(per_study[random.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return 100 * sum(per_study) / n, 100 * means[int(0.025 * n_boot)], 100 * means[int(0.975 * n_boot)]


runs = {}
for f in sorted(ROOT.glob('eval_seed_*_r*_s*')):
    m = re.fullmatch(r'eval_seed_(val|test)_r(\d+)_s(\d+)', f.name)
    if m and (f / 'per_study.json').exists():
        runs.setdefault((m.group(1), int(m.group(2))), {})[int(m.group(3))] = f
for split, pattern in (('val', 'eval_fidelity_F_r{}'), ('test', 'eval_fidelity_test_r{}')):
    for target in sorted({t for (_, t) in runs} | {500, 1000, 2000}):
        f = ROOT / pattern.format(target)
        if (f / 'per_study.json').exists() and (split, target) in runs:
            runs[(split, target)][1234] = f

result = {}
for (split, target), by_seed in sorted(runs.items()):
    data = {s: load(f) for s, f in sorted(by_seed.items())}
    print(f'\n=== {split}, target {target} bits, seeds {sorted(data)} ===')
    for metric in ('findings', 'identical'):
        print(f'  {metric}:')
        diffs = []
        for s, d in data.items():
            n = len(d['F'][metric])
            b, f_ = 100 * sum(d['Bft'][metric]) / n, 100 * sum(d['F'][metric]) / n
            diffs.append([x - y for x, y in zip(d['F'][metric], d['Bft'][metric])])
            print(f'    seed {s:>4}: Bft {b:5.1f}% ({d["Bft"]["bits"]:.0f} bits)   F {f_:5.1f}% ({d["F"]["bits"]:.0f} bits)   '
                  f'F-Bft {f_ - b:+.1f}')
        pooled = [sum(col) / len(col) for col in zip(*diffs)]
        mean, lo, hi = interval(pooled)
        print(f'    pooled over {len(diffs)} seeds: F-Bft {mean:+.1f} points (95% interval {lo:+.1f} to {hi:+.1f})')
        result[f'{split}_r{target}_{metric}'] = {'seeds': sorted(data), 'diff': mean, 'ci95': [lo, hi]}
out = ROOT / 'seed_summary.json'
out.write_text(json.dumps(result, indent=2))
print('\nsaved', out)
