"""Step 4. Evaluate conditions A (original), B (standard codec), C (task-aware), D (ablations).

Every codec condition is evaluated through the deployed path: compress -> write .bin file ->
read file -> decompress -> frozen R2Gen beam search (identical decoding args for all conditions).
The rate is the actual file size (all bytes of the per-study container; codec weights are pre-shared).

    python task_aware/evaluate.py --split val --out_dir ../results/ta_eval_val \
        --cond A=original --cond B_r1000=../results/ta_runs/B_r1000/best.pt \
        --cond C_r1000=../results/ta_runs/C_r1000/best.pt ... \
        --pair C_r1000:B_r1000 -- <R2Gen args>

Naming: '<family>_<anything>'; families (text before the first '_') form rate-quality curves and
are compared at matched actual rates (piecewise-linear in log-rate over the overlapping range).
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta_common as tc  # noqa: E402
import feature_codec as fcodec  # noqa: E402
import clinical_spans as cs  # noqa: E402

CLIN_KEYS = ['tp_mention', 'fp_mention', 'fn_mention', 'tp_abn', 'fp_abn', 'fn_abn', 'omissions',
             'polarity_errors', 'uncertainty_errors', 'unsupported_findings']


def nlg_scores(gts, res):
    """Corpus scores and per-study scores; each scorer optional (METEOR needs Java)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    corpus, per = {}, {}
    scorers = [(Bleu(4), ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4']), (Rouge(), 'ROUGE_L'), (Cider(), 'CIDEr')]
    try:
        from pycocoevalcap.meteor import Meteor
        scorers.append((Meteor(), 'METEOR'))
    except Exception as e:  # noqa: BLE001
        print('METEOR unavailable:', e)
    for scorer, names in scorers:
        try:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
        except Exception as e:  # noqa: BLE001
            print(f'{names} failed: {e}')
            continue
        if isinstance(names, list):
            for n, s, ss in zip(names, score, scores):
                corpus[n], per[n] = float(s), [float(v) for v in ss]
        else:
            corpus[names], per[names] = float(score), [float(v) for v in scores]
    return corpus, per


def clinical_summary(counts):
    s = {k: sum(c[k] for c in counts) for k in CLIN_KEYS}
    n = len(counts)
    p, r, f = cs.prf(s['tp_abn'], s['fp_abn'], s['fn_abn'])
    pm, rm, fm = cs.prf(s['tp_mention'], s['fp_mention'], s['fn_mention'])
    return {'abn_precision': p, 'abn_recall': r, 'abn_f1': f, 'mention_precision': pm, 'mention_recall': rm,
            'mention_f1': fm, 'omissions_per_study': s['omissions'] / n,
            'polarity_errors_per_study': s['polarity_errors'] / n,
            'uncertainty_errors_per_study': s['uncertainty_errors'] / n,
            'unsupported_findings_per_study': s['unsupported_findings'] / n}


def run_condition(name, spec, r2, store, bank, tokenizer, a, device, out_dir):
    n = len(store) if not a.max_n else min(a.max_n, len(store))
    raw_bytes = int(np.prod(store.features.shape[1:])) * 4  # float32 storage of both views
    recs = []
    codec = None
    if spec != 'original':
        codec, ck = fcodec.load_codec(spec, device)
        bdir = out_dir / 'bitstreams' / name
        bdir.mkdir(parents=True, exist_ok=True)
    max_dev = 0.0
    for i in range(0, n, a.batch_size):
        idx = list(range(i, min(n, i + a.batch_size)))
        x = store.batch(idx, device)
        B = x.shape[0]
        if codec is None:
            fh, nbytes = x, [raw_bytes] * B
        else:
            xf = x.flatten(0, 1)
            strings, hw = codec.compress(xf)
            nbytes, views = [], []
            for b in range(B):
                path = bdir / f'{store.ids[idx[b]]}.bin'
                path.write_bytes(fcodec.pack_study(strings[x.shape[1] * b:x.shape[1] * (b + 1)], hw))
                blob = path.read_bytes()                     # decode from the stored file
                v, hw2 = fcodec.unpack_study(blob)
                if hw2 != hw:
                    raise ValueError('grid mismatch')
                views += v
                nbytes.append(path.stat().st_size)
            fh = codec.decompress(views, hw).view_as(x)
            with torch.no_grad():
                ref_round = codec(xf, mode='round')['x_hat'].view_as(x)
            max_dev = max(max_dev, (fh - ref_round).abs().max().item())
        t, m, w = bank.batch(idx, device)
        with torch.no_grad():
            lp = r2.teacher_forced(*tc.views(fh), t)
            gen = r2.generate(*tc.views(fh))
            nmse = [tc.feature_nmse(fh[b:b + 1].flatten(0, 1), x[b:b + 1].flatten(0, 1)).item() for b in range(B)]
        for b in range(B):
            recs.append({'id': store.ids[idx[b]], 'bytes': nbytes[b], 'bits': 8 * nbytes[b],
                         'nll': tc.weighted_nll(lp[b:b + 1], t[b:b + 1], m[b:b + 1]).item(),
                         'wnll': tc.weighted_nll(lp[b:b + 1], t[b:b + 1], m[b:b + 1], w[b:b + 1]).item(),
                         'feature_nmse': nmse[b], 'generated': gen[b], 'reference': bank.reports[idx[b]]})
        print(f'\r{name}: {len(recs)}/{n}', end='', flush=True)
    print()
    return recs, {'decode_vs_forward_max_abs_diff': max_dev, 'raw_bytes_per_study': raw_bytes}


def summarize(name, recs, ref_recs, extra):
    gts = {i: [r['reference']] for i, r in enumerate(recs)}
    res = {i: [r['generated'] if r['generated'].strip() else '.'] for i, r in enumerate(recs)}
    corpus, per = nlg_scores(gts, res)
    for k, v in per.items():
        for r, s in zip(recs, v):
            r[k] = s
    for r, rr in zip(recs, ref_recs):
        lab_ref = cs.report_labels(r['reference'].split())
        lab_gen = cs.report_labels(r['generated'].split())
        lab_a = cs.report_labels(rr['generated'].split())
        r['clin_vs_ref'] = cs.compare_labels(lab_ref, lab_gen)
        r['clin_vs_A'] = cs.compare_labels(lab_a, lab_gen)
        r['labels'] = lab_gen
        r['same_text_as_A'] = r['generated'] == rr['generated']
    n = len(recs)
    s = {'condition': name, 'studies': n, 'bytes_per_study': sum(r['bytes'] for r in recs) / n,
         'bits_per_study': sum(r['bits'] for r in recs) / n,
         'compression_ratio': extra['raw_bytes_per_study'] / (sum(r['bytes'] for r in recs) / n),
         'text_nll': sum(r['nll'] for r in recs) / n, 'text_nll_weighted': sum(r['wnll'] for r in recs) / n,
         'feature_nmse': sum(r['feature_nmse'] for r in recs) / n,
         'identical_to_A_percent': 100.0 * sum(r['same_text_as_A'] for r in recs) / n, **corpus}
    s.update({f'ref_{k}': v for k, v in clinical_summary([r['clin_vs_ref'] for r in recs]).items()})
    s.update({f'A_{k}': v for k, v in clinical_summary([r['clin_vs_A'] for r in recs]).items()})
    s.update(extra)
    return s


def per_concept(recs):
    agg = defaultdict(lambda: [0, 0, 0])
    pos = {'positive', 'uncertain'}
    for r in recs:
        ref = cs.report_labels(r['reference'].split())
        gen = r['labels']
        for c in set(ref) | set(gen):
            a_, b_ = ref.get(c) in pos, gen.get(c) in pos
            agg[c][0] += a_ and b_
            agg[c][1] += b_ and not a_
            agg[c][2] += a_ and not b_
    return {c: dict(zip(['precision', 'recall', 'f1'], cs.prf(*v)), support=v[0] + v[2]) for c, v in sorted(agg.items())}


def paired_bootstrap(ra, rb, n_boot, seed):
    """Differences a-b on the same studies with 95% CI: mean per-study metrics and micro clinical F1."""
    rng = np.random.default_rng(seed)
    N = len(ra)
    assert [r['id'] for r in ra] == [r['id'] for r in rb]
    per_keys = [k for k in ('nll', 'wnll', 'BLEU_4', 'ROUGE_L', 'CIDEr', 'METEOR', 'bits') if k in ra[0] and k in rb[0]]

    def stats(sel):
        out = {k: float(np.mean([ra[i][k] for i in sel]) - np.mean([rb[i][k] for i in sel])) for k in per_keys}
        for tag in ('clin_vs_ref', 'clin_vs_A'):
            fa = clinical_summary([ra[i][tag] for i in sel])
            fb = clinical_summary([rb[i][tag] for i in sel])
            for k in ('abn_f1', 'mention_f1', 'omissions_per_study', 'polarity_errors_per_study',
                      'unsupported_findings_per_study'):
                out[f'{tag}:{k}'] = fa[k] - fb[k]
        return out

    point = stats(range(N))
    boots = defaultdict(list)
    for _ in range(n_boot):
        sel = rng.integers(0, N, N)
        for k, v in stats(sel).items():
            boots[k].append(v)
    return {k: {'diff': point[k], 'ci95': [float(np.nanpercentile(boots[k], 2.5)),
                                           float(np.nanpercentile(boots[k], 97.5))]} for k in point}


def matched_rate(summaries, fam_a, fam_b, metric):
    """Average metric difference (a-b) over the overlapping log-rate interval (BD-metric style)."""
    def curve(f):
        pts = sorted((math.log(s['bits_per_study']), s[metric]) for s in summaries
                     if s['condition'].split('_')[0] == f and s.get(metric) is not None
                     and not (isinstance(s[metric], float) and math.isnan(s[metric])))
        return np.array(pts) if len(pts) >= 2 else None
    ca, cb = curve(fam_a), curve(fam_b)
    if ca is None or cb is None:
        return None
    lo, hi = max(ca[0, 0], cb[0, 0]), min(ca[-1, 0], cb[-1, 0])
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, 200)
    d = np.interp(grid, ca[:, 0], ca[:, 1]) - np.interp(grid, cb[:, 0], cb[:, 1])
    return {'avg_diff': float(d.mean()), 'bits_range': [float(math.exp(lo)), float(math.exp(hi))]}


def plot(summaries, out_dir, a_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('ref_abn_f1', 'Clinical abnormal-finding F1 vs reference'),
               ('A_mention_f1', 'Status-aware mention F1 vs uncompressed output'),
               ('BLEU_4', 'BLEU-4'), ('text_nll', 'Teacher-forced NLL (lower is better)')]
    fams = sorted({s['condition'].split('_')[0] for s in summaries if s['condition'] != 'A'})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (m, title) in zip(axes, metrics):
        for f in fams:
            pts = sorted((s['bits_per_study'], s.get(m)) for s in summaries if s['condition'].split('_')[0] == f)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', label=f)
        if a_summary and a_summary.get(m) is not None:
            ax.axhline(a_summary[m], color='k', ls='--', lw=1, label='A (uncompressed)')
        ax.set_xscale('log')
        ax.set_xlabel('actual bits per study (both views)')
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'rate_quality.png', dpi=150)


def main():
    mine, theirs = tc.split_argv()
    p = argparse.ArgumentParser()
    p.add_argument('--feature_dir', required=True)
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--out_dir', required=True)
    p.add_argument('--cond', action='append', required=True, help='NAME=original or NAME=path/to/best.pt')
    p.add_argument('--pair', action='append', default=[], help='X:Y paired bootstrap of X minus Y')
    p.add_argument('--alpha', type=float, default=2.0, help='span weight used only for the reported weighted NLL')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_n', type=int, default=0)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(mine)
    args = tc.r2gen_args(theirs)
    tc.fix_seed(args.seed)
    device = torch.device(a.device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r2, tokenizer = tc.load_frozen_r2gen(args, device)
    store = tc.FeatureStore(a.feature_dir, a.split)
    bank = tc.TargetBank(tc.load_annotation(args), a.split, tokenizer, args.max_seq_length, a.alpha, store.ids)
    conds = [c.split('=', 1) for c in a.cond]
    if conds[0][0] != 'A' or conds[0][1] != 'original':
        conds.insert(0, ['A', 'original'])
    all_recs, extras, summaries = {}, {}, []
    for name, spec in conds:
        all_recs[name], extras[name] = run_condition(name, spec, r2, store, bank, tokenizer, a, device, out_dir)
    for name, _ in conds:
        summaries.append(summarize(name, all_recs[name], all_recs['A'], extras[name]))
    result = {'split': a.split, 'studies': len(all_recs['A']), 'decoding': {k: getattr(args, k) for k in (
        'sample_method', 'beam_size', 'block_trigrams', 'max_seq_length')},
        'r2gen_checkpoint_sha256': tc.sha256_file(args.load), 'rules_version': cs.RULES_VERSION,
        'summaries': summaries, 'per_concept': {k: per_concept(v) for k, v in all_recs.items()}, 'pairs': {}}
    for pr in a.pair:
        x, y = pr.split(':')
        result['pairs'][pr] = paired_bootstrap(all_recs[x], all_recs[y], a.bootstrap, a.seed)
    fams = sorted({s['condition'].split('_')[0] for s in summaries if s['condition'] != 'A'})
    result['matched_rate'] = {}
    for fa in fams:
        for fb in fams:
            if fa != fb:
                result['matched_rate'][f'{fa}-{fb}'] = {m: matched_rate(summaries, fa, fb, m) for m in (
                    'ref_abn_f1', 'ref_mention_f1', 'A_mention_f1', 'BLEU_4', 'CIDEr', 'text_nll')}
    tc.json_save(out_dir / 'summary.json', result)
    tc.json_save(out_dir / 'per_study.json', all_recs)
    keys = list(summaries[0].keys())
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(summaries)
    try:
        plot(summaries, out_dir, summaries[0])
    except Exception as e:  # noqa: BLE001
        print('plot failed:', e)
    cols = ['condition', 'bits_per_study', 'compression_ratio', 'text_nll', 'BLEU_4', 'ref_abn_f1', 'A_mention_f1',
            'identical_to_A_percent']
    print('\n' + ' | '.join(cols))
    for s in summaries:
        print(' | '.join(f'{s.get(c):.4g}' if isinstance(s.get(c), float) else str(s.get(c)) for c in cols))
    print('\nwritten:', out_dir.resolve())


if __name__ == '__main__':
    main()
