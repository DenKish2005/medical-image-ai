"""Compare teacher-forced codec KL on the validation split only.

Run from R2Gen root, with this file inside task_aware. Both uncompressed and
reconstructed feature predictions use the same reference prefix. Values are
per-study means across nonpadding target positions, then averaged by study.
This performs no training, generation, entropy coding, or test evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta_common as tc
import feature_codec as fcodec
from train_codec_distill import distillation_kl


def main():
    mine, theirs = tc.split_argv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature_dir', required=True)
    p.add_argument('--out', required=True, help='new JSON output file')
    p.add_argument('--cond', action='append', required=True, metavar='NAME=CHECKPOINT')
    p.add_argument('--pair', action='append', default=[], metavar='NAME1:NAME2')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args(mine)
    if a.batch_size < 1 or a.bootstrap < 1:
        raise ValueError('batch_size and bootstrap must be positive')
    outpath = Path(a.out)
    if outpath.exists():
        raise FileExistsError(f'{outpath} already exists; choose a new output')
    conditions = {}
    for spec in a.cond:
        if '=' not in spec:
            raise ValueError(f'Expected NAME=CHECKPOINT, got {spec}')
        name, path = spec.split('=', 1)
        if not name or not path or name in conditions:
            raise ValueError(f'Invalid or duplicate condition: {spec}')
        conditions[name] = path
    for pair in a.pair:
        names = pair.split(':')
        if len(names) != 2 or any(n not in conditions for n in names):
            raise ValueError(f'Invalid pair: {pair}')
    args = tc.r2gen_args(theirs)
    device = torch.device(a.device)
    tc.fix_seed(a.seed)
    r2, tokenizer = tc.load_frozen_r2gen(args, device)
    store = tc.FeatureStore(a.feature_dir, 'val')
    sha = tc.sha256_file(args.load)
    stored_sha = store.manifest.get('r2gen_checkpoint_sha256')
    if stored_sha is not None and stored_sha != sha:
        raise ValueError('Feature cache and R2Gen checkpoint hashes differ.')
    ann = tc.load_annotation(args)
    bank = tc.TargetBank(ann, 'val', tokenizer, args.max_seq_length, 2.0, store.ids)
    codecs, checkpoint_val = {}, {}
    for name, path in conditions.items():
        codec, ckpt = fcodec.load_codec(path, 'cpu')
        ckpt_sha = (ckpt.get('config') or {}).get('r2gen_checkpoint_sha256')
        if ckpt_sha is not None and ckpt_sha != sha:
            raise ValueError(f'{name}: codec checkpoint and R2Gen checkpoint hashes differ.')
        codecs[name] = codec.to(device).eval()
        checkpoint_val[name] = (ckpt.get('val') or {}).get('distill_kl')
    n = len(store)
    if n == 0:
        raise ValueError('Empty validation split.')
    values = {name: [] for name in conditions}
    print(f'Validation: {n} studies; conditions: {", ".join(conditions)}', flush=True)
    with torch.no_grad():
        for start in range(0, n, a.batch_size):
            idx = list(range(start, min(start + a.batch_size, n)))
            x = store.batch(idx, device)
            t, m, _ = bank.batch(idx, device)
            teacher = r2.teacher_forced(*tc.views(x), t)
            for name, codec in codecs.items():
                out = codec(x.flatten(0, 1), mode='round')
                fh = out['x_hat'].view_as(x)
                student = r2.teacher_forced(*tc.views(fh), t)
                for b in range(len(idx)):
                    values[name].append(distillation_kl(student[b:b+1], teacher[b:b+1], m[b:b+1]).item())
            print(f'Processed {min(start + a.batch_size, n)}/{n}', flush=True)
    means = {name: float(np.mean(v)) for name, v in values.items()}
    for name, mean in means.items():
        print(f'{name}: teacher-to-codec KL = {mean:.8f}; checkpoint KL = {checkpoint_val[name]}')
    rng = np.random.default_rng(a.seed)
    bootstrap_indices = rng.integers(0, n, size=(a.bootstrap, n))
    pairs = {}
    for pair in a.pair:
        first, second = pair.split(':')
        d = np.asarray(values[first]) - np.asarray(values[second])
        intervals = np.quantile(d[bootstrap_indices].mean(axis=1), [0.025, 0.975])
        pairs[pair] = {'diff': float(d.mean()), 'ci95': [float(x) for x in intervals]}
        print(f'{pair}: mean KL difference = {d.mean():+.8f}; paired bootstrap 95% interval = '
              f'[{intervals[0]:+.8f}, {intervals[1]:+.8f}]')
    result = {'split': 'val', 'studies': n, 'conditions': conditions, 'kl_mean': means,
              'paired_bootstrap': pairs, 'checkpoint_val_kl': checkpoint_val,
              'r2gen_checkpoint_sha256': sha, 'seed': a.seed, 'bootstrap_samples': a.bootstrap,
              'ids': store.ids, 'per_study_kl': values}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
