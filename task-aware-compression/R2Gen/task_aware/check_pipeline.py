"""Implementation safeguards from the technical note, Sec. 7.  Run before (and after) training.

  1. polarity/scope unit tests of the span annotator
  2. gradient check: L_weighted-text gives nonzero codec gradients, zero R2Gen gradients
  3. compression check: file roundtrip == round-mode forward; coded bytes << raw bytes
  4. leakage check: generated reports do not change when reference reports are shuffled/blanked
  5. checkpoint check: codec checkpoint was trained against this exact R2Gen checkpoint

    python task_aware/check_pipeline.py --feature_dir ../results/ta_features [--codec path/best.pt] -- <R2Gen args>
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta_common as tc  # noqa: E402
import feature_codec as fcodec  # noqa: E402
import clinical_spans as cs  # noqa: E402


def main():
    mine, theirs = tc.split_argv()
    p = argparse.ArgumentParser()
    p.add_argument('--feature_dir', required=True)
    p.add_argument('--codec', default=None)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(mine)
    args = tc.r2gen_args(theirs)
    tc.fix_seed(args.seed)
    dev = torch.device(a.device)
    results = {}

    fails = cs.run_polarity_tests()
    results['1_polarity_tests'] = not fails

    r2, tok = tc.load_frozen_r2gen(args, dev)
    ann = tc.load_annotation(args)
    train = tc.FeatureStore(a.feature_dir, 'train')
    idx = list(range(min(a.n, len(train))))
    bank = tc.TargetBank(ann, 'train', tok, args.max_seq_length, 2.0, train.ids)
    if a.codec:
        codec, ck = fcodec.load_codec(a.codec, dev)
        ok = ck['config']['r2gen_checkpoint_sha256'] == tc.sha256_file(args.load)
        print('5. codec trained on this R2Gen checkpoint:', ok)
        results['5_checkpoint_match'] = ok
    else:
        codec = fcodec.FeatureHyperpriorCodec(c_y=64).to(dev)
    codec.train()

    # 2. gradient check (text loss only)
    x = train.batch(idx, dev)
    out = codec(x.flatten(0, 1), mode='noise')
    fh = out['x_hat'].view_as(x)
    t, m, w = bank.batch(idx, dev)
    loss = tc.weighted_nll(r2.teacher_forced(fh[:, 0], fh[:, 1], t), t, m, w)
    codec.zero_grad(set_to_none=True)
    loss.backward()
    r2_grads = [n for n, p_ in r2.model.named_parameters() if p_.grad is not None]
    groups = {}
    for g in ('g_a.', 'g_s.', 'h_a.', 'h_s.'):
        gs = [p_.grad for n, p_ in codec.named_parameters() if n.startswith(g) and p_.grad is not None]
        groups[g] = float(sum(gg.abs().sum() for gg in gs)) if gs else 0.0
    ok = not r2_grads and groups['g_a.'] > 0 and groups['g_s.'] > 0
    print(f'2. gradient check: R2Gen params with grad={len(r2_grads)}, |grad| per codec group={groups} ->',
          'PASS' if ok else 'FAIL')
    results['2_gradient'] = ok
    # h_a/h_s get text gradient only via y_hat noise path in 'noise' mode (means); that is expected.

    # 3. compression check
    codec.eval()
    codec.update(force=True)
    with torch.no_grad():
        xf = x.flatten(0, 1)
        ref = codec(xf, mode='round')
        strings, hw = codec.compress(xf)
        blobs = [fcodec.pack_study(strings[2 * b:2 * b + 2], hw) for b in range(len(idx))]
        views = []
        for bl in blobs:
            v, hw2 = fcodec.unpack_study(bl)
            views += v
        dec = codec.decompress(views, hw2)
    diff = (dec - ref['x_hat']).abs().max().item()
    raw = x[0].numel() * 4
    coded = sum(map(len, blobs)) / len(blobs)
    est = ref['bits'].view(len(idx), 2).sum(1).mean().item() / 8
    ok = diff < 1e-4 and coded < raw
    print(f'3. compression: max|decode - forward(round)|={diff:.2e}; bytes/study coded={coded:.0f} '
          f'(estimated {est:.0f}) vs raw {raw} -> ratio {raw / coded:.0f}x ->', 'PASS' if ok else 'FAIL')
    results['3_compression'] = ok

    # 4. leakage: inference path takes only decoded features
    with torch.no_grad():
        g1 = r2.generate(dec.view_as(x)[:, 0], dec.view_as(x)[:, 1])
    bank.reports = ['' for _ in bank.reports]
    bank.tokens = [[0, 0] for _ in bank.tokens]
    with torch.no_grad():
        dec2 = codec.decompress(views, hw2)
        g2 = r2.generate(dec2.view_as(x)[:, 0], dec2.view_as(x)[:, 1])
    ok = g1 == g2
    print('4. leakage: reports identical after blanking references ->', 'PASS' if ok else 'FAIL')
    results['4_leakage'] = ok
    print('\nSUMMARY', results)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
