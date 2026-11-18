"""Step 2/3. Train the feature codec with R2Gen frozen.

Stage B (baseline):   L = mu*R + lambda_F * D_feature                      (no report loss)
Stage C (task-aware): L = mu*R + lambda_F * D_feature + lambda_T * L_weighted_text
                      initialised from a stage-B checkpoint (--init).

R      = estimated bits per STUDY (both views, y + z), -log2 likelihoods.
D_feat = normalized feature MSE (note Sec. 4.2).
L_text = clinically span-weighted teacher-forced NLL through the frozen R2Gen (note Sec. 4.3).

Rate control (--rate_mode):
  fixed  : mu = --lambda_rate (the note's L_rate + ... form when lambda_rate=1); sweep lambda_F/lambda_T.
  target : mu is a Lagrange multiplier adapted every step so the *rounded* (deployed) rate hits
           --target_bits per study.  Use this to obtain B and C at matched actual bit rates.

Only codec parameters are optimised; every R2Gen parameter has requires_grad=False and the
R2Gen forward is NOT wrapped in no_grad, so gradients reach F_hat and the codec.

Example (baseline, then task-aware at the same target rate):
  python task_aware/train_codec.py --stage baseline --feature_dir ../results/ta_features \
      --out_dir ../results/ta_runs/B_r2000 --rate_mode target --target_bits 2000 -- <R2Gen args>
  python task_aware/train_codec.py --stage task --init ../results/ta_runs/B_r2000/best.pt \
      --feature_dir ../results/ta_features --out_dir ../results/ta_runs/C_r2000_a2 \
      --rate_mode target --target_bits 2000 --lambda_t 1 --alpha 2 -- <R2Gen args>
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta_common as tc  # noqa: E402
import feature_codec as fcodec  # noqa: E402


def parse():
    mine, theirs = tc.split_argv()
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['baseline', 'task'])
    p.add_argument('--feature_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--init', default=None, help='codec checkpoint to start from (required for --stage task)')
    p.add_argument('--c_y', type=int, default=128)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--rate_mode', default='target', choices=['fixed', 'target'])
    p.add_argument('--lambda_rate', type=float, default=1e-3, help='mu in fixed mode (per bit per study)')
    p.add_argument('--target_bits', type=float, default=None, help='target rounded bits per study (target mode)')
    p.add_argument('--rate_tol', type=float, default=0.10, help='relative tolerance for best-epoch selection')
    p.add_argument('--mu_init', type=float, default=1e-3)
    p.add_argument('--mu_lr', type=float, default=0.02, help='step on log(mu) per update')
    p.add_argument('--lambda_f', type=float, default=1.0)
    p.add_argument('--lambda_t', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=2.0, help='span weight is 1+alpha')
    p.add_argument('--feature_norm', default='global', choices=['global', 'channel'])
    p.add_argument('--quant', default='noise', choices=['noise', 'ste'])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--aux_lr', type=float, default=1e-3)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--val_every', type=int, default=1)
    p.add_argument('--max_train', type=int, default=0)
    p.add_argument('--max_val', type=int, default=0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(mine)
    if a.stage == 'task' and a.lambda_t <= 0:
        raise ValueError('stage task needs --lambda_t > 0')
    if a.stage == 'baseline':
        a.lambda_t = 0.0
    if a.rate_mode == 'target' and not a.target_bits:
        raise ValueError('--rate_mode target needs --target_bits')
    if a.stage == 'task' and not a.init:
        print('WARNING: stage task without --init; the note initialises C from the baseline codec.')
    return a, tc.r2gen_args(theirs)


@torch.no_grad()
def channel_stats(store, max_n=0, chunk=64):
    n = len(store) if not max_n else min(max_n, len(store))
    s = s2 = None
    cnt = 0
    for i in range(0, n, chunk):
        x = torch.from_numpy(np.array(store.features[i:min(n, i + chunk)], dtype=np.float32))
        x = x.flatten(0, 1).double()  # views x samples
        if s is None:
            s = torch.zeros(x.shape[1], dtype=torch.float64)
            s2 = torch.zeros_like(s)
        s += x.sum(dim=(0, 2, 3))
        s2 += x.square().sum(dim=(0, 2, 3))
        cnt += x.shape[0] * x.shape[2] * x.shape[3]
    mean = s / cnt
    var = (s2 / cnt - mean.square()).clamp_min(0)
    return mean.float(), var.sqrt().float(), (s2 / cnt).float()


def run_val(codec, r2, store, bank, idx_all, a, mu, device, orig_nll_cache):
    codec.eval()
    codec.update(force=True)
    tot = {'n': 0, 'est_bits': 0.0, 'actual_bytes': 0, 'feat': 0.0, 'nll': 0.0, 'wnll': 0.0, 'orig_nll': 0.0}
    with torch.no_grad():
        for i in range(0, len(idx_all), a.batch_size):
            idx = idx_all[i:i + a.batch_size]
            x = store.batch(idx, device)
            B = x.shape[0]
            xf = x.flatten(0, 1)
            out = codec(xf, mode='round')
            fh = out['x_hat'].view_as(x)
            t, m, w = bank.batch(idx, device)
            lp = r2.teacher_forced(*tc.views(fh), t)
            # per-study means, then averaged over studies
            for b in range(B):
                tot['nll'] += tc.weighted_nll(lp[b:b + 1], t[b:b + 1], m[b:b + 1]).item()
                tot['wnll'] += tc.weighted_nll(lp[b:b + 1], t[b:b + 1], m[b:b + 1], w[b:b + 1]).item()
                key = idx[b]
                if key not in orig_nll_cache:
                    lpo = r2.teacher_forced(*tc.views(x[b:b + 1]), t[b:b + 1])
                    orig_nll_cache[key] = tc.weighted_nll(lpo, t[b:b + 1], m[b:b + 1]).item()
                tot['orig_nll'] += orig_nll_cache[key]
            tot['feat'] += tc.feature_nmse(out['x_hat'], xf, a._energy).item() * B
            tot['est_bits'] += out['bits'].sum().item()
            strings, hw = codec.compress(xf)
            for b in range(B):
                tot['actual_bytes'] += len(fcodec.pack_study(strings[x.shape[1] * b:x.shape[1] * (b + 1)], hw))
            tot['n'] += B
    n = tot['n']
    r = {'studies': n, 'est_bits_per_study': tot['est_bits'] / n,
         'actual_bits_per_study': 8.0 * tot['actual_bytes'] / n,
         'actual_bytes_per_study': tot['actual_bytes'] / n,
         'feature_nmse': tot['feat'] / n, 'text_nll': tot['nll'] / n, 'text_nll_weighted': tot['wnll'] / n,
         'text_nll_uncompressed': tot['orig_nll'] / n}
    r['objective'] = mu * r['actual_bits_per_study'] + a.lambda_f * r['feature_nmse'] + \
        a.lambda_t * r['text_nll_weighted']
    return r


def main():
    a, args = parse()
    device = torch.device(a.device)
    out = Path(a.out_dir)
    if (out / 'best.pt').exists():
        raise FileExistsError(f'{out} already has a checkpoint; choose a new --out_dir')
    out.mkdir(parents=True, exist_ok=True)
    tc.fix_seed(a.seed)

    r2, tokenizer = tc.load_frozen_r2gen(args, device)
    ann = tc.load_annotation(args)
    train, val = tc.FeatureStore(a.feature_dir, 'train'), tc.FeatureStore(a.feature_dir, 'val')
    r2_sha = tc.sha256_file(args.load)
    if train.manifest.get('r2gen_checkpoint_sha256') not in (None, r2_sha):
        raise ValueError('Cached features were extracted with a different R2Gen checkpoint.')
    tb = tc.TargetBank(ann, 'train', tokenizer, args.max_seq_length, a.alpha, train.ids)
    vb = tc.TargetBank(ann, 'val', tokenizer, args.max_seq_length, a.alpha, val.ids)
    n_train = len(train) if not a.max_train else min(a.max_train, len(train))
    n_val = len(val) if not a.max_val else min(a.max_val, len(val))
    val_idx = list(range(n_val))

    if a.init:
        codec, init_ckpt = fcodec.load_codec(a.init, 'cpu')
        codec_cfg = codec.config
        energy = init_ckpt.get('channel_energy')
        energy = torch.as_tensor(energy) if energy is not None else None
    else:
        codec = fcodec.FeatureHyperpriorCodec(c_y=a.c_y, hidden=a.hidden)
        mean, std, energy = channel_stats(train, n_train)  # TRAIN split only
        codec.set_normalization(mean, std)
        codec_cfg, init_ckpt = codec.config, None
    codec = codec.to(device)
    a._energy = energy.to(device) if (energy is not None and a.feature_norm == 'channel') else None

    main_p, aux_p = codec.trainable_groups()
    opt = torch.optim.Adam(main_p, lr=a.lr)
    aux_opt = torch.optim.Adam(aux_p, lr=a.aux_lr)
    steps_per_epoch = math.ceil(n_train / a.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps_per_epoch, eta_min=a.lr * 0.1)
    if a.rate_mode == 'fixed':
        log_mu = math.log(a.lambda_rate)
    else:
        log_mu = math.log(init_ckpt['mu']) if (init_ckpt and init_ckpt.get('mu') and
                                                init_ckpt.get('train_args', {}).get('target_bits') == a.target_bits) \
            else math.log(a.mu_init)

    config = {'train_args': {k: v for k, v in vars(a).items() if not k.startswith('_')},
              'r2gen_args': vars(args), 'r2gen_checkpoint_sha256': r2_sha, 'codec_config': codec_cfg,
              'feature_manifest': train.manifest, 'train_weight_audit': tb.audit(),
              'n_train': n_train, 'n_val': n_val, 'torch_version': torch.__version__,
              'loss': 'mu*R_bits_per_study + lambda_f*NMSE + lambda_t*sum(w*m*nll)/sum(w*m)',
              'frozen': 'ResNet + all R2Gen params (requires_grad=False, eval mode)'}
    import compressai
    config['compressai_version'] = compressai.__version__
    tc.json_save(out / 'config.json', config)
    print('train weight audit:', config['train_weight_audit'])

    orig_cache, history, best = {}, [], None
    v0 = run_val(codec, r2, val, vb, val_idx, a, math.exp(log_mu), device, orig_cache)
    # container + rANS flush overhead (actual file bits - estimated bits), refreshed at every validation,
    # so that the controller targets ACTUAL coded bits per study
    overhead = v0['actual_bits_per_study'] - v0['est_bits_per_study']
    print('VAL init:', {k: round(v, 5) if isinstance(v, float) else v for k, v in v0.items()})
    history.append({'epoch': 0, 'val': v0})
    rng = np.random.default_rng(a.seed)
    step = 0
    for epoch in range(1, a.epochs + 1):
        codec.train()
        perm = rng.permutation(n_train)
        t0, agg = time.time(), {'loss': 0.0, 'bits': 0.0, 'bits_round': 0.0, 'feat': 0.0, 'wnll': 0.0, 'k': 0}
        for i in range(0, n_train, a.batch_size):
            idx = perm[i:i + a.batch_size].tolist()
            x = train.batch(idx, device)
            xf = x.flatten(0, 1)
            out_c = codec(xf, mode=a.quant)
            bits = out_c['bits'].view(x.shape[0], x.shape[1]).sum(1).mean()
            d_f = tc.feature_nmse(out_c['x_hat'], xf, a._energy)
            mu = math.exp(log_mu)
            loss = mu * bits + a.lambda_f * d_f
            wnll = torch.zeros((), device=device)
            if a.lambda_t > 0:
                fh = out_c['x_hat'].view_as(x)
                t, m, w = tb.batch(idx, device)
                lp = r2.teacher_forced(*tc.views(fh), t)
                wnll = tc.weighted_nll(lp, t, m, w)
                loss = loss + a.lambda_t * wnll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                if any(p.grad is not None for p in r2.model.parameters()):
                    raise RuntimeError('Frozen R2Gen received gradients.')
                for g in ('g_a.', 'g_s.'):
                    gs = [p.grad for n, p in codec.named_parameters() if n.startswith(g) and p.grad is not None]
                    if not gs or not any(torch.count_nonzero(x_) for x_ in gs):
                        raise RuntimeError(f'No gradient reaches codec {g}')
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(main_p, a.clip)
            opt.step()
            sched.step()
            aux_opt.zero_grad(set_to_none=True)
            aux = codec.aux_loss()
            aux.backward()
            aux_opt.step()
            # rate control on the deployed (rounded) rate
            with torch.no_grad():
                bits_round = codec(xf, mode='round')['bits'].view(x.shape[0], x.shape[1]).sum(1).mean().item()
            if a.rate_mode == 'target':
                log_mu += a.mu_lr * math.log(max(bits_round + overhead, 1.0) / a.target_bits)
                log_mu = min(max(log_mu, math.log(1e-9)), math.log(1e3))
            step += 1
            k = len(idx)
            agg['loss'] += loss.item() * k
            agg['bits'] += bits.item() * k
            agg['bits_round'] += bits_round * k
            agg['feat'] += d_f.item() * k
            agg['wnll'] += wnll.item() * k
            agg['k'] += k
        tr = {kk: vv / agg['k'] for kk, vv in agg.items() if kk != 'k'}
        rec = {'epoch': epoch, 'mu': math.exp(log_mu), 'train': tr, 'sec': time.time() - t0}
        print(f"ep {epoch:3d} loss={tr['loss']:.4f} bits(noise)={tr['bits']:.0f} bits(round)={tr['bits_round']:.0f} "
              f"nmse={tr['feat']:.4f} wnll={tr['wnll']:.4f} mu={math.exp(log_mu):.3e} ({rec['sec']:.0f}s)", flush=True)
        if epoch % a.val_every == 0 or epoch == a.epochs:
            v = run_val(codec, r2, val, vb, val_idx, a, math.exp(log_mu), device, orig_cache)
            rec['val'] = v
            overhead = v['actual_bits_per_study'] - v['est_bits_per_study']
            print('   VAL', {k: round(x_, 4) if isinstance(x_, float) else x_ for k, x_ in v.items()}, flush=True)
            # selection on VALIDATION only; in target mode require the rate to be within 10% of target
            ok = a.rate_mode != 'target' or abs(v['actual_bits_per_study'] / a.target_bits - 1) <= a.rate_tol
            score = v['objective'] if a.rate_mode == 'fixed' else \
                a.lambda_f * v['feature_nmse'] + a.lambda_t * v['text_nll_weighted']
            if ok and (best is None or score < best[0]):
                best = (score, epoch)
                save(out / 'best.pt', codec, config, log_mu, epoch, v, energy)
        history.append(rec)
        tc.json_save(out / 'history.json', history)
    save(out / 'last.pt', codec, config, log_mu, a.epochs, history[-1].get('val'), energy)
    if best is None:
        print('WARNING: no epoch met the rate tolerance; best.pt := last.pt. Train longer or adjust --mu_lr.')
        save(out / 'best.pt', codec, config, log_mu, a.epochs, history[-1].get('val'), energy)
    print('best epoch:', best[1] if best else None, '->', (out / 'best.pt').resolve())


def save(path, codec, config, log_mu, epoch, val, energy):
    torch.save({'codec_config': codec.config, 'codec_state': fcodec.codec_state_dict(codec),
                'train_args': config['train_args'], 'config': config, 'mu': math.exp(log_mu), 'epoch': epoch,
                'val': val, 'channel_energy': energy}, path)


if __name__ == '__main__':
    main()
