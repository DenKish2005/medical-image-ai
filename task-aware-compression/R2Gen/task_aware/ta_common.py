"""Shared helpers: frozen R2Gen loading, cached-feature store, targets/weights, losses."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R2GEN_DIR = HERE.parent
if str(R2GEN_DIR) not in sys.path:
    sys.path.insert(0, str(R2GEN_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import clinical_spans as cs  # noqa: E402

if not torch.cuda.is_available():
    # R2Gen's beam_search hard-codes .cuda(); make it a no-op so CPU smoke tests can run.
    torch.Tensor.cuda = lambda self, *a, **k: self


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def json_save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True), encoding='utf-8')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_argv(argv=None):
    """Script args before '--', R2Gen (main_test.parse_agrs) args after '--'."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        i = argv.index('--')
        return argv[:i], argv[i + 1:]
    return argv, []


def r2gen_args(r2gen_argv):
    from main_test import parse_agrs
    old = sys.argv
    try:
        sys.argv = [old[0]] + list(r2gen_argv)
        args = parse_agrs()
    finally:
        sys.argv = old
    args.visual_extractor_pretrained = False  # weights come from the R2Gen checkpoint
    if not args.load:
        raise ValueError('Pass the pretrained R2Gen checkpoint with --load (after "--").')
    return args


def load_frozen_r2gen(args, device):
    """Load the pretrained R2Gen, put it in eval mode and freeze every parameter."""
    from models.r2gen import R2GenModel
    from modules.tokenizers import Tokenizer
    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer)
    ckpt = torch.load(args.load, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    if all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenR2Gen(model, tokenizer, args), tokenizer


class FrozenR2Gen:
    """Runs R2Gen's encoder-decoder from layer4 maps, exactly as forward_iu_xray does after the CNN.

    Gradient tracking is deliberately NOT disabled in teacher_forced(): gradients flow through the
    frozen ops into F_hat (and the codec) while R2Gen parameters have requires_grad=False.
    """

    def __init__(self, model, tokenizer, args):
        self.model, self.tokenizer, self.args = model, tokenizer, args

    @staticmethod
    def to_r2gen_inputs(f0, f1=None):
        fs = (f0,) if f1 is None else (f0, f1)       # one view (mimic_cxr) or two (iu_xray)
        att = torch.cat([f.flatten(2).transpose(1, 2) for f in fs], dim=1)  # [B, V*H*W, 2048]
        fc = torch.cat([f.mean(dim=(2, 3)) for f in fs], dim=1)            # == AvgPool2d(7) for 7x7
        return fc, att

    def cnn_features(self, images):
        if images.dim() == 4:                        # mimic_cxr: [B, 3, H, W] -> one view
            images = images.unsqueeze(1)
        with torch.no_grad():
            return [self.model.visual_extractor.model(images[:, i]) for i in range(images.shape[1])]

    def teacher_forced(self, f0, f1, targets):
        fc, att = self.to_r2gen_inputs(f0, f1)
        return self.model.encoder_decoder(fc, att, targets, mode='forward')  # log-probs [B, T-1, V]

    @torch.no_grad()
    def generate(self, f0, f1):
        fc, att = self.to_r2gen_inputs(f0, f1)
        seq, _ = self.model.encoder_decoder(fc, att, mode='sample')
        return self.tokenizer.decode_batch(seq.cpu().numpy())


def views(x):
    """[B, V, C, H, W] -> (view0, view1) for two-view studies, (view0, None) for single images."""
    return x[:, 0], (x[:, 1] if x.shape[1] > 1 else None)


def weighted_nll(logprobs, targets, masks, weights=None):
    """sum(mask*w*nll)/sum(mask*w) with R2Gen's shift/truncation (modules/loss.py)."""
    T = logprobs.shape[1]
    tgt = targets[:, 1:][:, :T]
    m = masks[:, 1:][:, :T].to(logprobs.dtype)
    if weights is not None:
        m = m * weights[:, :T].to(logprobs.dtype)
    nll = -logprobs.gather(2, tgt.long().unsqueeze(2)).squeeze(2)
    return (nll * m).sum() / m.sum().clamp_min(1e-8)


class FeatureStore:
    """Cached frozen-ResNet layer4 maps: <dir>/<split>_features.npy [N, 2, 2048, H, W] + <split>_ids.json."""

    def __init__(self, feature_dir, split):
        self.dir = Path(feature_dir)
        self.split = split
        self.features = np.load(self.dir / f'{split}_features.npy', mmap_mode='r')
        self.ids = json.loads((self.dir / f'{split}_ids.json').read_text())
        if len(self.ids) != self.features.shape[0]:
            raise ValueError('ids/features length mismatch')
        self.manifest = json.loads((self.dir / 'manifest.json').read_text())

    def __len__(self):
        return len(self.ids)

    def batch(self, idx, device):
        x = torch.from_numpy(np.array(self.features[np.asarray(idx)]))
        return x.to(device=device, dtype=torch.float32)


class TargetBank:
    """Tokenized references + clinical span weights for one split (training supervision only)."""

    def __init__(self, ann, split, tokenizer, max_seq_length, alpha, ids):
        by_id = {e['id']: e for e in ann[split]}
        self.ids, self.reports, self.tokens, self.weights, self.info = ids, [], [], [], []
        for sid in ids:
            report = by_id[sid]['report']
            words = tokenizer.clean_report(report).split()
            toks = tokenizer(report)[:max_seq_length]
            n_t = len(toks) - 1
            w, info = cs.token_weights(words, n_t, alpha)
            self.reports.append(tokenizer.clean_report(report))
            self.tokens.append(toks)
            self.weights.append(w)
            self.info.append(info)

    def batch(self, idx, device):
        L = max(len(self.tokens[i]) for i in idx)
        t = torch.zeros(len(idx), L, dtype=torch.long)
        m = torch.zeros(len(idx), L)
        w = torch.ones(len(idx), L - 1)
        for r, i in enumerate(idx):
            n = len(self.tokens[i])
            t[r, :n] = torch.tensor(self.tokens[i])
            m[r, :n] = 1
            w[r, :n - 1] = torch.tensor(self.weights[i])
        return t.to(device), m.to(device), w.to(device)

    def audit(self):
        n = sum(i['n_targets'] for i in self.info)
        up = sum(i['upweighted_tokens'] for i in self.info)
        return {'studies': len(self.info), 'target_tokens_incl_eos': n, 'upweighted_tokens': up,
                'upweighted_percent': 100.0 * up / max(n, 1),
                'mentions': sum(i['mentions'] for i in self.info),
                'truncated_spans_left_at_weight_1': sum(i['truncated_spans'] for i in self.info),
                'rules_version': cs.RULES_VERSION, 'rules_sha256': cs.rules_sha256()}


def load_annotation(args):
    return json.loads(Path(args.ann_path).read_text())


def feature_nmse(f_hat, f, per_channel_energy=None, eps=1e-8):
    """Note Sec. 4.2: mean((F_hat-F)^2 / (mean(F^2)+eps)); per-sample global energy, or fixed
    per-channel training-set energy E[F_c^2] when given."""
    err = (f_hat - f).square()
    if per_channel_energy is not None:
        return (err / (per_channel_energy.view(1, -1, 1, 1) + eps)).mean()
    energy = f.square().flatten(1).mean(1).view(-1, 1, 1, 1)
    return (err / (energy + eps)).mean()
