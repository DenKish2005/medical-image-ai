"""Feature-native scale/mean hyperprior codec for R2Gen ResNet-101 layer4 features.

Implements the codec in the technical note, Section 3:
    g_a : 3x3 conv 2048->512 + GELU ; 3x3 conv 512->C_y          (stride 1, keeps 7x7 grid)
    h_a : 3x3 conv C_y->C_h stride 2 + GELU ; 3x3 conv C_h->C_z stride 2   (C_h = max(C_y/2, 32))
    h_s : two x2 transposed-conv blocks, crop to HxW, output (mu_y, sigma_y>0)
    g_s : 3x3 conv C_y->512 + GELU ; 1x1 conv 512->2048
    entropy: factorized p(z) (EntropyBottleneck) + conditional Gaussian p(y|z) (GaussianConditional)

Training uses additive uniform noise (mode='noise', as specified in the note) or the
common "STE for synthesis, noise for rate" variant (mode='ste').  Evaluation uses
mode='round', which is numerically identical to compress()->bytes->decompress().

The codec input is the raw (un-normalized) layer4 map F in R^{B x 2048 x H x W}; a fixed
per-channel affine normalization estimated on the TRAINING split only is stored as buffers.
"""
import io
import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

SCALES_MIN, SCALES_MAX, SCALES_LEVELS = 0.11, 256.0, 64
BITSTREAM_VERSION = 1


def get_scale_table(lo=SCALES_MIN, hi=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), levels))


def ste_round(x):
    return (torch.round(x) - x).detach() + x


def _z_hw(h, w):
    # two stride-2 convs with k=3, p=1: n -> ceil(n/2) -> ceil(ceil(n/2)/2)
    return ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2


class FeatureHyperpriorCodec(nn.Module):
    def __init__(self, in_channels=2048, hidden=512, c_y=128, c_z=None):
        super().__init__()
        c_h = max(c_y // 2, 32)
        c_z = c_z or c_h
        self.config = {'in_channels': in_channels, 'hidden': hidden, 'c_y': c_y, 'c_z': c_z}
        self.c_y, self.c_z = c_y, c_z
        self.register_buffer('norm_mean', torch.zeros(1, in_channels, 1, 1))
        self.register_buffer('norm_std', torch.ones(1, in_channels, 1, 1))
        self.g_a = nn.Sequential(nn.Conv2d(in_channels, hidden, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(hidden, c_y, 3, 1, 1))
        self.h_a = nn.Sequential(nn.Conv2d(c_y, c_h, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(c_h, c_z, 3, 2, 1))
        self.h_s = nn.Sequential(nn.ConvTranspose2d(c_z, c_h, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(c_h, 2 * c_y, 4, 2, 1))
        self.g_s = nn.Sequential(nn.Conv2d(c_y, hidden, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(hidden, in_channels, 1))
        self.entropy_bottleneck = EntropyBottleneck(c_z)
        self.gaussian_conditional = GaussianConditional(None)

    # ------------------------------------------------------------------ utils
    @torch.no_grad()
    def set_normalization(self, mean, std, eps=1e-6):
        self.norm_mean.copy_(mean.reshape(self.norm_mean.shape))
        self.norm_std.copy_(std.reshape(self.norm_std.shape).clamp_min(eps))

    def trainable_groups(self):
        """Main (g_a, g_s, h_a, h_s, entropy densities) vs. aux (EntropyBottleneck quantiles)."""
        main, aux = [], []
        for n, p in self.named_parameters():
            (aux if n.endswith('.quantiles') else main).append(p)
        return main, aux

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    def update(self, force=True):
        self.gaussian_conditional.update_scale_table(get_scale_table(), force=force)
        self.entropy_bottleneck.update(force=force)

    def _hyper_params(self, z_hat, hw):
        p = self.h_s(z_hat)[..., :hw[0], :hw[1]]
        if p.shape[-2:] != torch.Size(hw):
            raise ValueError(f'h_s output {tuple(p.shape[-2:])} smaller than latent grid {hw}')
        means, scales = p.chunk(2, dim=1)
        return means, F.softplus(scales)

    # ---------------------------------------------------------------- forward
    def forward(self, x, mode='noise'):
        """Returns dict(x_hat, likelihoods{y,z}, bits[B]).  mode in {'noise','ste','round'}."""
        if mode not in ('noise', 'ste', 'round'):
            raise ValueError(mode)
        y = self.g_a((x - self.norm_mean) / self.norm_std)
        z = self.h_a(y)
        if mode == 'round':
            z_hat, z_lik = self.entropy_bottleneck(z, training=False)
        else:
            z_noisy, z_lik = self.entropy_bottleneck(z, training=True)
            if mode == 'noise':
                z_hat = z_noisy
            else:
                med = self.entropy_bottleneck._get_medians()
                z_hat = ste_round(z - med) + med
        means, scales = self._hyper_params(z_hat, y.shape[-2:])
        if mode == 'round':
            y_hat, y_lik = self.gaussian_conditional(y, scales, means=means, training=False)
        else:
            y_noisy, y_lik = self.gaussian_conditional(y, scales, means=means, training=True)
            y_hat = y_noisy if mode == 'noise' else ste_round(y - means) + means
        x_hat = self.g_s(y_hat) * self.norm_std + self.norm_mean
        bits = (-torch.log2(y_lik)).flatten(1).sum(1) + (-torch.log2(z_lik)).flatten(1).sum(1)
        return {'x_hat': x_hat, 'likelihoods': {'y': y_lik, 'z': z_lik}, 'bits': bits}

    # ------------------------------------------------------- real entropy coding
    @torch.no_grad()
    def compress(self, x):
        """x: [B, C, H, W] -> list (len B) of (y_string, z_string), plus (H, W)."""
        y = self.g_a((x - self.norm_mean) / self.norm_std)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        means, scales = self._hyper_params(z_hat, y.shape[-2:])
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means)
        return list(zip(y_strings, z_strings)), tuple(y.shape[-2:])

    @torch.no_grad()
    def decompress(self, strings, hw):
        """strings: list of (y_string, z_string) for views sharing grid hw -> x_hat [B, C, H, W]."""
        y_strings = [s[0] for s in strings]
        z_strings = [s[1] for s in strings]
        z_hat = self.entropy_bottleneck.decompress(z_strings, _z_hw(*hw))
        means, scales = self._hyper_params(z_hat, hw)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means)
        return self.g_s(y_hat) * self.norm_std + self.norm_mean


# ----------------------------------------------------------- bitstream container
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(buf):
    shift = result = 0
    while True:
        b = buf.read(1)[0]
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result
        shift += 7


def pack_study(view_strings, hw):
    """Compact per-study container: version, n_views, H, W, then varint-length-prefixed strings.
    Every byte returned here is counted as the coded rate."""
    if not (0 < hw[0] < 256 and 0 < hw[1] < 256):
        raise ValueError(hw)
    out = bytearray(struct.pack('BBBB', BITSTREAM_VERSION, len(view_strings), hw[0], hw[1]))
    for y_s, z_s in view_strings:
        out += _varint(len(y_s)) + y_s + _varint(len(z_s)) + z_s
    return bytes(out)


def unpack_study(blob):
    buf = io.BytesIO(blob)
    version, n_views, h, w = struct.unpack('BBBB', buf.read(4))
    if version != BITSTREAM_VERSION:
        raise ValueError(f'Unsupported bitstream version {version}')
    views = []
    for _ in range(n_views):
        y_s = buf.read(_read_varint(buf))
        z_s = buf.read(_read_varint(buf))
        views.append((y_s, z_s))
    if buf.read(1):
        raise ValueError('Trailing bytes in bitstream')
    return views, (h, w)


# --------------------------------------------------------------- checkpoint I/O
def codec_state_dict(codec):
    codec.update(force=True)  # freeze CDF tables that are saved with the weights
    return codec.state_dict()


def load_codec(path_or_ckpt, device='cpu'):
    ckpt = torch.load(path_or_ckpt, map_location='cpu', weights_only=False) \
        if isinstance(path_or_ckpt, (str, bytes)) or hasattr(path_or_ckpt, '__fspath__') else path_or_ckpt
    codec = FeatureHyperpriorCodec(**ckpt['codec_config'])
    state = ckpt['codec_state']
    # Entropy-model CDF buffers have data-dependent sizes: resize before loading so the
    # exact saved tables (not recomputed ones) are used for coding.
    buffers = dict(codec.named_buffers())
    for name, value in state.items():
        if name in buffers and buffers[name].shape != value.shape:
            buffers[name].resize_(value.shape)
    codec.load_state_dict(state, strict=True)
    return codec.to(device).eval(), ckpt
