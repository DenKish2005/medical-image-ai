"""Optional, non-invasive bridge for existing R2Gen/BLIP notebook model layers.

Usage: with FeatureReplacement(model, 'visual_encoder.layer4', codec_callback):
           output = model(...)
Callback takes one tensor and returns an identical-shape tensor. The caller
must guarantee feature compatibility. This is a notebook adaptation point,
NOT an automatic conversion of report-generation tasks to diagnosis labels.
"""
from __future__ import annotations

import torch


def module_names(model):
    return [name for name, _ in model.named_modules() if name]


class FeatureReplacement:
    def __init__(self, model, path, transform):
        self.model, self.path, self.transform = model, path, transform
        self.handle = None

    def __enter__(self):
        modules = dict(self.model.named_modules())
        if self.path not in modules:
            raise KeyError(f"Module '{self.path}' does not exist; inspect module_names(model)")

        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError("Feature hook requires a tensor with [B,C,H,W]")
            changed = self.transform(output)
            if not isinstance(changed, torch.Tensor) or changed.shape != output.shape:
                raise ValueError("Codec callback must preserve original feature shape")
            return changed

        self.handle = modules[self.path].register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


def make_replacement_callback(codec, scorer, *, split, rate_fraction=0.25,
                              mode='combined', rate_log=None):
    """Build an INFERENCE-ONLY hook compatible with arbitrary [B,C,H,W] backbones.

    The callback's exact split name and codec input channels must match the
    hooked module; train the codec on the same backbone and feature shape.
    `rate_log` can be an external list receiving measured-byte statistics.
    """
    from .allocation import allocate_bits, actual_rate
    from .codec import fp16_reference_bits

    if not 0 < rate_fraction <= 1:
        raise ValueError('rate_fraction must be in (0,1]')

    @torch.no_grad()
    def callback(feature):
        if feature.shape[1] != codec.encoder.in_channels:
            raise ValueError('Hook feature channels disagree with the codec checkpoint')
        imp,unc=scorer(feature)
        target=max(1,round(fp16_reference_bits(feature[0])*rate_fraction))
        depths=allocate_bits(imp,unc,channels=codec.latent_channels,target_bits=target,mode=mode)
        messages=codec.encode(feature,depths,rate_fraction,split=split,rate_id=str(rate_fraction))
        decoded=codec.decode(messages,expected_split=split,device=feature.device)
        if rate_log is not None:
            rate_log.extend(actual_rate(m,target) for m in messages)
        return decoded.to(dtype=feature.dtype)

    return callback
