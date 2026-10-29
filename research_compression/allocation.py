"""Explicit constrained spatial bit allocation; lesion annotations NEVER enter inference."""
from __future__ import annotations

import heapq
import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialRiskHeads(nn.Module):
    """Small feature-only predictors of local importance and uncertainty proxies."""
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.importance = nn.Conv2d(hidden, 1, 1)
        self.uncertainty = nn.Conv2d(hidden, 1, 1)

    def forward(self, feature):
        z = self.shared(feature)
        return torch.sigmoid(self.importance(z)[:, 0]), torch.sigmoid(self.uncertainty(z)[:, 0])


def _normalize(values: torch.Tensor):
    v = values.flatten(1)
    lo, hi = v.amin(dim=1), v.amax(dim=1)
    return ((values - lo[:, None, None]) /
            (hi - lo).clamp_min(1e-8)[:, None, None])


def teacher_spatial_targets(feature: torch.Tensor, downstream: nn.Module):
    """Gradient-sensitivity and predictive-entropy-sensitivity teacher maps.

    Note: entropy gradient is NOT calibrated epistemic uncertainty. For proper
    uncertainty studies replace with validated MC-dropout/ensemble targets.
    """
    with torch.enable_grad():
        feature = feature.detach().requires_grad_(True)
        logits = downstream(feature)
        importance_grad = torch.autograd.grad(logits.sigmoid().sum(), feature, retain_graph=True)[0]
        p = logits.sigmoid().clamp(1e-6, 1-1e-6)
        entropy = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean(dim=1).sum()
        uncertainty_grad = torch.autograd.grad(entropy, feature)[0]
        importance = (importance_grad*feature).abs().mean(dim=1)
        uncertainty = uncertainty_grad.abs().mean(dim=1)
    return _normalize(importance).detach(), _normalize(uncertainty).detach()


def risk_head_loss(pred_importance, pred_uncertainty, target_importance, target_uncertainty):
    return F.mse_loss(pred_importance, target_importance) + F.mse_loss(pred_uncertainty, target_uncertainty)


def allocate_bits(importance: torch.Tensor, uncertainty: torch.Tensor, *, channels: int,
                  target_bits: int, mode: str = "combined", alpha: float = 0.5,
                  reserve_bytes: int = 192, seed: int = 0) -> torch.Tensor:
    """Greedy 2→4→8 bits/element, conservative fixed reserve for metadata.

    Computes estimated *packed-symbol* budget only; compressor may exceed the
    target after real serialization. Always check achieved len(message)*8.
    When the minimum 2-bit map is unaffordable, returns it without claiming
    feasibility. Allocation cost: all C latent channels at each spatial cell.
    """
    if importance.shape != uncertainty.shape or importance.ndim != 3:
        raise ValueError("Importance and uncertainty must have shape [B,H,W]")
    if channels < 1 or target_bits < 1 or reserve_bytes < 0:
        raise ValueError("Invalid allocation budget or channels")
    if mode not in {"uniform", "random", "saliency", "uncertainty", "combined"}:
        raise ValueError("Unknown allocator mode")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    b, h, w = importance.shape
    imp = _normalize(importance.detach().float()).cpu()
    unc = _normalize(uncertainty.detach().float()).cpu()
    if mode == "saliency":
        score = imp
    elif mode == "uncertainty":
        score = unc
    elif mode == "combined":
        score = alpha*imp + (1-alpha)*unc
    elif mode == "random":
        generator = torch.Generator().manual_seed(seed)
        score = torch.rand(imp.shape, generator=generator)
    else:
        score = torch.zeros_like(imp)

    out = torch.full((b, h, w), 2, dtype=torch.uint8)
    available = max(0, target_bits - reserve_bytes*8 - (2*h*w + 7)//8 * 8)
    base = 2 * channels * h * w
    for image_index in range(b):
        extra = max(0, available - base)
        if mode == "uniform":
            for target_depth in (4, 8):
                cost = (target_depth - int(out[image_index, 0, 0]))*channels*h*w
                if cost <= extra:
                    out[image_index].fill_(target_depth)
                    extra -= cost
            continue
        priorities = score[image_index].reshape(-1).tolist()
        depths = [2] * (h*w)
        # Each additional 2 bits initially yields high utility; 4→8 yields
        # smaller marginal utility per bit. Heap prioritizes those utility/costs.
        heap = [(-float(priority)/max(2*channels,1), idx, 2) for idx,priority in enumerate(priorities)]
        heapq.heapify(heap)
        while heap and extra >= 2*channels:
            neg_priority, idx, old = heapq.heappop(heap)
            if depths[idx] != old:
                continue
            next_depth = 4 if old == 2 else 8
            cost = (next_depth-old)*channels
            if extra < cost:
                continue
            depths[idx] = next_depth
            extra -= cost
            if next_depth == 4:
                # Diminishing utility for additional 4 bits.
                heapq.heappush(heap, (-priorities[idx]/(8*channels), idx, 4))
        out[image_index] = torch.tensor(depths, dtype=torch.uint8).reshape(h,w)
    return out.to(importance.device)


def actual_rate(message: bytes, target_bits: int):
    bits = 8*len(message)
    return {"actual_bytes": len(message), "actual_bits": bits,
            "target_bits": int(target_bits), "budget_met": bits <= target_bits}


def allocation_map_bits(height: int, width: int) -> int:
    return 8*math.ceil(height*width*2/8)
