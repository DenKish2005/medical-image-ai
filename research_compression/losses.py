"""Multilabel diagnostic supervision and teacher/ROI preservation objectives."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_weighted_bce(logits, targets, valid_mask, positive_weights=None):
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("BCE tensors must share [B,num_diseases] shape")
    if not valid_mask.bool().any():
        return logits.sum()*0
    kwargs = {} if positive_weights is None else {"pos_weight": positive_weights}
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", **kwargs)
    valid = valid_mask.to(losses.dtype)
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def bernoulli_distillation(student_logits, teacher_logits, temperature: float = 2.0):
    """Correct KL for independent multilabel Bernoulli outputs (NOT softmax KL)."""
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Bad distillation temperature/shape")
    p = torch.sigmoid(teacher_logits.detach()/temperature).clamp(1e-6, 1-1e-6)
    log_q = F.logsigmoid(student_logits/temperature)
    log_not_q = F.logsigmoid(-student_logits/temperature)
    kl = p*(p.log()-log_q) + (1-p)*(torch.log1p(-p)-log_not_q)
    return kl.mean()*(temperature**2)


def roi_consistency(feature, reconstructed, roi_mask):
    """ROI-weighted feature MSE; boxes used ONLY during training."""
    if roi_mask is None:
        return feature.sum()*0
    if feature.shape != reconstructed.shape or roi_mask.shape != feature.shape[:1]+feature.shape[2:]:
        raise ValueError("ROI mask must have [B,H,W] for features [B,C,H,W]")
    valid = roi_mask[:, None].to(dtype=feature.dtype)
    return (((feature-reconstructed)**2)*valid).sum() / (valid.sum()*feature.shape[1]).clamp_min(1)


def compression_loss(student_logits, teacher_logits, targets, valid_mask, feature,
                     reconstructed, estimated_bits, *, rate_target_bits: float,
                     positive_weights=None, roi_mask=None, classification_weight=1.0,
                     reconstruction_weight=0.05, distill_weight=0.5, lesion_weight=0.0,
                     rate_weight=1e-6, over_budget_weight=1e-5):
    classification = masked_weighted_bce(student_logits, targets, valid_mask, positive_weights)
    distill = bernoulli_distillation(student_logits, teacher_logits)
    reconstruction = F.mse_loss(reconstructed, feature.detach())
    lesion = roi_consistency(feature.detach(), reconstructed, roi_mask)
    # Estimated (differentiable) rate is a surrogate, not achieved stream size.
    rate = estimated_bits.float().mean()
    constraint = F.relu(estimated_bits.float() - rate_target_bits).mean()
    total = (classification_weight*classification + reconstruction_weight*reconstruction +
             distill_weight*distill + lesion_weight*lesion + rate_weight*rate +
             over_budget_weight*constraint)
    return total, {"bce":classification.detach(), "distill":distill.detach(),
                   "mse":reconstruction.detach(), "roi":lesion.detach(),
                   "estimated_bits":rate.detach(), "constraint":constraint.detach()}
