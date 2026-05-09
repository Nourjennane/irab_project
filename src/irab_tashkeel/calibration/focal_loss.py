"""Focal cross-entropy + confidence penalty.

Two training-time alternatives to vanilla cross-entropy when the
goal is to *reduce overconfidence* on hard examples:

  - ``focal_loss``: down-weights easy examples (high p_correct), so
    the gradient focuses on hard ones. Reduces overconfidence on
    common-class wins.
  - ``confidence_penalty``: adds a small penalty whenever the model
    places too much mass on the top class, regardless of correctness.
    Equivalent to maximum-entropy regularization restricted to the
    top class.

Drop-in: replace ``F.cross_entropy`` with ``focal_loss`` in any head's
loss calculation. Both functions handle ignore_index=-100.
"""
from __future__ import annotations

from typing import Optional


def focal_loss(
    logits: "torch.Tensor", labels: "torch.Tensor",
    *,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> "torch.Tensor":
    """Multi-class focal loss.

    L_FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    import torch
    import torch.nn.functional as F

    log_probs = F.log_softmax(logits, dim=-1)
    n_classes = logits.size(-1)

    valid = labels != ignore_index
    if valid.sum().item() == 0:
        return logits.sum() * 0.0
    log_v = log_probs[valid]                # (N, C)
    lab_v = labels[valid]                   # (N,)

    # NLL on gold class
    nll = -log_v.gather(-1, lab_v.unsqueeze(-1)).squeeze(-1)
    p_t = torch.exp(-nll).clamp(min=1e-8, max=1 - 1e-8)
    focal_w = (1.0 - p_t).pow(gamma)
    loss_focal = (focal_w * nll).mean()

    if label_smoothing > 0:
        smooth = -log_v.mean(dim=-1).mean()
        loss_focal = (1 - label_smoothing) * loss_focal + label_smoothing * smooth
    return loss_focal


def confidence_penalty(
    logits: "torch.Tensor", *, beta: float = 0.1,
) -> "torch.Tensor":
    """Penalise the negative entropy of the softmax — i.e. reward
    higher entropy. Returns ``beta * (-entropy_mean)``; subtract from
    main loss."""
    import torch
    import torch.nn.functional as F
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    entropy = -(p * log_p).sum(dim=-1)              # higher = less peaky
    return -beta * entropy.mean()                    # negative = penalty added
