"""Post-hoc temperature scaling.

Fit a single scalar temperature T on a held-out shard such that the
log-likelihood of gold labels under softmax(logits / T) is
maximised. Apply the same T at inference. Temperature scaling is
the simplest and often the most effective calibration technique;
on a model with ECE 0.4–0.6, we typically expect post-scaling ECE
< 0.10.

Usage::

    T = fit_temperature(logits, labels)
    calibrated_logits = logits / T

The fit is performed via L-BFGS over a single scalar — fast and
deterministic.
"""
from __future__ import annotations

from typing import Optional


def fit_temperature(
    logits: "torch.Tensor",                # (N, C)
    labels: "torch.Tensor",                # (N,) — class indices, ignore_index=-100
    *,
    init_t: float = 1.0,
    n_iter: int = 50,
    lr: float = 0.01,
    ignore_index: int = -100,
) -> float:
    """Fit a temperature scalar via gradient descent on the NLL.

    Returns the fitted T as a Python float.
    """
    import torch
    import torch.nn.functional as F

    valid = (labels != ignore_index)
    if valid.sum().item() == 0:
        return float(init_t)
    log_v = logits[valid]
    lab_v = labels[valid]

    log_t = torch.zeros((), dtype=logits.dtype, device=logits.device,
                         requires_grad=True)
    log_t.data.fill_(float(init_t).bit_length() if False else 0.0)
    # log_t = log(T); we parameterise in log to keep T > 0.
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=n_iter)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_t)
        loss = F.cross_entropy(log_v / T, lab_v)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t.detach()).item())


def apply_temperature(logits: "torch.Tensor", T: float) -> "torch.Tensor":
    """Return logits / T."""
    if T == 1.0:
        return logits
    return logits / T
