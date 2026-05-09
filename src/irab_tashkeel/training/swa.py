"""Stochastic Weight Averaging (SWA) helper.

Averages model weights over the *last K evaluation snapshots* of a
stage. The idea: at the end of training, the SWA-averaged weights
typically generalize better than the last single checkpoint, because
SGD bounces around the loss-landscape minimum and the average lands
closer to the true centre.

Usage in trainer::

    swa = SWASnapshot(model)
    ...
    if global_step % swa_interval == 0:
        swa.update(model)
    ...
    # at stage end:
    swa.copy_into(model)        # swap in averaged weights for eval/save
    metrics = run_eval(model)
    swa.restore(model)          # bring the live training weights back

This implementation is plain torch — no external optimizer wrapper,
no contrib package. It avoids ``torch.optim.swa_utils.AveragedModel``
because that fights with our custom forward signature.
"""
from __future__ import annotations

from typing import Optional


class SWASnapshot:
    """Running mean of model parameters with copy-in/restore semantics."""

    def __init__(self, model, max_snapshots: int = 8):
        self.max_snapshots = max_snapshots
        self.n = 0
        self._mean = {n: p.detach().clone()
                      for n, p in model.named_parameters()}
        self._live: Optional[dict] = None  # set when copy_into is active

    def update(self, model) -> None:
        """Fold the current model parameters into the running mean."""
        self.n += 1
        alpha = 1.0 / self.n
        for name, p in model.named_parameters():
            self._mean[name].mul_(1 - alpha).add_(p.detach(), alpha=alpha)

    def copy_into(self, model) -> None:
        """Swap the model's live parameters with the SWA mean. Cache live."""
        self._live = {n: p.detach().clone() for n, p in model.named_parameters()}
        with _no_grad():
            for n, p in model.named_parameters():
                p.copy_(self._mean[n])

    def restore(self, model) -> None:
        """Restore the live (pre-copy_into) parameters back into the model."""
        if self._live is None:
            return
        with _no_grad():
            for n, p in model.named_parameters():
                if n in self._live:
                    p.copy_(self._live[n])
        self._live = None


def _no_grad():
    import torch
    return torch.no_grad()
