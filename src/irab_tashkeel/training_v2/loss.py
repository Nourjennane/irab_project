"""Multi-head weighted loss.

Computes cross-entropy per head, weights by stage-specific
:class:`HeadLossWeights`, sums to a single scalar. The morph
weight is split equally across the 7 morph axes (gender, number,
definite, person, aspect, mood, voice) so the *total* morph
contribution is normalised regardless of which axes are present
on the batch.

Args & returns
--------------

  ``logits``      : dict of head_name → tensor (B, W, n_class)
  ``labels``      : dict of head_name → tensor (B, W); IGNORE = -100
  ``weights``     : :class:`HeadLossWeights`

Returns:

  ``loss``        : scalar tensor
  ``per_head``    : dict head_name → scalar (un-weighted CE)
"""
from __future__ import annotations

from typing import Any, Dict

from .config import HeadLossWeights


MORPH_AXES = ("gender", "number", "definite", "person", "aspect", "mood", "voice")


def compute_multi_head_loss(
    logits: Dict[str, "torch.Tensor"],
    labels: Dict[str, "torch.Tensor"],
    weights: HeadLossWeights,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    per_head: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=next(iter(logits.values())).device)

    def _ce(name: str):
        if name not in logits or name not in labels:
            return None
        log = logits[name]
        lab = labels[name]
        return F.cross_entropy(
            log.reshape(-1, log.size(-1)), lab.reshape(-1),
            ignore_index=-100,
        )

    # Iʿrāb heads
    for head in ("case", "role", "marker", "pos"):
        l = _ce(head)
        if l is None:
            continue
        per_head[head] = l
        total = total + l * float(getattr(weights, head))

    # Morph axes (split the morph weight equally)
    morph_per_axis: Dict[str, torch.Tensor] = {}
    for axis in MORPH_AXES:
        l = _ce(f"morph_{axis}")
        if l is None:
            continue
        morph_per_axis[axis] = l
    if morph_per_axis:
        per_axis_weight = float(weights.morph) / max(len(morph_per_axis), 1)
        for l in morph_per_axis.values():
            total = total + l * per_axis_weight
        per_head["morph"] = sum(morph_per_axis.values()) / len(morph_per_axis)
        for axis, l in morph_per_axis.items():
            per_head[f"morph_{axis}"] = l

    return {"loss": total, "per_head": per_head}
