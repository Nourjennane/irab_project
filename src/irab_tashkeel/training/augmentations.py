"""Train-time augmentations.

Currently:

  - construction_dropout(edge_index, p) — randomly zero out
    construction-membership and overlap edges. Forces the model to
    rely on raw encoder signal occasionally instead of construction
    shortcuts. Only applied during training.

  - dep_dropout(edge_index, p) — same idea for dep edges.

  - morph_label_dropout(labels, p) — randomly mark a fraction of
    morph axis labels as IGNORE so the model doesn't memorize the
    morph taxonomy as a shortcut.
"""
from __future__ import annotations

from typing import Dict


# Edge-type ids matching models.graph_refiner.EDGE_TYPES
_CONSTRUCTION_EDGE_IDS = {3, 6}     # construction_member, overlap
_DEP_EDGE_IDS = {1}                  # dep
IGNORE = -100


def construction_dropout(edge_index: "torch.Tensor", p: float = 0.12) -> "torch.Tensor":
    """Randomly zero out construction-related edges with probability ``p``.

    Operates on a (B, W, W) int tensor of edge type ids; zeros out
    cells whose value is in ``_CONSTRUCTION_EDGE_IDS``. Not
    in-place — returns a new tensor.
    """
    import torch
    if p <= 0.0 or edge_index is None:
        return edge_index
    out = edge_index.clone()
    mask = torch.zeros_like(out, dtype=torch.bool)
    for tid in _CONSTRUCTION_EDGE_IDS:
        mask |= (out == tid)
    drop = (torch.rand_like(out, dtype=torch.float) < p) & mask
    out[drop] = 0
    return out


def dep_dropout(edge_index: "torch.Tensor", p: float = 0.05) -> "torch.Tensor":
    """Randomly zero out dep edges with probability ``p``."""
    import torch
    if p <= 0.0 or edge_index is None:
        return edge_index
    out = edge_index.clone()
    mask = torch.zeros_like(out, dtype=torch.bool)
    for tid in _DEP_EDGE_IDS:
        mask |= (out == tid)
    drop = (torch.rand_like(out, dtype=torch.float) < p) & mask
    out[drop] = 0
    return out


def morph_label_dropout(labels: Dict[str, "torch.Tensor"], p: float = 0.10) -> Dict[str, "torch.Tensor"]:
    """For each morph axis label tensor, randomly mark a fraction of valid
    positions as IGNORE so the model does not lean on morph as a shortcut.
    """
    import torch
    if p <= 0.0:
        return labels
    out = dict(labels)
    for k, v in labels.items():
        if not k.startswith("morph_"):
            continue
        keep = (torch.rand_like(v, dtype=torch.float) >= p)
        out[k] = torch.where(keep, v, torch.full_like(v, IGNORE))
    return out
