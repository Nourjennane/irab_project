"""Layer-wise learning-rate decay for fine-tuning.

When fine-tuning a pretrained encoder + freshly initialised heads,
applying the same lr to every layer often over-disturbs the encoder's
lower layers — which encode general linguistic features we want to
preserve — while under-training the heads.

The classic remedy: layer-wise LR decay. Lower encoder layers get
``base_lr * decay_rate ** (depth_below_top)``; the heads keep
``base_lr``. This usually yields 1-3 points of generalization on
small-corpus fine-tuning.

Returns a list of param_groups suitable for ``torch.optim.AdamW``::

    optimizer = torch.optim.AdamW(
        build_param_groups(model, base_lr=1e-5, decay=0.85),
        weight_decay=0.01,
    )
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


_ENCODER_PREFIXES = ("encoder.", "shared.", "model.encoder.")


def _layer_index(name: str) -> int:
    """Best-effort layer-index extraction for T5-style encoder block paths."""
    m = re.search(r"\.block\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    return -1


def build_param_groups(
    model, *,
    base_lr: float = 1e-5,
    decay: float = 0.85,
    min_lr_factor: float = 0.05,
) -> List[Dict[str, Any]]:
    """Construct param-group list with layer-wise LR decay.

    All parameters whose name starts with a head/non-encoder prefix
    (``case_head.``, ``role_head.``, ``marker_head.``, etc.) get the
    full ``base_lr``. Encoder block ``i`` (counting from the bottom)
    gets ``base_lr * decay ** (top_block_idx - i)``, floored at
    ``min_lr_factor * base_lr``.
    """
    # Discover the top encoder block index
    top = -1
    for n, _ in model.named_parameters():
        i = _layer_index(n)
        if i > top:
            top = i

    head_params: List = []
    encoder_groups: Dict[int, List] = {}
    other_params: List = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_encoder = any(n.startswith(pre) for pre in _ENCODER_PREFIXES)
        i = _layer_index(n) if is_encoder else -1
        if i >= 0:
            encoder_groups.setdefault(i, []).append((n, p))
        elif is_encoder:
            # Encoder embedding / final norm — give them the lowest LR
            encoder_groups.setdefault(0, []).append((n, p))
        elif _is_head_name(n):
            head_params.append((n, p))
        else:
            other_params.append((n, p))

    groups: List[Dict[str, Any]] = []
    for i, items in sorted(encoder_groups.items()):
        depth_below = max(0, top - i)
        scale = max(decay ** depth_below, min_lr_factor)
        groups.append({
            "params": [p for _, p in items],
            "lr": base_lr * scale,
            "name": f"encoder_block_{i}",
        })
    if head_params:
        groups.append({"params": [p for _, p in head_params],
                       "lr": base_lr, "name": "heads"})
    if other_params:
        groups.append({"params": [p for _, p in other_params],
                       "lr": base_lr, "name": "other"})
    return groups


_HEAD_PATTERNS = ("head", "_head", "classifier", "logit", "dep_proj",
                  "dep_feature_encoder", "graph_refiner", "case_hierarchy",
                  "marker_hierarchy")


def _is_head_name(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _HEAD_PATTERNS)
