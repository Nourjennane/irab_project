"""Multi-head weighted loss + recovery-patch auxiliary objectives.

Computes cross-entropy per head with **label smoothing** (item 6 A)
and adds:

  - entropy regularization (item 6 B)
  - structured-consistency penalty (item 9): case incompatible with
    role, or marker incompatible with case → penalty
  - exact-fully aux loss (item 10): sentence-level exactness reward

All extras are toggleable so we can ablate.

Args & returns
--------------

  ``logits``      : dict of head_name → tensor (B, W, n_class)
  ``labels``      : dict of head_name → tensor (B, W); IGNORE = -100
  ``weights``     : :class:`HeadLossWeights`
  ``label_smoothing``        — item 6 (A); default 0.0 = off
  ``entropy_reg_lambda``     — item 6 (B); default 0.0
  ``consistency_lambda``     — item 9; default 0.0
  ``fully_aux_lambda``       — item 10 inner weight; default 0.0
  ``token_mask``             — (B, W) of {0, 1} for fully-aux pooling

Returns:

  ``loss``        : scalar tensor
  ``per_head``    : dict head_name → scalar (un-weighted CE)
  ``aux``         : dict of named aux scalars (entropy / consistency / fully)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .config import HeadLossWeights


MORPH_AXES = ("gender", "number", "definite", "person", "aspect", "mood", "voice")


# ---------------------------------------------------------------------------
# Item 9 — structured consistency tables
# ---------------------------------------------------------------------------

# Pairs that should NOT co-occur. Each entry is (head_a_label, head_b_label).
# Computed against pred argmax: if both heads predict an incompatible pair
# we add a small penalty proportional to the joint probability.

INCOMPATIBLE_CASE_ROLE = {
    # raf-only roles
    ("raf",  "mafoul_bih"): True,
    ("raf",  "mudaaf_ilayh"): True,
    ("raf",  "majroor"): True,
    # nasb-only roles
    ("nasb", "mubtada"): True,
    ("nasb", "fail"): True,
    ("nasb", "mudaaf_ilayh"): True,
    # jarr-only roles
    ("jarr", "mubtada"): True,
    ("jarr", "fail"): True,
    ("jarr", "mafoul_bih"): True,
}

INCOMPATIBLE_CASE_MARKER = {
    ("raf",  "fatha"): True,
    ("raf",  "kasra"): True,
    ("nasb", "damma"): True,
    ("nasb", "kasra"): True,
    ("jarr", "damma"): True,
    ("jarr", "fatha"): True,
}


def _consistency_penalty(
    logits: Dict[str, "torch.Tensor"],
) -> "torch.Tensor":
    """Tiny soft penalty: probability mass placed on incompatible (case, role)
    or (case, marker) pairs is summed and returned as a scalar.

    Pure-prediction signal — uses softmax over the heads but does not
    require gold labels, so it works on every batch.
    """
    import torch
    import torch.nn.functional as F
    from ..structured.schema import (
        CASE_LABELS, ROLE_LABELS, MARKER_LABELS,
    )

    if "case" not in logits or "role" not in logits:
        return logits[next(iter(logits))].sum() * 0.0
    case_p = F.softmax(logits["case"], dim=-1)         # (B, W, C)
    role_p = F.softmax(logits["role"], dim=-1)         # (B, W, R)
    marker_p = (F.softmax(logits["marker"], dim=-1)
                if "marker" in logits else None)

    pen = case_p.sum() * 0.0
    # Build per-pair masks
    case_idx = {l: i for i, l in enumerate(CASE_LABELS)}
    role_idx = {l: i for i, l in enumerate(ROLE_LABELS)}
    marker_idx = {l: i for i, l in enumerate(MARKER_LABELS)} if marker_p is not None else {}

    for (c, r) in INCOMPATIBLE_CASE_ROLE:
        ci = case_idx.get(c); ri = role_idx.get(r)
        if ci is None or ri is None:
            continue
        joint = case_p[..., ci] * role_p[..., ri]
        pen = pen + joint.mean()

    if marker_p is not None:
        for (c, m) in INCOMPATIBLE_CASE_MARKER:
            ci = case_idx.get(c); mi = marker_idx.get(m)
            if ci is None or mi is None:
                continue
            joint = case_p[..., ci] * marker_p[..., mi]
            pen = pen + joint.mean()
    return pen


def _entropy_reg(logits: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    """Penalise overconfident predictions: -mean(max p) → encourages
    higher-entropy outputs. Capped via the lambda."""
    import torch
    import torch.nn.functional as F
    out = logits[next(iter(logits))].sum() * 0.0
    n = 0
    for name, log in logits.items():
        p = F.softmax(log, dim=-1)
        # max prob per token → high values = overconfident
        out = out + p.max(dim=-1).values.mean()
        n += 1
    return out / max(n, 1)


def _fully_aux(
    logits: Dict[str, "torch.Tensor"],
    labels: Dict[str, "torch.Tensor"],
    token_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Sentence-level exactness aux: differentiable approximation of
    "all 3 (case/role/marker) tokens correct in the sentence".

    Computed as -log(P_case_correct * P_role_correct * P_marker_correct)
    averaged across tokens with all 3 gold labels populated.
    """
    import torch
    import torch.nn.functional as F
    needed = ("case", "role", "marker")
    if not all(k in logits and k in labels for k in needed):
        return logits[next(iter(logits))].sum() * 0.0

    log_probs = []
    valid_masks = []
    for k in needed:
        log = logits[k]
        lab = labels[k]
        # Per-token log prob of the gold class
        lp = F.log_softmax(log, dim=-1)
        gold_lp = lp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        log_probs.append(gold_lp)
        valid_masks.append(lab != -100)

    valid = valid_masks[0] & valid_masks[1] & valid_masks[2]
    if token_mask is not None:
        valid = valid & token_mask.bool()
    if valid.sum().item() == 0:
        return log_probs[0].sum() * 0.0

    joint_lp = sum(lp.where(valid, torch.zeros_like(lp))
                    for lp in log_probs)
    # Negative log-likelihood of the all-three-correct event
    return -(joint_lp.sum() / valid.sum().clamp(min=1))


def compute_multi_head_loss(
    logits: Dict[str, "torch.Tensor"],
    labels: Dict[str, "torch.Tensor"],
    weights: HeadLossWeights,
    *,
    label_smoothing: float = 0.0,
    entropy_reg_lambda: float = 0.0,
    consistency_lambda: float = 0.0,
    fully_aux_lambda: float = 0.0,
    token_mask: Optional["torch.Tensor"] = None,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    per_head: Dict[str, torch.Tensor] = {}
    # Initialise total as a zero-multiple of any logit tensor so it
    # inherits a grad_fn (otherwise loss.backward() on a fresh
    # torch.tensor(0.0) raises "element 0 does not require grad").
    any_logits = next(iter(logits.values()))
    total = (any_logits.sum() * 0.0)

    def _ce(name: str):
        if name not in logits or name not in labels:
            return None
        log = logits[name]
        lab = labels[name]
        if (lab != -100).sum().item() == 0:
            return None
        return F.cross_entropy(
            log.reshape(-1, log.size(-1)), lab.reshape(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
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

    aux: Dict[str, torch.Tensor] = {}

    # Step-4 ambiguity phase: governor-prediction CE loss.
    # IMPORTANT: do NOT pass label_smoothing here — the governor logits
    # contain large-negative masked positions (pad / self-loop) and
    # smoothing would multiply tiny weight × huge negative log_softmax,
    # ballooning the loss. Plain CE with ignore_index handles the
    # masked positions cleanly.
    if "governor" in logits and "governor" in labels:
        gov_log = logits["governor"]
        gov_lab = labels["governor"]
        if (gov_lab != -100).sum().item() > 0:
            gov_loss = F.cross_entropy(
                gov_log.reshape(-1, gov_log.size(-1)),
                gov_lab.reshape(-1),
                ignore_index=-100,
            )
            aux["governor_ce"] = gov_loss
            total = total + 0.5 * gov_loss

    # Item 6 B — entropy regularization (penalise overconfident logits)
    if entropy_reg_lambda > 0:
        e = _entropy_reg(logits)
        aux["entropy"] = e
        total = total + entropy_reg_lambda * e

    # Item 9 — structured-consistency penalty
    if consistency_lambda > 0:
        c = _consistency_penalty(logits)
        aux["consistency"] = c
        total = total + consistency_lambda * c

    # Item 10 — exact-fully aux objective
    if fully_aux_lambda > 0:
        fa = _fully_aux(logits, labels, token_mask=token_mask)
        aux["fully_aux"] = fa
        total = total + fully_aux_lambda * fa

    return {"loss": total, "per_head": per_head, "aux": aux}
