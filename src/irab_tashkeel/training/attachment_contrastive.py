"""Step-5 of ambiguity phase — attachment-candidate contrastive loss.

For each token whose gold dep_head_idx is known, treat that head as
the positive attachment candidate and sample K negative candidates
(other tokens in the sentence). The model's governor-head logits
should rank the positive higher than every negative by a margin.

This is a *training signal* for the governor head; it complements
the cross-entropy loss in `loss.py` by injecting hard structural
contrasts. Negatives are not random — they are sampled from
plausible-but-wrong candidates:

  - the token immediately to the left/right (most attractive
    confound for short attachments)
  - the nearest noun
  - the nearest preposition
  - the nearest verb / kana / inna particle

Training-time only; no eval-time use.
"""
from __future__ import annotations

from typing import List, Optional


def _nearest_pos(s_tokens, anchor: int, target_pos: set,
                  max_dist: int = 6) -> Optional[int]:
    n = len(s_tokens)
    for d in range(1, max_dist + 1):
        for cand in (anchor - d, anchor + d):
            if 0 <= cand < n and (s_tokens[cand].pos.value or "") in target_pos:
                return cand
    return None


def sample_negatives(sentence, token_index: int, k: int = 4) -> List[int]:
    """Sample up to k plausible-but-wrong governor candidates."""
    n = len(sentence.tokens)
    if n <= 1:
        return []
    out: List[int] = []
    # Adjacent tokens
    for off in (-1, 1, -2, 2):
        c = token_index + off
        if 0 <= c < n and c != token_index:
            out.append(c)
    # Nearest noun / verb / preposition / particle
    for ps in ({"NOUN", "PROPN", "PRON"}, {"VERB", "AUX"}, {"ADP"}, {"PART"}):
        c = _nearest_pos(sentence.tokens, token_index, ps)
        if c is not None and c != token_index and c not in out:
            out.append(c)
        if len(out) >= k:
            break
    # Drop the gold head if it accidentally got included
    gold = sentence.tokens[token_index].dep_head_idx
    out = [c for c in out if c != gold]
    return out[:k]


def contrastive_attachment_loss(
    governor_logits: "torch.Tensor",        # (B, W, W)
    sentences,
    word_mask: "torch.Tensor",              # (B, W)
    *,
    margin: float = 1.0,
    k_neg: int = 4,
) -> "torch.Tensor":
    """Triplet-margin loss for attachment.

    For each (b, i) where sentences[b].tokens[i] has a valid gold
    head, ensure governor_logits[b, i, gold_head] > governor_logits[b,
    i, neg] + margin for every sampled negative.
    """
    import torch
    B, W, _ = governor_logits.shape
    losses = []
    for b in range(B):
        s = sentences[b]
        n = min(W, len(s.tokens))
        for i in range(n):
            if word_mask[b, i].item() == 0:
                continue
            head = s.tokens[i].dep_head_idx
            if head is None or head < 0 or head >= n or head == i:
                continue
            negs = sample_negatives(s, i, k=k_neg)
            negs = [neg for neg in negs if 0 <= neg < n and word_mask[b, neg].item() == 1]
            if not negs:
                continue
            pos_score = governor_logits[b, i, head]
            for neg in negs:
                neg_score = governor_logits[b, i, neg]
                losses.append(torch.clamp(neg_score - pos_score + margin, min=0.0))
    if not losses:
        return governor_logits.sum() * 0.0
    return torch.stack(losses).mean()
