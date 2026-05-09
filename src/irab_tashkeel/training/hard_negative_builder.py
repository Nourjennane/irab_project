"""Hard-negative pair builder for contrastive structural training.

Generates contrastive pairs ``(anchor, hard_negative)`` from training
sentences such that the hard_negative is *structurally similar but
syntactically different*. Supported confusion families:

  - same_surface_diff_role     — same word, different role label
  - same_marker_diff_attach    — same case marker, different attachment
  - same_construction_diff_gov — same construction family, different governor
  - near_syntax_one_change     — near-identical syntax, one grammatical change

These pairs feed an InfoNCE / cosine-margin auxiliary loss
(:func:`contrastive_loss`). Random negatives are NOT used — the goal
is to break shallow heuristics, not to make the model nearest-neighbour
distinct from arbitrary noise.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..data_v2.schema_v2 import Sentence, Token


@dataclass
class HardNegPair:
    anchor: Sentence
    negative: Sentence
    family: str            # confusion family name
    anchor_token: int      # token index in anchor that defines the contrast
    negative_token: int    # token index in negative


def _by_role(sents: List[Sentence]) -> Dict[str, List[Tuple[Sentence, int]]]:
    out: Dict[str, List[Tuple[Sentence, int]]] = defaultdict(list)
    for s in sents:
        for t in s.tokens:
            r = t.role.value
            if r and r != "UNK":
                out[r].append((s, t.index))
    return out


def _by_surface(sents: List[Sentence]) -> Dict[str, List[Tuple[Sentence, int]]]:
    out: Dict[str, List[Tuple[Sentence, int]]] = defaultdict(list)
    for s in sents:
        for t in s.tokens:
            out[t.surface].append((s, t.index))
    return out


def build_same_surface_diff_role(
    sents: List[Sentence], rng: random.Random,
    max_pairs: int = 5000,
) -> List[HardNegPair]:
    """Pairs where same surface form takes a different role across sentences."""
    by_surf = _by_surface(sents)
    pairs: List[HardNegPair] = []
    for surf, occurrences in by_surf.items():
        if len(occurrences) < 2:
            continue
        # Group by role within this surface
        by_role: Dict[str, List[Tuple[Sentence, int]]] = defaultdict(list)
        for s, ti in occurrences:
            by_role[s.tokens[ti].role.value].append((s, ti))
        roles = [r for r, lst in by_role.items() if lst]
        if len(roles) < 2:
            continue
        rng.shuffle(roles)
        for r1 in roles:
            for r2 in roles:
                if r1 == r2:
                    continue
                a_s, a_t = rng.choice(by_role[r1])
                n_s, n_t = rng.choice(by_role[r2])
                if a_s.sentence_id == n_s.sentence_id:
                    continue
                pairs.append(HardNegPair(
                    anchor=a_s, negative=n_s, family="same_surface_diff_role",
                    anchor_token=a_t, negative_token=n_t,
                ))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def build_same_construction_diff_gov(
    sents: List[Sentence], rng: random.Random,
    max_pairs: int = 5000,
) -> List[HardNegPair]:
    """Pairs from the same construction family with different governor heads."""
    by_fam: Dict[str, List[Sentence]] = defaultdict(list)
    for s in sents:
        for c in s.constructions:
            by_fam[c.family].append(s)
    pairs: List[HardNegPair] = []
    for fam, lst in by_fam.items():
        if len(lst) < 2:
            continue
        rng.shuffle(lst)
        for i in range(0, len(lst) - 1, 2):
            a, b = lst[i], lst[i + 1]
            if a.sentence_id == b.sentence_id:
                continue
            pairs.append(HardNegPair(
                anchor=a, negative=b, family="same_construction_diff_gov",
                anchor_token=0, negative_token=0,
            ))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_near_syntax_one_change(
    sents: List[Sentence], rng: random.Random,
    max_pairs: int = 5000,
) -> List[HardNegPair]:
    """Pairs whose surface forms differ by exactly one token but
    structures differ — caught via Hamming distance on surface lists."""
    surf_keys: Dict[Tuple[str, ...], List[Sentence]] = defaultdict(list)
    for s in sents:
        if s.n_tokens < 3 or s.n_tokens > 25:
            continue
        # Coarse bucket: length + first token
        key = (str(s.n_tokens), s.tokens[0].surface)
        surf_keys[key].append(s)
    pairs: List[HardNegPair] = []
    for bucket in surf_keys.values():
        if len(bucket) < 2:
            continue
        for i, a in enumerate(bucket):
            for b in bucket[i + 1:]:
                if a.sentence_id == b.sentence_id:
                    continue
                if len(a.tokens) != len(b.tokens):
                    continue
                diffs = sum(1 for x, y in zip(a.tokens, b.tokens)
                             if x.surface != y.surface)
                if diffs == 1:
                    pairs.append(HardNegPair(
                        anchor=a, negative=b, family="near_syntax_one_change",
                        anchor_token=0, negative_token=0,
                    ))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def build_hard_negatives(
    sents: List[Sentence], *, seed: int = 0,
    per_family_cap: int = 3000,
) -> List[HardNegPair]:
    """Build a mixed pool of hard-negative pairs, capped per family."""
    rng = random.Random(seed)
    out: List[HardNegPair] = []
    out += build_same_surface_diff_role(sents, rng, per_family_cap)
    out += build_same_construction_diff_gov(sents, rng, per_family_cap)
    out += build_near_syntax_one_change(sents, rng, per_family_cap)
    return out


# ===========================================================================
# Contrastive loss (cosine-margin)
# ===========================================================================

def contrastive_loss(
    anchor_emb: "torch.Tensor",     # (B, D)
    positive_emb: "torch.Tensor",   # (B, D) — gold-conditioned embedding
    negative_emb: "torch.Tensor",   # (B, D) — hard negative embedding
    *, margin: float = 0.2,
) -> "torch.Tensor":
    """Triplet cosine-margin loss.

    Pulls anchor closer to positive than to the hard negative by ``margin``.
    """
    import torch
    import torch.nn.functional as F
    a = F.normalize(anchor_emb, dim=-1)
    p = F.normalize(positive_emb, dim=-1)
    n = F.normalize(negative_emb, dim=-1)
    pos_sim = (a * p).sum(dim=-1)
    neg_sim = (a * n).sum(dim=-1)
    return torch.clamp(neg_sim - pos_sim + margin, min=0.0).mean()


def info_nce_loss(
    anchor_emb: "torch.Tensor",      # (B, D)
    positive_emb: "torch.Tensor",    # (B, D)
    negative_embs: "torch.Tensor",   # (B, K, D)
    *, temperature: float = 0.1,
) -> "torch.Tensor":
    """InfoNCE: anchor⋅positive vs anchor⋅each negative."""
    import torch
    import torch.nn.functional as F
    a = F.normalize(anchor_emb, dim=-1)
    p = F.normalize(positive_emb, dim=-1)
    n = F.normalize(negative_embs, dim=-1)
    pos = (a * p).sum(dim=-1, keepdim=True) / temperature
    neg = torch.einsum("bd,bkd->bk", a, n) / temperature
    logits = torch.cat([pos, neg], dim=1)
    target = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, target)
