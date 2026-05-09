"""Disagreement-based active learning.

Given multiple checkpoints' predictions on the same unseen
sentence, score sentences by how much the checkpoints disagree on
case / role / marker per token. High disagreement = the model
ensemble is uncertain about this sentence's structure.

Works with ≥ 2 checkpoints. Common usage: compare
phase3a + recovery + graph and rank candidates where they diverge.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from ..eval_v2 import SentencePrediction


def _vote_disagree(votes: List[str]) -> float:
    """1 - max_share. 0 = unanimous, ~1 = maximal disagreement."""
    if not votes:
        return 0.0
    c = Counter(votes)
    return 1.0 - (c.most_common(1)[0][1] / len(votes))


def score_disagreement(
    *preds_per_ckpt: List[SentencePrediction],
) -> Dict[str, float]:
    """Score every sentence by mean per-token disagreement on all axes."""
    by_sid: Dict[str, List[List]] = {}
    for ckpt_preds in preds_per_ckpt:
        for p in ckpt_preds:
            by_sid.setdefault(p.sentence_id, []).append(p.tokens)

    scores: Dict[str, float] = {}
    for sid, ckpt_token_lists in by_sid.items():
        if len(ckpt_token_lists) < 2:
            continue
        n_words = min(len(tl) for tl in ckpt_token_lists)
        if n_words == 0:
            continue
        total = 0.0
        for j in range(n_words):
            for axis in ("case", "role", "marker"):
                votes = []
                for tl in ckpt_token_lists:
                    v = getattr(tl[j], axis, None)
                    if v is not None:
                        votes.append(v)
                total += _vote_disagree(votes)
        scores[sid] = total / (n_words * 3)
    return scores


def rank_by_disagreement(
    *preds_per_ckpt: List[SentencePrediction],
) -> List[Tuple[str, float]]:
    s = score_disagreement(*preds_per_ckpt)
    return sorted(s.items(), key=lambda x: -x[1])
