"""Uncertainty-based active learning candidate scoring.

For each unseen sentence, compute a per-sentence uncertainty score
from the model's per-token softmax. High score → high information
gain if annotated.

Score families:

  - max_entropy        : average per-token entropy across heads
  - min_top1           : 1 − mean of top-1 prob (high = uncertain)
  - margin             : mean of (top1 − top2) inverted

Returns ranked candidates sorted by score descending.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from ..eval_v2 import SentencePrediction


def _entropy(probs: List[float]) -> float:
    e = 0.0
    for p in probs:
        if p > 0:
            e -= p * math.log(p + 1e-12)
    return e


def _top_two_diff(probs: List[float]) -> float:
    s = sorted(probs, reverse=True)
    if len(s) < 2:
        return 1.0
    return s[0] - s[1]


def score_max_entropy(pred: SentencePrediction) -> float:
    """Mean entropy across role / case / marker per token (proxy from
    confidences — only top1 is stored, so entropy is approximated by
    -p_top1*log(p_top1) which is monotonic in 1-p_top1)."""
    if not pred.tokens:
        return 0.0
    s = 0.0
    n = 0
    for t in pred.tokens:
        for c in (t.role_conf, t.case_conf, t.marker_conf):
            if c is None:
                continue
            # Approximate sentence-level entropy with -p log p (degenerate
            # when only top1 is known; useful as a rough proxy).
            s += -(c * math.log(c + 1e-12))
            n += 1
    return s / max(n, 1)


def score_min_top1(pred: SentencePrediction) -> float:
    if not pred.tokens:
        return 0.0
    confs = []
    for t in pred.tokens:
        for c in (t.role_conf, t.case_conf, t.marker_conf):
            if c is not None:
                confs.append(c)
    if not confs:
        return 0.0
    return 1.0 - (sum(confs) / len(confs))


def rank_by_uncertainty(
    predictions: List[SentencePrediction],
    *, scorer=score_min_top1,
) -> List[Tuple[str, float]]:
    """Return [(sentence_id, score)] sorted descending by uncertainty."""
    ranked = [(p.sentence_id, float(scorer(p))) for p in predictions]
    ranked.sort(key=lambda x: -x[1])
    return ranked
