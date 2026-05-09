"""Grammar-consistency reranker for structured decoding.

Combines the per-token log-prob score (from
:mod:`candidates`) with the grammar-consistency scorers (from
:mod:`scorer`) into a single rerank score, then returns the
top-k sentence candidates.

The reranker is *non-trainable* — the only knobs are the four
weights below, which can be tuned per-construction-family but
default to the empirical mid-point that works on the
frozen-baseline Phase 3-A.

Failure-mode note
-----------------

The frozen baseline established that learned rerankers (Phase R-C
soft-bias, Phase R2 override) plateau at this scale. The Step 8
reranker is *deterministic* — it doesn't learn; it scores and
sorts. This makes it qualitatively different from the failed
inference-side mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from ..data_v2.schema_v2 import Sentence
from .candidates import SentenceCandidate
from .scorer import (
    agreement_consistency, construction_consistency, dep_role_consistency,
)


@dataclass
class RerankerWeights:
    """Score combination: weighted sum of log-prob + consistency components.

    Values are heuristic mid-points. A future Step 8.1 work could
    tune them per-construction-family on a held-out set, but the
    weights live as data here, not as learnable parameters.
    """
    log_prob_weight:                float = 1.0
    construction_consistency:       float = 1.5
    dep_role_consistency:           float = 1.0
    agreement_consistency:          float = 0.5


@dataclass
class ScoredSentenceCandidate:
    candidate:           SentenceCandidate
    rerank_score:        float
    log_prob:            float
    construction_score:  float
    dep_role_score:      float
    agreement_score:     float


def rerank(
    candidates: List[SentenceCandidate], sentence: Sentence,
    weights: RerankerWeights = RerankerWeights(),
) -> List[ScoredSentenceCandidate]:
    """Rescore + reorder ``candidates`` by combined score.

    Returns the same candidates, sorted by ``rerank_score`` desc,
    each wrapped in a :class:`ScoredSentenceCandidate` with the
    component scores exposed for diagnostic logging.
    """
    out: List[ScoredSentenceCandidate] = []
    for c in candidates:
        cs = construction_consistency(c, sentence)
        ds = dep_role_consistency(c, sentence)
        ag = agreement_consistency(c, sentence)
        rerank_score = (
            weights.log_prob_weight        * c.score
            + weights.construction_consistency * cs
            + weights.dep_role_consistency     * ds
            + weights.agreement_consistency    * ag
        )
        out.append(ScoredSentenceCandidate(
            candidate=c, rerank_score=rerank_score, log_prob=c.score,
            construction_score=cs, dep_role_score=ds, agreement_score=ag,
        ))
    out.sort(key=lambda x: -x.rerank_score)
    return out


def best_candidate(
    candidates: List[SentenceCandidate], sentence: Sentence,
    weights: RerankerWeights = RerankerWeights(),
) -> ScoredSentenceCandidate:
    """Convenience: return the single best-reranked candidate."""
    scored = rerank(candidates, sentence, weights)
    if not scored:
        raise ValueError("no candidates to rerank")
    return scored[0]


def ambiguity_margin(scored: List[ScoredSentenceCandidate]) -> float:
    """Return ``rerank_score[0] − rerank_score[1]``.

    Small margin = high ambiguity. The decoder caller can use
    this to decide whether to surface alternatives in the
    output (Step 8 ambiguity-resolution requirement).
    """
    if len(scored) < 2:
        return float("inf")
    return scored[0].rerank_score - scored[1].rerank_score
