"""decoding — Step 8 structured-decoding inference layer.

Public API:

    candidates.TokenCandidate
    candidates.SentenceCandidate
    candidates.topk_per_token(case_logits, role_logits, marker_logits, ...)
    candidates.enumerate_sentence_candidates(per_token, max_global_beams, k_per_token)

    scorer.construction_consistency(candidate, sentence)
    scorer.dep_role_consistency(candidate, sentence)
    scorer.agreement_consistency(candidate, sentence)

    reranker.RerankerWeights
    reranker.ScoredSentenceCandidate
    reranker.rerank(candidates, sentence, weights)
    reranker.best_candidate(candidates, sentence, weights)
    reranker.ambiguity_margin(scored)
"""
from .candidates import (
    SentenceCandidate, TokenCandidate,
    enumerate_sentence_candidates, topk_per_token,
)
from .reranker import (
    RerankerWeights, ScoredSentenceCandidate,
    ambiguity_margin, best_candidate, rerank,
)
from .scorer import (
    agreement_consistency, construction_consistency, dep_role_consistency,
    CANONICAL_CONSTRUCTION_CASE, DEPREL_TO_CANONICAL_ROLES,
)

__all__ = [
    "SentenceCandidate", "TokenCandidate",
    "enumerate_sentence_candidates", "topk_per_token",
    "RerankerWeights", "ScoredSentenceCandidate",
    "ambiguity_margin", "best_candidate", "rerank",
    "agreement_consistency", "construction_consistency", "dep_role_consistency",
    "CANONICAL_CONSTRUCTION_CASE", "DEPREL_TO_CANONICAL_ROLES",
]
