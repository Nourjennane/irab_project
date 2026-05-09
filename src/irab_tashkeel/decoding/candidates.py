"""Candidate enumeration for structured decoding.

Top-k joint candidate generation per word position, expanded into
sentence-level parse candidates via cartesian product (capped to
``max_global_beams`` for tractability).

Operates on logit tensors so it's encoder-agnostic — pluggable
behind any next-gen model that exposes the standard
case/role/marker/pos logit dict.

This module is *inference-time only*. No training, no learnable
parameters — just structured argmax candidate enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TokenCandidate:
    """One (case, role, marker) candidate for a single token position."""
    token_index: int
    case:        Optional[str]
    role:        Optional[str]
    marker:      Optional[str]
    score:       float                 # log-prob sum across the three fields
    case_conf:   float
    role_conf:   float
    marker_conf: float


@dataclass
class SentenceCandidate:
    """A complete parse candidate: one TokenCandidate per word."""
    tokens:      List[TokenCandidate]
    score:       float                 # sum of token scores
    notes:       List[str] = field(default_factory=list)


def topk_per_token(
    case_logits: np.ndarray,           # (W, n_case)
    role_logits: np.ndarray,           # (W, n_role)
    marker_logits: np.ndarray,         # (W, n_marker)
    case_labels: List[str], role_labels: List[str], marker_labels: List[str],
    *,
    k_case: int = 2, k_role: int = 3, k_marker: int = 2,
) -> List[List[TokenCandidate]]:
    """Top-k per-token candidates ordered by combined log-prob.

    Returns one ``List[TokenCandidate]`` per word position, each of
    length up to ``k_case * k_role * k_marker``, sorted by score
    descending.
    """
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)

    case_p   = _softmax(case_logits)
    role_p   = _softmax(role_logits)
    marker_p = _softmax(marker_logits)

    W = case_logits.shape[0]
    out: List[List[TokenCandidate]] = []

    for w in range(W):
        case_ranked   = np.argsort(-case_p[w])  [:k_case]
        role_ranked   = np.argsort(-role_p[w])  [:k_role]
        marker_ranked = np.argsort(-marker_p[w])[:k_marker]

        cands: List[TokenCandidate] = []
        for ci in case_ranked:
            for ri in role_ranked:
                for mi in marker_ranked:
                    cp = float(case_p[w, ci])
                    rp = float(role_p[w, ri])
                    mp = float(marker_p[w, mi])
                    score = float(np.log(cp + 1e-12)
                                   + np.log(rp + 1e-12)
                                   + np.log(mp + 1e-12))
                    cands.append(TokenCandidate(
                        token_index=w,
                        case=case_labels[ci] if ci < len(case_labels) else None,
                        role=role_labels[ri] if ri < len(role_labels) else None,
                        marker=marker_labels[mi] if mi < len(marker_labels) else None,
                        score=score,
                        case_conf=cp, role_conf=rp, marker_conf=mp,
                    ))
        cands.sort(key=lambda t: -t.score)
        out.append(cands)
    return out


def enumerate_sentence_candidates(
    per_token: List[List[TokenCandidate]],
    *, max_global_beams: int = 32, k_per_token: int = 2,
) -> List[SentenceCandidate]:
    """Beam-search-like enumeration of sentence-level parse candidates.

    Greedily extends a beam of size ``max_global_beams`` by adding
    each token's top-``k_per_token`` candidates. Returns the top
    ``max_global_beams`` sentence candidates by total score.

    Note: not full cartesian search — that's exponential. The beam
    cap keeps decode latency bounded for practical sentence
    lengths.
    """
    if not per_token:
        return []

    # Start beam with the first token's top-k
    beams: List[SentenceCandidate] = [
        SentenceCandidate(tokens=[c], score=c.score)
        for c in per_token[0][:k_per_token]
    ]

    for w in range(1, len(per_token)):
        next_beams: List[SentenceCandidate] = []
        for b in beams:
            for c in per_token[w][:k_per_token]:
                next_beams.append(SentenceCandidate(
                    tokens=b.tokens + [c],
                    score=b.score + c.score,
                ))
        next_beams.sort(key=lambda s: -s.score)
        beams = next_beams[:max_global_beams]

    return beams
