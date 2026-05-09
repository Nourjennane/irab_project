"""Ambiguity-aware scoring.

Counts a prediction as correct if it matches **any** of the
declared analyses for an ambiguous token (primary OR secondary).
Falls back to ``eval_v2``-style strict scoring on tokens with no
ambiguity annotation.

Inputs:
  - sentences      : List[Sentence]      — schema_v2
  - predictions    : List[SentencePrediction]
  - ambiguities    : Dict[sentence_id -> List[AmbiguityExample]]
                      (loaded from data_v2/ambiguity_corpus/)

Outputs:
  - ambiguity_resolved_accuracy
  - strict_fully (unchanged baseline for comparison)
  - permissive_fully (with alt analyses honored)
  - permissive_minus_strict_delta
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..ambiguity.schema import AmbiguityExample, TokenAnalysis
from ..eval_v2 import SentencePrediction, TokenOutcome, extract_outcomes


@dataclass
class AmbiguityMetrics:
    n_total:                 int
    n_strict_correct:        int
    n_permissive_correct:    int
    n_ambiguous_tokens:      int
    n_ambiguous_resolved:    int

    @property
    def strict_fully(self) -> float:
        return self.n_strict_correct / max(self.n_total, 1)

    @property
    def permissive_fully(self) -> float:
        return self.n_permissive_correct / max(self.n_total, 1)

    @property
    def ambiguity_resolved_accuracy(self) -> float:
        return self.n_ambiguous_resolved / max(self.n_ambiguous_tokens, 1)


def _matches_analysis(o: TokenOutcome, a: TokenAnalysis) -> bool:
    if a.case is not None and o.pred_case != a.case:
        return False
    if a.role is not None and o.pred_role != a.role:
        return False
    if a.marker is not None and o.pred_marker != a.marker:
        return False
    return True


def evaluate_with_ambiguity(
    sentences: List, predictions: List[SentencePrediction],
    ambiguities: Optional[Dict[str, List[AmbiguityExample]]] = None,
) -> AmbiguityMetrics:
    ambiguities = ambiguities or {}
    outcomes = extract_outcomes(sentences, predictions)

    n_total = 0
    n_strict = 0
    n_perm = 0
    n_amb = 0
    n_amb_ok = 0

    for o in outcomes:
        if not o.is_fully_observable:
            continue
        n_total += 1
        if o.fully_correct is True:
            n_strict += 1
            n_perm += 1
            continue

        # Check if this token has an ambiguity annotation
        amb_list = ambiguities.get(o.sentence_id, [])
        is_ambiguous = False
        permissive_match = False
        for amb in amb_list:
            if o.token_index not in amb.span_tokens:
                continue
            is_ambiguous = True
            # Try every alternative analysis
            for analysis_dict in [amb.primary_analysis] + amb.secondary_analyses:
                if o.token_index not in analysis_dict:
                    continue
                a = analysis_dict[o.token_index]
                if _matches_analysis(o, a):
                    permissive_match = True
                    break
            if permissive_match:
                break

        if is_ambiguous:
            n_amb += 1
            if permissive_match:
                n_amb_ok += 1
                n_perm += 1

    return AmbiguityMetrics(
        n_total=n_total,
        n_strict_correct=n_strict,
        n_permissive_correct=n_perm,
        n_ambiguous_tokens=n_amb,
        n_ambiguous_resolved=n_amb_ok,
    )
