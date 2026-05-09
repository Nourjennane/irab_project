"""Ambiguity-robustness metrics for eval_v2.

When a gold construction has ``alternative_analyses`` populated,
the model is rewarded for either matching the primary analysis OR
matching any of the alternatives. Two reports:

- **strict_match_rate** — fraction matching the primary analysis only
- **lenient_match_rate** — fraction matching primary OR any alternative
- **ambiguity_premium** — lenient - strict (how much the model
  benefits from the leniency)

A high ambiguity_premium suggests the model is learning the
*alternative* parses; a near-zero premium means the model is
either always right (no premium needed) or systematically wrong
(no alternative was chosen either).

This metric only activates when constructions carry
``alternative_analyses`` — currently only when overlap is detected.
Future Step 9 reasoning supervision will populate alternatives more
broadly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..data_v2.schema_v2 import Construction, Sentence
from .predictions import (
    ConstructionPrediction, SentencePrediction, index_predictions_by_sentence,
)


@dataclass
class AmbiguityReport:
    n_gold_constructions:        int = 0
    n_with_alternatives:         int = 0
    n_strict_match:              int = 0
    n_lenient_match:             int = 0
    strict_match_rate:           float = 0.0
    lenient_match_rate:          float = 0.0
    ambiguity_premium:           float = 0.0


def _construction_match(
    pred: ConstructionPrediction, gold: Construction,
    overlap_threshold: float = 0.5,
) -> bool:
    if pred.family != gold.family:
        return False
    gset = set(gold.token_indices)
    pset = set(pred.token_indices)
    if not gset or not pset:
        return False
    inter = len(gset & pset)
    union = len(gset | pset)
    return (inter / union) >= overlap_threshold


def _matches_any_alternative(
    pred: ConstructionPrediction, alt_analyses: List[dict],
    overlap_threshold: float = 0.5,
) -> bool:
    pset = set(pred.token_indices)
    for alt in alt_analyses:
        alt_family = alt.get("family")
        alt_tokens = set(alt.get("token_indices", []))
        if alt_family != pred.family:
            continue
        if not alt_tokens:
            continue
        inter = len(alt_tokens & pset)
        union = len(alt_tokens | pset)
        if union > 0 and inter / union >= overlap_threshold:
            return True
    return False


def ambiguity_robustness(
    sentences: Iterable[Sentence],
    predictions: Iterable[SentencePrediction],
    overlap_threshold: float = 0.5,
) -> AmbiguityReport:
    """Compute strict + lenient construction match rates.

    Lenient counts predictions that match either the primary or any
    alternative analysis. Strict counts only primary matches.
    """
    pred_idx = index_predictions_by_sentence(list(predictions))
    n_gold = 0
    n_with_alt = 0
    n_strict = 0
    n_lenient = 0

    for s in sentences:
        sp = pred_idx.get(s.sentence_id)
        pred_cs = sp.constructions if sp else []
        for g in s.constructions:
            n_gold += 1
            has_alt = bool(g.alternative_analyses)
            if has_alt:
                n_with_alt += 1
            # find best matching predicted construction
            best_strict = False
            best_lenient = False
            for p in pred_cs:
                if _construction_match(p, g, overlap_threshold):
                    best_strict = True
                    best_lenient = True
                    break
                if has_alt and _matches_any_alternative(
                    p, g.alternative_analyses, overlap_threshold,
                ):
                    best_lenient = True
            if best_strict:
                n_strict += 1
            if best_lenient:
                n_lenient += 1

    strict_rate  = n_strict / max(n_gold, 1)
    lenient_rate = n_lenient / max(n_gold, 1)
    return AmbiguityReport(
        n_gold_constructions=n_gold,
        n_with_alternatives=n_with_alt,
        n_strict_match=n_strict,
        n_lenient_match=n_lenient,
        strict_match_rate=round(strict_rate, 4),
        lenient_match_rate=round(lenient_rate, 4),
        ambiguity_premium=round(lenient_rate - strict_rate, 4),
    )
