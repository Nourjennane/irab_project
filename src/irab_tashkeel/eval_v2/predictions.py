"""Prediction containers for eval_v2.

Defines the shape that any next-gen predictor must produce so
eval_v2 can compute metrics. All prediction inputs to the
:mod:`metrics` and :mod:`stratified` modules go through this
container.

Design choice: rather than coupling eval_v2 to a specific model
class, predictions are flat dataclasses keyed by the same
``Sentence.sentence_id`` and per-token ``Token.index`` used in
schema_v2. Any predictor — frozen-baseline ``StructuredPredictor``,
a future trained next-gen system, or a baseline like raw Stanza —
can adapt by emitting these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TokenPrediction:
    """Per-token prediction record.

    All fields are optional — a model that only emits case can
    leave role / marker as None. eval_v2 will report metrics only
    on fields where both prediction and gold are populated.
    """
    sentence_id:  str = ""
    token_index:  int = 0
    case:         Optional[str]   = None
    role:         Optional[str]   = None
    marker:       Optional[str]   = None
    pos:          Optional[str]   = None
    case_conf:    Optional[float] = None
    role_conf:    Optional[float] = None
    marker_conf:  Optional[float] = None
    pos_conf:     Optional[float] = None
    notes:        List[str]       = field(default_factory=list)


@dataclass
class ConstructionPrediction:
    """Per-construction prediction record (when a model emits these)."""
    sentence_id:       str       = ""
    construction_id:   str       = ""    # echo from gold construction OR new id
    family:            str       = ""
    subgroup:          str       = ""
    token_indices:     List[int] = field(default_factory=list)
    head_idx:          Optional[int] = None
    confidence:        float     = 1.0


@dataclass
class SentencePrediction:
    """Bundle of token + (optional) construction predictions for one sentence."""
    sentence_id:     str = ""
    tokens:          List[TokenPrediction] = field(default_factory=list)
    constructions:   List[ConstructionPrediction] = field(default_factory=list)


# ===========================================================================
# Indexing helpers (for fast metric computation)
# ===========================================================================

def index_predictions_by_sentence(
    predictions: List[SentencePrediction],
) -> Dict[str, SentencePrediction]:
    return {p.sentence_id: p for p in predictions}


def index_token_predictions(
    sentence_pred: SentencePrediction,
) -> Dict[int, TokenPrediction]:
    return {t.token_index: t for t in sentence_pred.tokens}
