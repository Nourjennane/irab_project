"""Clause-level metrics for eval_v2.

Available when ``Sentence.clauses`` is populated (Step 5 clause
detection). Reports:

- per-clause case_acc / role_acc / marker_em / fully on tokens
  inside that clause
- clause-detection P/R/F1 (matching by token-overlap ≥ 0.5)
- clause-internal consistency (e.g., does the predicted role
  hierarchy match the gold clause hierarchy)

Most loaders today don't populate clauses, so these metrics
return empty dicts gracefully; they activate as the corpus
matures.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from ..data_v2.schema_v2 import Sentence
from .metrics import TokenOutcome, aggregate_outcomes
from .predictions import SentencePrediction, index_predictions_by_sentence


@dataclass
class ClauseDetectionMetrics:
    n_gold:    int = 0
    n_pred:    int = 0
    n_match:   int = 0
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0


def per_clause_metrics(
    outcomes: List[TokenOutcome], sentences: Iterable[Sentence],
) -> Dict[str, Dict[str, Any]]:
    """Bucket outcomes by clause-id (when sentence has clauses).

    Returns ``{clause_id: aggregate_outcomes(tokens_in_clause)}``.
    Empty when no sentence has clauses populated.
    """
    # Build a token-index → clause_id index per sentence
    clause_index: Dict[str, Dict[int, str]] = {}
    for s in sentences:
        if not s.clauses:
            continue
        idx: Dict[int, str] = {}
        for cl in s.clauses:
            for ti in cl.token_indices:
                idx[ti] = cl.clause_id
        clause_index[s.sentence_id] = idx

    if not clause_index:
        return {}

    by_clause: Dict[str, List[TokenOutcome]] = defaultdict(list)
    for o in outcomes:
        sid = o.sentence_id
        if sid not in clause_index:
            continue
        cid = clause_index[sid].get(o.token_index)
        if not cid:
            continue
        by_clause[cid].append(o)

    return {cid: aggregate_outcomes(group) for cid, group in by_clause.items()}


def clause_detection_metrics(
    sentences: Iterable[Sentence],
    predictions: Iterable[SentencePrediction],
) -> ClauseDetectionMetrics:
    """Empty for now — :class:`SentencePrediction` doesn't carry
    predicted clauses yet. Returns zeros gracefully so the report
    layer can include this metric without special-casing.

    This reserved slot will activate when a future predictor emits
    ``ClausePrediction`` records. Until then the field stays at
    0/0/0 in eval reports.
    """
    return ClauseDetectionMetrics()
