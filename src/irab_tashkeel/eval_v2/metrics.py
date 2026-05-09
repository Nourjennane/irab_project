"""Core metrics for eval_v2.

Token-level + construction-level + clause-level + calibration +
ambiguity + completeness-aware + structural-depth + semantic-
pressure metrics, all keyed off schema_v2 fields and produced from
:class:`SentencePrediction` objects.

The frozen-baseline ``per_construction_summary`` aggregations
remain available; this module reproduces them on schema_v2 inputs
and adds new axes the frozen baseline could not stratify by
(annotation completeness, dep_depth, clause_depth, semantic
pressure).

Returned objects are JSON-serialisable dicts so the metric layer
can feed dashboards, paper tables, and downstream analyses
without coupling to a specific report renderer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..data_v2.schema_v2 import Sentence, Token
from .predictions import (
    SentencePrediction, TokenPrediction, index_predictions_by_sentence,
    index_token_predictions,
)


# ===========================================================================
# Per-token correctness
# ===========================================================================

@dataclass
class TokenOutcome:
    """One observation: model prediction × gold × correctness flags."""
    sentence_id: str
    token_index: int
    word: str

    # Gold
    gold_case:    Optional[str]
    gold_role:    Optional[str]
    gold_marker:  Optional[str]

    # Prediction
    pred_case:    Optional[str]
    pred_role:    Optional[str]
    pred_marker:  Optional[str]
    pred_case_conf:   Optional[float]
    pred_role_conf:   Optional[float]
    pred_marker_conf: Optional[float]

    # Correctness (None when gold is missing → not measurable)
    case_correct:   Optional[bool]
    role_correct:   Optional[bool]
    marker_correct: Optional[bool]
    fully_correct:  Optional[bool]

    # Stratification metadata
    domain: str
    annotation_quality: str
    completeness_pct: float
    construction_families: List[str]
    difficulty_level: int
    dependency_depth: int
    clause_depth: int
    semantic_pressure: int
    sentence_length: int

    @property
    def is_observable(self) -> bool:
        """True iff at least one gold field is present and could be scored."""
        return any(g is not None for g in
                   (self.gold_case, self.gold_role, self.gold_marker))

    @property
    def is_fully_observable(self) -> bool:
        """True iff all 3 gold fields are present (the model's true
        error rate is computable only on this subset)."""
        return all(g is not None for g in
                   (self.gold_case, self.gold_role, self.gold_marker))


# ===========================================================================
# Outcome extraction
# ===========================================================================

def _gold_token(t: Token) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (t.case.value, t.role.value, t.marker.value)


def _build_outcome(
    sentence: Sentence, token: Token,
    pred: Optional[TokenPrediction],
) -> TokenOutcome:
    gc, gr, gm = _gold_token(token)
    pc = pred.case if pred else None
    pr = pred.role if pred else None
    pm = pred.marker if pred else None

    cc = (pc == gc) if gc is not None else None
    rc = (pr == gr) if gr is not None else None
    mc = (pm == gm) if gm is not None else None
    fc: Optional[bool]
    if any(v is None for v in (gc, gr, gm)):
        fc = None
    else:
        fc = (cc is True and rc is True and mc is True)

    return TokenOutcome(
        sentence_id=sentence.sentence_id,
        token_index=token.index,
        word=token.surface,
        gold_case=gc, gold_role=gr, gold_marker=gm,
        pred_case=pc, pred_role=pr, pred_marker=pm,
        pred_case_conf=pred.case_conf if pred else None,
        pred_role_conf=pred.role_conf if pred else None,
        pred_marker_conf=pred.marker_conf if pred else None,
        case_correct=cc, role_correct=rc, marker_correct=mc,
        fully_correct=fc,
        domain=sentence.metadata.domain,
        annotation_quality=sentence.metadata.annotation_quality,
        completeness_pct=sentence.completeness.fields_complete_pct,
        construction_families=sorted({c.family for c in sentence.constructions}),
        difficulty_level=sentence.curriculum.difficulty_level,
        dependency_depth=sentence.curriculum.dependency_depth,
        clause_depth=sentence.curriculum.clause_depth,
        semantic_pressure=sentence.curriculum.semantic_pressure_score,
        sentence_length=sentence.curriculum.sentence_length_tokens or sentence.n_tokens,
    )


def extract_outcomes(
    sentences: Iterable[Sentence],
    predictions: Iterable[SentencePrediction],
) -> List[TokenOutcome]:
    """Pair each token with its prediction and return a flat list of outcomes."""
    pred_idx = index_predictions_by_sentence(list(predictions))
    out: List[TokenOutcome] = []
    for s in sentences:
        sp = pred_idx.get(s.sentence_id)
        per_tok = index_token_predictions(sp) if sp else {}
        for t in s.tokens:
            tp = per_tok.get(t.index)
            out.append(_build_outcome(s, t, tp))
    return out


# ===========================================================================
# Aggregation
# ===========================================================================

def _safe_div(a: int, b: int) -> float:
    return a / b if b > 0 else 0.0


def aggregate_outcomes(outcomes: List[TokenOutcome]) -> Dict[str, Any]:
    """Compute case / role / marker / fully accuracy + calibration on a slice."""
    n = len(outcomes)
    n_obs_case   = sum(1 for o in outcomes if o.case_correct   is not None)
    n_obs_role   = sum(1 for o in outcomes if o.role_correct   is not None)
    n_obs_marker = sum(1 for o in outcomes if o.marker_correct is not None)
    n_obs_fully  = sum(1 for o in outcomes if o.fully_correct  is not None)

    n_corr_case   = sum(1 for o in outcomes if o.case_correct   is True)
    n_corr_role   = sum(1 for o in outcomes if o.role_correct   is True)
    n_corr_marker = sum(1 for o in outcomes if o.marker_correct is True)
    n_corr_fully  = sum(1 for o in outcomes if o.fully_correct  is True)

    # Calibration: mean role_conf on correct vs wrong
    confs_correct = [o.pred_role_conf for o in outcomes
                     if o.role_correct is True and o.pred_role_conf is not None]
    confs_wrong   = [o.pred_role_conf for o in outcomes
                     if o.role_correct is False and o.pred_role_conf is not None]
    calib_correct = sum(confs_correct) / len(confs_correct) if confs_correct else 0.0
    calib_wrong   = sum(confs_wrong)   / len(confs_wrong)   if confs_wrong   else 0.0

    return {
        "n_words":            n,
        "n_observable_case":  n_obs_case,
        "n_observable_role":  n_obs_role,
        "n_observable_marker": n_obs_marker,
        "n_observable_fully": n_obs_fully,
        "case_acc":   round(_safe_div(n_corr_case,   n_obs_case),   4),
        "role_acc":   round(_safe_div(n_corr_role,   n_obs_role),   4),
        "marker_em":  round(_safe_div(n_corr_marker, n_obs_marker), 4),
        "fully":      round(_safe_div(n_corr_fully,  n_obs_fully),  4),
        "calib_correct": round(calib_correct, 4),
        "calib_wrong":   round(calib_wrong, 4),
        "calib_gap":     round(calib_correct - calib_wrong, 4),
    }


# ===========================================================================
# Construction-level metric
# ===========================================================================

@dataclass
class ConstructionDetectionMetrics:
    """Detection P / R / F1 on a per-family basis."""
    n_gold:       int = 0
    n_pred:       int = 0
    n_match:      int = 0
    precision:    float = 0.0
    recall:       float = 0.0
    f1:           float = 0.0


def construction_detection_metrics(
    gold_sentences: Iterable[Sentence],
    predictions: Iterable[SentencePrediction],
) -> Dict[str, ConstructionDetectionMetrics]:
    """Compute per-family P/R/F1 for construction detection.

    Match criterion: same ``family`` AND overlapping ``token_indices``.
    A predicted construction is a true positive if it matches at
    least one gold construction by family + token overlap ≥ 0.5.
    """
    pred_idx = index_predictions_by_sentence(list(predictions))
    per_family: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"gold": 0, "pred": 0, "match": 0}
    )

    for s in gold_sentences:
        sp = pred_idx.get(s.sentence_id)
        gold_cs = s.constructions
        pred_cs = sp.constructions if sp else []
        # Count
        for g in gold_cs:
            per_family[g.family]["gold"] += 1
        for p in pred_cs:
            per_family[p.family]["pred"] += 1

        # Match
        for p in pred_cs:
            for g in gold_cs:
                if g.family != p.family: continue
                gset = set(g.token_indices)
                pset = set(p.token_indices)
                if not gset or not pset: continue
                inter = len(gset & pset)
                union = len(gset | pset)
                if inter / max(union, 1) >= 0.5:
                    per_family[g.family]["match"] += 1
                    break

    out: Dict[str, ConstructionDetectionMetrics] = {}
    for fam, c in per_family.items():
        m = ConstructionDetectionMetrics(
            n_gold=c["gold"], n_pred=c["pred"], n_match=c["match"],
        )
        m.precision = _safe_div(c["match"], c["pred"])
        m.recall    = _safe_div(c["match"], c["gold"])
        if (m.precision + m.recall) > 0:
            m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)
        out[fam] = m
    return out


# ===========================================================================
# Top-level convenience
# ===========================================================================

def overall_metrics(
    sentences: Iterable[Sentence],
    predictions: Iterable[SentencePrediction],
    *,
    fully_observable_only: bool = False,
) -> Dict[str, Any]:
    """Compute overall metrics + construction detection in one call.

    Setting ``fully_observable_only=True`` restricts to outcomes
    where all 3 gold fields are present — the model's *true* error
    rate, the metric the Step 16 ceiling analysis identified as the
    one that actually reflects model capability.
    """
    outcomes = extract_outcomes(sentences, predictions)
    if fully_observable_only:
        outcomes = [o for o in outcomes if o.is_fully_observable]

    return {
        "overall": aggregate_outcomes(outcomes),
        "n_observable_outcomes": sum(1 for o in outcomes if o.is_observable),
        "n_fully_observable": sum(1 for o in outcomes if o.is_fully_observable),
        "n_total_outcomes": len(outcomes),
        "construction_detection":
            {fam: vars(m) for fam, m in
             construction_detection_metrics(sentences, predictions).items()},
    }
