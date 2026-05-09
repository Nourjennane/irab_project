"""Stratified metrics for eval_v2.

Partitions a flat list of :class:`TokenOutcome` records along
multiple axes (construction family, domain, difficulty,
annotation completeness, semantic pressure, dep / clause depth,
prediction-confidence bucket) and reports
:func:`aggregate_outcomes` on each slice.

This is the load-bearing layer for the Step 16 ceiling analysis:
once we strip annotation_sparsity, we want to compute the model's
*true* error rate. Stratified metrics let us report the same
numbers separately on the
``annotation_completeness=fully_observable`` subset vs the full
corpus.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from .metrics import TokenOutcome, aggregate_outcomes


# ===========================================================================
# Axis partitioning
# ===========================================================================

def _length_bucket(n: int) -> str:
    if n <= 8:  return "short"
    if n <= 16: return "medium"
    if n <= 32: return "long"
    return "xlong"


def _confidence_bucket(p: float) -> str:
    if p < 0.5:  return "<0.5"
    if p < 0.7:  return "0.5-0.7"
    if p < 0.9:  return "0.7-0.9"
    return ">=0.9"


def _completeness_bucket(p: float) -> str:
    if p >= 0.99:  return "fully_observable"
    if p >= 0.66:  return "two_of_three"
    if p >= 0.33:  return "one_of_three"
    return "none"


def stratify(
    outcomes: List[TokenOutcome], axis: str,
) -> Dict[str, List[TokenOutcome]]:
    """Bucket outcomes by ``axis``. Returns ``{key: [outcomes]}``."""
    buckets: Dict[str, List[TokenOutcome]] = defaultdict(list)
    for o in outcomes:
        if axis == "construction_family":
            if o.construction_families:
                for f in o.construction_families:
                    buckets[f].append(o)
            else:
                buckets["_no_construction"].append(o)
            buckets["_overall"].append(o)
        elif axis == "domain":
            buckets[o.domain or "unknown"].append(o)
        elif axis == "annotation_quality":
            buckets[o.annotation_quality or "unknown"].append(o)
        elif axis == "completeness":
            buckets[_completeness_bucket(o.completeness_pct)].append(o)
        elif axis == "difficulty":
            buckets[f"diff{o.difficulty_level}"].append(o)
        elif axis == "semantic_pressure":
            buckets[f"sp{o.semantic_pressure}"].append(o)
        elif axis == "dependency_depth":
            buckets[f"depth{o.dependency_depth}"].append(o)
        elif axis == "clause_depth":
            buckets[f"clause{o.clause_depth}"].append(o)
        elif axis == "length_bucket":
            buckets[_length_bucket(o.sentence_length)].append(o)
        elif axis == "case_conf_bucket":
            if o.pred_case_conf is None: continue
            buckets[_confidence_bucket(o.pred_case_conf)].append(o)
        elif axis == "role_conf_bucket":
            if o.pred_role_conf is None: continue
            buckets[_confidence_bucket(o.pred_role_conf)].append(o)
        else:
            raise ValueError(f"unknown stratify axis: {axis!r}")
    return buckets


# ===========================================================================
# Top-level
# ===========================================================================

def stratified_metrics(
    outcomes: List[TokenOutcome],
    axes: Iterable[str] = (
        "construction_family", "domain", "completeness",
        "difficulty", "semantic_pressure", "length_bucket",
    ),
) -> Dict[str, Dict[str, Any]]:
    """Return ``{axis: {bucket: aggregate}}`` for each requested axis."""
    out: Dict[str, Dict[str, Any]] = {}
    for axis in axes:
        buckets = stratify(outcomes, axis)
        out[axis] = {key: aggregate_outcomes(group)
                     for key, group in sorted(buckets.items())}
    return out


def filtered_metrics(
    outcomes: List[TokenOutcome],
    *,
    annotation_quality: List[str] = None,
    domain: List[str] = None,
    completeness_min_pct: float = None,
    construction_family: str = None,
    difficulty_min: int = None,
    difficulty_max: int = None,
    semantic_pressure_min: int = None,
    confidence_min: float = None,
) -> Dict[str, Any]:
    """Compute metrics on a subset filtered by the given criteria.

    Returns a single aggregate (not stratified). Useful for
    answering targeted questions like "how does the model do on
    fully-observable kana_sisters in MASAQ?"
    """
    flt: List[TokenOutcome] = []
    for o in outcomes:
        if annotation_quality and o.annotation_quality not in annotation_quality:
            continue
        if domain and o.domain not in domain:
            continue
        if completeness_min_pct is not None and o.completeness_pct < completeness_min_pct:
            continue
        if construction_family and construction_family not in o.construction_families:
            continue
        if difficulty_min is not None and o.difficulty_level < difficulty_min:
            continue
        if difficulty_max is not None and o.difficulty_level > difficulty_max:
            continue
        if semantic_pressure_min is not None and o.semantic_pressure < semantic_pressure_min:
            continue
        if confidence_min is not None:
            confs = [o.pred_case_conf, o.pred_role_conf, o.pred_marker_conf]
            if not all(c is not None and c >= confidence_min for c in confs):
                continue
        flt.append(o)
    return {
        "n_filtered": len(flt),
        **aggregate_outcomes(flt),
    }
