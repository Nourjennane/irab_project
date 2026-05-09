"""Per-token failure record for held-out evaluation slices.

For every token where the model's prediction differs from gold (on
fully-observable rows), produce a :class:`FailureRecord` carrying:

  - sentence_id, token index, surface
  - gold case/role/marker, predicted case/role/marker, confidences
  - which axis or axes were wrong (case-only, role-only, marker-only,
    multi-axis)
  - construction families this token participates in
  - dependency depth at this token
  - clause depth, semantic pressure, ambiguity score
  - sentence length (token count)
  - long-range flag (head distance ≥ 5)
  - overlap flag (token in ≥ 2 constructions)
  - per-axis confusion type (e.g., "raf→nasb" for case)

These records feed the bucket/confusion/structural/calibration
analyses in the sibling modules. Pure data extraction; no
visualisation, no I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..data_v2.schema_v2 import Sentence
from ..eval_v2 import SentencePrediction, TokenOutcome, extract_outcomes


@dataclass
class FailureRecord:
    sentence_id: str
    token_index: int
    surface: str

    gold_case:    Optional[str]
    gold_role:    Optional[str]
    gold_marker:  Optional[str]
    pred_case:    Optional[str]
    pred_role:    Optional[str]
    pred_marker:  Optional[str]

    case_correct:   Optional[bool]
    role_correct:   Optional[bool]
    marker_correct: Optional[bool]
    fully_correct:  Optional[bool]

    case_conf:    Optional[float]
    role_conf:    Optional[float]
    marker_conf:  Optional[float]

    # Confusion strings, e.g., "raf→nasb"; None if correct or unobservable.
    case_confusion:   Optional[str]
    role_confusion:   Optional[str]
    marker_confusion: Optional[str]

    # Structural metadata
    construction_families: List[str] = field(default_factory=list)
    dependency_depth:      int = 0
    clause_depth:          int = 0
    semantic_pressure:     int = 0
    ambiguity_score:       float = 0.0
    sentence_length:       int = 0
    long_range:            bool = False
    overlap:               bool = False
    head_distance:         int = 0

    # Calibration class — a coarse bucket from the role confidence.
    calibration_bucket:    str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _confusion(gold: Optional[str], pred: Optional[str]) -> Optional[str]:
    if gold is None or pred is None or gold == pred:
        return None
    return f"{gold}→{pred}"


def _calibration_bucket(role_conf: Optional[float]) -> str:
    if role_conf is None:
        return "unknown"
    if role_conf >= 0.95:
        return "very_high_conf"
    if role_conf >= 0.80:
        return "high_conf"
    if role_conf >= 0.50:
        return "medium_conf"
    return "low_conf"


def build_failure_records(
    sentences: List[Sentence],
    predictions: List[SentencePrediction],
    *,
    only_failures: bool = True,
) -> List[FailureRecord]:
    """Build per-token records, optionally filtering to failures only."""
    by_sid = {s.sentence_id: s for s in sentences}
    outcomes = extract_outcomes(sentences, predictions)

    records: List[FailureRecord] = []
    for o in outcomes:
        if not o.is_fully_observable:
            continue
        if only_failures and o.fully_correct is True:
            continue
        s = by_sid.get(o.sentence_id)
        if s is None:
            continue
        # Find dep head distance for this token
        head_dist = 0
        if o.token_index < len(s.tokens):
            t = s.tokens[o.token_index]
            head = t.dep_head_idx
            if head is not None and head >= 0:
                head_dist = abs(head - o.token_index)
        long_range = head_dist >= 5

        # Construction overlap: this token in ≥ 2 constructions
        n_in = sum(1 for c in s.constructions if o.token_index in c.token_indices)
        overlap = n_in >= 2

        records.append(FailureRecord(
            sentence_id=o.sentence_id,
            token_index=o.token_index,
            surface=o.word,
            gold_case=o.gold_case, gold_role=o.gold_role, gold_marker=o.gold_marker,
            pred_case=o.pred_case, pred_role=o.pred_role, pred_marker=o.pred_marker,
            case_correct=o.case_correct, role_correct=o.role_correct,
            marker_correct=o.marker_correct, fully_correct=o.fully_correct,
            case_conf=o.pred_case_conf, role_conf=o.pred_role_conf,
            marker_conf=o.pred_marker_conf,
            case_confusion=_confusion(o.gold_case, o.pred_case),
            role_confusion=_confusion(o.gold_role, o.pred_role),
            marker_confusion=_confusion(o.gold_marker, o.pred_marker),
            construction_families=sorted(o.construction_families),
            dependency_depth=o.dependency_depth,
            clause_depth=o.clause_depth,
            semantic_pressure=o.semantic_pressure,
            ambiguity_score=0.0,           # not on outcome; sentence-level fetched below
            sentence_length=o.sentence_length,
            long_range=long_range,
            overlap=overlap,
            head_distance=head_dist,
            calibration_bucket=_calibration_bucket(o.pred_role_conf),
        ))
        if s.curriculum.ambiguity_score:
            records[-1].ambiguity_score = float(s.curriculum.ambiguity_score)
    return records
