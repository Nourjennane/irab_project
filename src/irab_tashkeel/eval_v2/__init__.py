"""eval_v2 — schema_v2-native evaluation engine.

Public API:

    metrics.aggregate_outcomes(outcomes)
    metrics.extract_outcomes(sentences, predictions)
    metrics.overall_metrics(sentences, predictions, fully_observable_only=False)
    metrics.construction_detection_metrics(sentences, predictions)

    stratified.stratify(outcomes, axis)
    stratified.stratified_metrics(outcomes, axes=...)
    stratified.filtered_metrics(outcomes, ...)

    calibration.calibration_report(outcomes, field, n_bins=10)
    calibration.calibration_for_all_fields(outcomes)

    clause_metrics.per_clause_metrics(outcomes, sentences)
    clause_metrics.clause_detection_metrics(sentences, predictions)

    ambiguity_metrics.ambiguity_robustness(sentences, predictions)

    reasoning_metrics.reasoning_match(sentences, predictions=None)

    predictions.SentencePrediction
    predictions.TokenPrediction
    predictions.ConstructionPrediction

See ``docs/data_v2/EVAL_V2_CONTRACT.md`` for the data contract
between data_v2 and eval_v2.
"""
from .ambiguity_metrics import AmbiguityReport, ambiguity_robustness
from .calibration import (
    CalibrationBin, CalibrationReport,
    calibration_for_all_fields, calibration_report,
)
from .clause_metrics import (
    ClauseDetectionMetrics, clause_detection_metrics, per_clause_metrics,
)
from .metrics import (
    aggregate_outcomes, construction_detection_metrics, extract_outcomes,
    overall_metrics, TokenOutcome, ConstructionDetectionMetrics,
)
from .predictions import (
    ConstructionPrediction, SentencePrediction, TokenPrediction,
    index_predictions_by_sentence, index_token_predictions,
)
from .reasoning_metrics import ReasoningReport, reasoning_match
from .stratified import (
    filtered_metrics, stratified_metrics, stratify,
)

__all__ = [
    "AmbiguityReport", "ambiguity_robustness",
    "CalibrationBin", "CalibrationReport",
    "calibration_for_all_fields", "calibration_report",
    "ClauseDetectionMetrics", "clause_detection_metrics", "per_clause_metrics",
    "aggregate_outcomes", "construction_detection_metrics",
    "extract_outcomes", "overall_metrics",
    "TokenOutcome", "ConstructionDetectionMetrics",
    "ConstructionPrediction", "SentencePrediction", "TokenPrediction",
    "index_predictions_by_sentence", "index_token_predictions",
    "ReasoningReport", "reasoning_match",
    "filtered_metrics", "stratified_metrics", "stratify",
]
