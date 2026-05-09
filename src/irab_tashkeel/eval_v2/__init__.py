"""eval_v2 — schema_v2-native evaluation engine.

Public API:

    metrics.aggregate_outcomes(outcomes)
    metrics.extract_outcomes(sentences, predictions)
    metrics.overall_metrics(sentences, predictions, fully_observable_only=False)
    metrics.construction_detection_metrics(sentences, predictions)

    stratified.stratify(outcomes, axis)
    stratified.stratified_metrics(outcomes, axes=...)
    stratified.filtered_metrics(outcomes, ...)

    predictions.SentencePrediction
    predictions.TokenPrediction
    predictions.ConstructionPrediction

See ``docs/data_v2/EVAL_V2_CONTRACT.md`` for the data contract
between data_v2 and eval_v2.
"""
from .metrics import (
    aggregate_outcomes, construction_detection_metrics, extract_outcomes,
    overall_metrics, TokenOutcome, ConstructionDetectionMetrics,
)
from .predictions import (
    ConstructionPrediction, SentencePrediction, TokenPrediction,
    index_predictions_by_sentence, index_token_predictions,
)
from .stratified import (
    filtered_metrics, stratified_metrics, stratify,
)

__all__ = [
    "aggregate_outcomes", "construction_detection_metrics",
    "extract_outcomes", "overall_metrics",
    "TokenOutcome", "ConstructionDetectionMetrics",
    "ConstructionPrediction", "SentencePrediction", "TokenPrediction",
    "index_predictions_by_sentence", "index_token_predictions",
    "filtered_metrics", "stratified_metrics", "stratify",
]
