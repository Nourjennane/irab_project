"""eval_v3 — ambiguity / uncertainty / structural metrics."""
from .ambiguity_metrics import AmbiguityMetrics, evaluate_with_ambiguity
from .uncertainty_metrics import (
    calibrated_fully, confidence_correctness_alignment,
    high_confidence_error_rate, selective_accuracy,
)
from .structural_metrics import (
    attachment_accuracy, governor_accuracy, overlap_accuracy,
)

__all__ = [
    "AmbiguityMetrics", "evaluate_with_ambiguity",
    "calibrated_fully", "confidence_correctness_alignment",
    "high_confidence_error_rate", "selective_accuracy",
    "attachment_accuracy", "governor_accuracy", "overlap_accuracy",
]
