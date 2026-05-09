"""analysis — failure / confusion / structural / calibration tooling."""
from .failure_analysis import FailureRecord, build_failure_records
from .failure_buckets import bucket_failures
from .confusion_analysis import (
    confusion_matrix, confusion_summary, top_confusions,
)
from .structural_breakdown import structural_breakdown
from .calibration_analysis import (
    calibration_summary, ece, high_confidence_wrongs, reliability_bins,
)

__all__ = [
    "FailureRecord", "build_failure_records",
    "bucket_failures",
    "confusion_matrix", "confusion_summary", "top_confusions",
    "structural_breakdown",
    "calibration_summary", "ece", "high_confidence_wrongs", "reliability_bins",
]
