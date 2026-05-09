"""Calibration metrics for eval_v2.

Computes:
  - Expected Calibration Error (ECE) on a metric (case / role / marker)
  - Reliability-diagram bin counts
  - Confidence-stratified accuracy

These complement the simple ``calib_gap = mean(conf|correct) -
mean(conf|wrong)`` already produced by :func:`metrics.aggregate_outcomes`.

ECE is the standard metric for probabilistic classifier
miscalibration: bin predictions by confidence, compare bin's mean
confidence to bin's accuracy, weight by bin size, sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .metrics import TokenOutcome


@dataclass
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    n: int                 # number of predictions in bin
    mean_conf: float       # mean predicted confidence in bin
    accuracy: float        # fraction correct in bin


@dataclass
class CalibrationReport:
    field:        str                 # "case" / "role" / "marker"
    n_total:      int
    ece:          float
    bins:         List[CalibrationBin]
    overall_acc:  float
    mean_conf:    float


def _accessor(field: str):
    if field == "case":
        return lambda o: (o.pred_case_conf, o.case_correct)
    if field == "role":
        return lambda o: (o.pred_role_conf, o.role_correct)
    if field == "marker":
        return lambda o: (o.pred_marker_conf, o.marker_correct)
    raise ValueError(f"unknown field {field!r}")


def calibration_report(
    outcomes: List[TokenOutcome], field: str,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute ECE + reliability-diagram bins for a single field.

    Outcomes with conf=None or correctness=None are skipped.
    """
    acc = _accessor(field)
    rows = []
    for o in outcomes:
        conf, correct = acc(o)
        if conf is None or correct is None:
            continue
        rows.append((float(conf), bool(correct)))

    n_total = len(rows)
    if n_total == 0:
        return CalibrationReport(
            field=field, n_total=0, ece=0.0, bins=[],
            overall_acc=0.0, mean_conf=0.0,
        )

    overall_acc = sum(1 for _, c in rows if c) / n_total
    mean_conf = sum(c for c, _ in rows) / n_total

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins: List[CalibrationBin] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # right-closed last bin (so 1.0 falls in)
        if i == n_bins - 1:
            mask = [(c, ok) for c, ok in rows if lo <= c <= hi]
        else:
            mask = [(c, ok) for c, ok in rows if lo <= c < hi]
        if not mask:
            bins.append(CalibrationBin(lo, hi, 0, 0.0, 0.0))
            continue
        n = len(mask)
        m_conf = sum(c for c, _ in mask) / n
        m_acc  = sum(1 for _, ok in mask if ok) / n
        bins.append(CalibrationBin(lo, hi, n, m_conf, m_acc))
        ece += abs(m_conf - m_acc) * (n / n_total)

    return CalibrationReport(
        field=field, n_total=n_total, ece=round(ece, 4),
        bins=bins, overall_acc=round(overall_acc, 4),
        mean_conf=round(mean_conf, 4),
    )


def calibration_for_all_fields(
    outcomes: List[TokenOutcome], n_bins: int = 10,
) -> Dict[str, CalibrationReport]:
    """Compute calibration for case / role / marker."""
    return {f: calibration_report(outcomes, f, n_bins=n_bins)
            for f in ("case", "role", "marker")}
