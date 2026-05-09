"""Confidence-bucketed calibration analysis on failure records.

Produces:

  - per-axis reliability bin counts (10 bins, 0.0..1.0)
  - per-axis ECE
  - "high-confidence wrong" — count and listing of records where the
    model was very confident (≥ 0.95) but wrong on that axis. These
    are the most pedagogically interesting failures and the most
    important for active learning.
  - per-bucket ECE (when paired with bucket assignments)

Pure analysis; no I/O.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .failure_analysis import FailureRecord


def _bin_index(c: float, n_bins: int = 10) -> int:
    if c is None:
        return -1
    return min(n_bins - 1, max(0, int(c * n_bins)))


def reliability_bins(
    records_or_outcomes: List, axis: str, n_bins: int = 10,
) -> Dict[str, list]:
    """Per-bin (count, accuracy, mean_confidence). Works on either
    FailureRecord (where we know correctness directly) or on raw
    outcome tuples (in case caller passes them).

    Returns a dict with parallel arrays of length n_bins.
    """
    bin_n        = [0] * n_bins
    bin_correct  = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    for r in records_or_outcomes:
        c = getattr(r, f"{axis}_conf", None)
        gold = getattr(r, f"gold_{axis}", None)
        pred = getattr(r, f"pred_{axis}", None)
        if c is None or gold is None or pred is None:
            continue
        b = _bin_index(c, n_bins)
        if b < 0:
            continue
        bin_n[b] += 1
        bin_conf_sum[b] += c
        if gold == pred:
            bin_correct[b] += 1

    return {
        "n":          bin_n,
        "correct":    bin_correct,
        "conf_sum":   bin_conf_sum,
        "accuracy":   [c / n if n else 0.0 for c, n in zip(bin_correct, bin_n)],
        "mean_conf":  [s / n if n else 0.0 for s, n in zip(bin_conf_sum, bin_n)],
    }


def ece(reliability: Dict[str, list]) -> float:
    n_total = sum(reliability["n"]) or 1
    e = 0.0
    for b in range(len(reliability["n"])):
        n_b = reliability["n"][b]
        if n_b == 0:
            continue
        e += n_b * abs(reliability["accuracy"][b] - reliability["mean_conf"][b])
    return round(e / n_total, 4)


def high_confidence_wrongs(
    records: List[FailureRecord], axis: str = "role", threshold: float = 0.95,
) -> List[FailureRecord]:
    """Records where role_conf ≥ threshold but the axis label is wrong."""
    out = []
    for r in records:
        conf = getattr(r, f"{axis}_conf", None)
        if conf is None or conf < threshold:
            continue
        gold = getattr(r, f"gold_{axis}", None)
        pred = getattr(r, f"pred_{axis}", None)
        if gold is None or pred is None or gold == pred:
            continue
        out.append(r)
    out.sort(key=lambda r: -(getattr(r, f"{axis}_conf", 0.0) or 0.0))
    return out


def calibration_summary(records: List[FailureRecord]) -> Dict:
    """All-in-one calibration summary across axes."""
    summary: Dict = {}
    for axis in ("case", "role", "marker"):
        rb = reliability_bins(records, axis)
        summary[axis] = {
            "reliability": rb,
            "ece":         ece(rb),
            "high_conf_wrong_count":
                sum(1 for r in records
                     if (getattr(r, f"{axis}_conf", 0.0) or 0.0) >= 0.95
                     and getattr(r, f"gold_{axis}") is not None
                     and getattr(r, f"pred_{axis}") is not None
                     and getattr(r, f"gold_{axis}") != getattr(r, f"pred_{axis}")),
        }
    return summary
