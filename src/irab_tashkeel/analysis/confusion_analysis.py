"""Per-axis confusion matrices.

For case / role / marker, count (gold, pred) co-occurrences across
the failure-observable rows. Separate from confusion *bucketing*
(``failure_buckets``); this module produces the dense matrix and a
ranked top-N report.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .failure_analysis import FailureRecord


def confusion_matrix(records: List[FailureRecord], axis: str
                     ) -> Dict[str, Dict[str, int]]:
    """Build a Dict[gold][pred] = count for the given axis.

    Records with gold or pred missing on this axis are skipped.
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        gold = getattr(r, f"gold_{axis}")
        pred = getattr(r, f"pred_{axis}")
        if gold is None or pred is None:
            continue
        matrix[gold][pred] += 1
    return {g: dict(d) for g, d in matrix.items()}


def top_confusions(records: List[FailureRecord], axis: str, top_n: int = 20
                   ) -> List[Tuple[str, str, int]]:
    """Return the top-N (gold, pred) wrong pairs by count."""
    c: Counter = Counter()
    for r in records:
        gold = getattr(r, f"gold_{axis}")
        pred = getattr(r, f"pred_{axis}")
        if gold is None or pred is None or gold == pred:
            continue
        c[(gold, pred)] += 1
    return [(g, p, n) for (g, p), n in c.most_common(top_n)]


def confusion_summary(records: List[FailureRecord]) -> Dict[str, dict]:
    return {
        "case":   {
            "matrix": confusion_matrix(records, "case"),
            "top":    top_confusions(records, "case"),
        },
        "role":   {
            "matrix": confusion_matrix(records, "role"),
            "top":    top_confusions(records, "role"),
        },
        "marker": {
            "matrix": confusion_matrix(records, "marker"),
            "top":    top_confusions(records, "marker"),
        },
    }
