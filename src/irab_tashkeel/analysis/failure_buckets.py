"""Slice failure records into structural buckets.

Produces ``Dict[bucket_name -> List[FailureRecord]]`` for the
following axes:

  - long_range_failures        (head_distance ≥ 5)
  - nested_clause_failures     (clause_depth ≥ 2)
  - overlap_failures           (token in ≥ 2 constructions)
  - ambiguity_failures         (sentence semantic_pressure ≥ 2)
  - rare_construction_failures (per-family below threshold)
  - construction_family_<fam>  (one bucket per detected family)
  - calibration_failures       (very_high_conf yet wrong)
  - role_confusions[gold→pred] (per-confusion-type buckets)
  - marker_confusions[gold→pred]
  - case_confusions[gold→pred]

Each bucket is also sorted by descending |role_conf|, so the
"hardest" failures (high-confidence wrong) bubble to the top.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from .failure_analysis import FailureRecord


def bucket_failures(records: List[FailureRecord],
                    rare_construction_threshold: int = 5,
                    ) -> Dict[str, List[FailureRecord]]:
    """Slice records into named buckets.

    ``rare_construction_threshold`` — a construction family whose
    overall failure count is below this is added to ``rare_construction``.
    """
    buckets: Dict[str, List[FailureRecord]] = defaultdict(list)

    fam_counts: Counter = Counter()
    for r in records:
        for f in r.construction_families:
            fam_counts[f] += 1

    for r in records:
        if r.long_range:
            buckets["long_range_failures"].append(r)
        if r.clause_depth >= 2:
            buckets["nested_clause_failures"].append(r)
        if r.overlap:
            buckets["overlap_failures"].append(r)
        if r.semantic_pressure >= 2:
            buckets["ambiguity_failures"].append(r)
        if r.calibration_bucket == "very_high_conf":
            buckets["calibration_failures"].append(r)

        for f in r.construction_families:
            buckets[f"construction_family__{f}"].append(r)
            if fam_counts[f] < rare_construction_threshold:
                buckets["rare_construction_failures"].append(r)

        if r.role_confusion is not None:
            buckets[f"role_confusion__{r.role_confusion}"].append(r)
        if r.case_confusion is not None:
            buckets[f"case_confusion__{r.case_confusion}"].append(r)
        if r.marker_confusion is not None:
            buckets[f"marker_confusion__{r.marker_confusion}"].append(r)

    # Sort by |role_conf| desc so high-confidence wrongs bubble up.
    def _sort_key(r: FailureRecord) -> float:
        return -(r.role_conf or 0.0)

    return {k: sorted(v, key=_sort_key) for k, v in buckets.items()}
