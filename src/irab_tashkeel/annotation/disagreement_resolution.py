"""Multi-annotator disagreement resolution.

When ≥ 2 annotators independently work the same ``AmbiguityExample``,
their `edited.jsonl` records may diverge. This module computes
agreement and surfaces disagreements for a tiebreaker pass.

Strategies:

  - ``majority``           — pick the analysis ≥ 2 annotators agree on
  - ``confidence_weighted`` — weight votes by annotator confidence
  - ``escalate``           — flag for senior reviewer when no majority

The output is a "consolidated" view that downstream training and
evaluation should consume instead of any single annotator's work.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..ambiguity.schema import AmbiguityExample


@dataclass
class Disagreement:
    ambiguity_id: str
    n_annotators: int
    annotator_ids: List[str]
    primary_signatures: List[str]   # one per annotator (for diff display)
    majority_signature: Optional[str]
    needs_escalation: bool


def _signature(example: AmbiguityExample) -> str:
    """A stable string capturing this annotator's primary analysis."""
    parts = []
    for tok_idx in sorted(example.primary_analysis.keys()):
        a = example.primary_analysis[tok_idx]
        parts.append(f"{tok_idx}:{a.case}/{a.role}/{a.marker}/{a.governor_token}")
    return "|".join(parts)


def collect_annotations(root: Path, kind: str
                          ) -> Dict[str, List[Tuple[str, AmbiguityExample]]]:
    """Walk ``edited.jsonl`` for the given kind and group by ambiguity_id."""
    f = Path(root) / kind / "edited.jsonl"
    if not f.exists():
        return {}
    by_id: Dict[str, List[Tuple[str, AmbiguityExample]]] = defaultdict(list)
    for line in f.open():
        d = json.loads(line)
        ex = AmbiguityExample.from_dict(d.get("example", {}))
        by_id[d["ambiguity_id"]].append(
            (d.get("annotator_id", ""), ex)
        )
    return by_id


def resolve_majority(annotations: List[Tuple[str, AmbiguityExample]]
                      ) -> Tuple[Optional[AmbiguityExample], Disagreement]:
    """Majority vote across annotators on the primary-analysis signature."""
    if not annotations:
        return None, Disagreement("", 0, [], [], None, False)

    sigs = [_signature(ex) for _, ex in annotations]
    sig_counts = Counter(sigs)
    top_sig, top_count = sig_counts.most_common(1)[0]

    needs_escalation = (top_count * 2) <= len(annotations)
    chosen = None
    if not needs_escalation:
        for _, ex in annotations:
            if _signature(ex) == top_sig:
                chosen = ex
                break

    annotator_ids = [aid for aid, _ in annotations]
    return chosen, Disagreement(
        ambiguity_id=annotations[0][1].ambiguity_id,
        n_annotators=len(annotations),
        annotator_ids=annotator_ids,
        primary_signatures=sigs,
        majority_signature=top_sig if not needs_escalation else None,
        needs_escalation=needs_escalation,
    )


def consolidate_kind(root: Path, kind: str) -> Dict[str, AmbiguityExample]:
    """Run majority resolution across all annotated examples for one kind.
    Returns a {ambiguity_id: chosen_example} for those that resolved.
    Sentences needing escalation are skipped.
    """
    by_id = collect_annotations(root, kind)
    out: Dict[str, AmbiguityExample] = {}
    for amb_id, anns in by_id.items():
        chosen, dis = resolve_majority(anns)
        if chosen is not None and not dis.needs_escalation:
            out[amb_id] = chosen
    return out
