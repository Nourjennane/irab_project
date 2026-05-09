"""Diversity-based active-learning sampling.

Goal: avoid annotating 100 sentences that all express the same
construction template. We want the annotation queue to **cover the
structural distribution**.

Approach: greedy maximum-coverage over sentence "signatures":

  signature(s) = (
      coarse_dep_pattern   — first 8 deprels joined,
      coarse_construction_signature — sorted families,
      length_bucket        — <10 / 10-19 / 20-29 / 30+,
  )

Pick sentences one at a time, each time choosing the highest-uncertainty
sentence whose signature has not yet been seen (or has been seen
fewest times).

Returns ranked sentence_ids in pick order.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple


def _length_bucket(n_tokens: int) -> str:
    if n_tokens < 10: return "<10"
    if n_tokens < 20: return "10-19"
    if n_tokens < 30: return "20-29"
    return "30+"


def signature(sentence: dict) -> Tuple[str, str, str]:
    """Coarse triple-signature for grouping similar sentences."""
    tokens = sentence.get("tokens", [])
    dep_pat = ">".join(
        (t.get("dep_label", {}) or {}).get("value", "_")
        for t in tokens[:8]
    )
    fam_sig = "+".join(sorted({c.get("family", "")
                                for c in sentence.get("constructions", [])})) or "none"
    length = _length_bucket(len(tokens))
    return (dep_pat, fam_sig, length)


def diversity_rank(
    sentences: List[dict],
    sentence_scores: Dict[str, float],
    k: int = 200,
) -> List[Tuple[str, float, Tuple[str, str, str]]]:
    """Greedy max-coverage: pick top-k sentences by score, breaking
    redundancy by upweighting unseen signatures.

    Returns [(sentence_id, score, signature), ...] in pick order.
    """
    by_id = {s.get("sentence_id"): s for s in sentences}
    seen_sigs: Counter = Counter()
    picked: List[Tuple[str, float, Tuple[str, str, str]]] = []

    # Sort by base uncertainty desc; iterate and accept if signature is
    # rare among already-picked.
    ordered = sorted(by_id.keys(),
                      key=lambda sid: -sentence_scores.get(sid, 0.0))

    for sid in ordered:
        if len(picked) >= k:
            break
        s = by_id.get(sid)
        if not s:
            continue
        sig = signature(s)
        # Penalty proportional to how many times this signature was already
        # picked. Effective score = base − 0.05 * count.
        base = sentence_scores.get(sid, 0.0)
        eff = base - 0.05 * seen_sigs[sig]
        if eff < 0:
            continue
        picked.append((sid, base, sig))
        seen_sigs[sig] += 1

    return picked
