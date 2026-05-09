"""Stratified ``fully`` accuracy by structural axis.

Operates on `TokenOutcome` rather than `FailureRecord` because we
need the correct rows too (for the denominator). Returns:

  {
    "by_dep_depth":     {0: {n, n_correct, fully}, 1: {...}, ...},
    "by_clause_depth":  {0: {...}, 1: {...}, 2: {...}, ...},
    "by_sent_length":   {"<10": {...}, "10-19": {...}, "20-29": {...}, "30+": {...}},
    "by_overlap":       {True: {...}, False: {...}},
    "by_long_range":    {True: {...}, False: {...}},
    "by_construction_family": {family: {...}, ...},
    "by_semantic_pressure":   {0: {...}, 1: {...}, 2: {...}, ...},
  }

Useful for the headline "where does the model break" figure.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..eval_v2 import TokenOutcome


def _slot(n: int, n_correct: int) -> Dict:
    return {"n": n, "n_correct": n_correct,
            "fully": round(n_correct / max(n, 1), 4)}


def _aggregate_by(outcomes: List[TokenOutcome], key_fn) -> Dict:
    n_total: Dict = defaultdict(int)
    n_correct: Dict = defaultdict(int)
    for o in outcomes:
        if o.fully_correct is None:
            continue
        k = key_fn(o)
        n_total[k] += 1
        if o.fully_correct:
            n_correct[k] += 1
    return {k: _slot(n_total[k], n_correct[k]) for k in n_total}


def structural_breakdown(outcomes: List[TokenOutcome]) -> Dict[str, Dict]:
    def _len_bucket(o):
        L = o.sentence_length
        if L < 10:  return "<10"
        if L < 20:  return "10-19"
        if L < 30:  return "20-29"
        return "30+"

    fully_obs = [o for o in outcomes if o.is_fully_observable]

    out: Dict[str, Dict] = {}
    out["by_dep_depth"]    = _aggregate_by(fully_obs, lambda o: o.dependency_depth)
    out["by_clause_depth"] = _aggregate_by(fully_obs, lambda o: o.clause_depth)
    out["by_sent_length"]  = _aggregate_by(fully_obs, _len_bucket)
    out["by_overlap"]      = _aggregate_by(
        fully_obs, lambda o: bool(len(o.construction_families) >= 2)
    )
    out["by_long_range"]   = _aggregate_by(
        fully_obs, lambda o: bool(o.dependency_depth >= 5)
    )
    out["by_semantic_pressure"] = _aggregate_by(
        fully_obs, lambda o: o.semantic_pressure
    )

    # Construction-family breakdown — one bucket per family the token
    # participates in (a token in N families contributes to all N).
    fam: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "n_correct": 0}
    )
    for o in fully_obs:
        if o.fully_correct is None:
            continue
        for f in o.construction_families:
            fam[f]["n"] += 1
            if o.fully_correct:
                fam[f]["n_correct"] += 1
    out["by_construction_family"] = {
        f: _slot(d["n"], d["n_correct"]) for f, d in fam.items()
    }

    return out
