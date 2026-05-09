"""Hard-case mining — combine signals into a single composite score.

Inputs (all per-sentence):

  - uncertainty_score       (from uncertainty_sampling)
  - disagreement_score      (from disagreement_sampling)
  - structural_difficulty   (semantic_pressure + clause_depth + ambiguity)
  - calibration_evidence    (high-confidence wrongs, when known)

Composite::

  score = α₁ * uncertainty
        + α₂ * disagreement
        + α₃ * structural_difficulty
        + α₄ * calibration_evidence

Returns ranked candidate sentence ids. Used to populate
``data_v2/annotation_candidates/queue.jsonl``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


DEFAULT_WEIGHTS = {
    "uncertainty":   0.30,
    "disagreement":  0.30,
    "structural":    0.25,
    "calibration":   0.15,
}


def structural_difficulty(sentence: dict) -> float:
    """0..1 score from existing curriculum metadata."""
    cur = sentence.get("curriculum", {}) or {}
    sp  = float(cur.get("semantic_pressure_score", 0) or 0) / 4.0
    cd  = float(cur.get("clause_depth", 0) or 0) / 5.0
    amb = float(cur.get("ambiguity_score", 0) or 0)
    return min(1.0, sp + cd + amb)


def composite_score(
    sentence: dict,
    *,
    uncertainty: float = 0.0,
    disagreement: float = 0.0,
    calibration: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = weights or DEFAULT_WEIGHTS
    return (
        w["uncertainty"]   * uncertainty
        + w["disagreement"]  * disagreement
        + w["structural"]    * structural_difficulty(sentence)
        + w["calibration"]   * calibration
    )


def rank_candidates(
    sentences: List[dict],
    uncertainty_scores: Dict[str, float],
    disagreement_scores: Dict[str, float],
    *,
    calibration_scores: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 500,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Return [(sentence_id, composite, components)] in score-desc order."""
    out: List[Tuple[str, float, Dict[str, float]]] = []
    for s in sentences:
        sid = s.get("sentence_id", "")
        comps = {
            "uncertainty":  uncertainty_scores.get(sid, 0.0),
            "disagreement": disagreement_scores.get(sid, 0.0),
            "structural":   structural_difficulty(s),
            "calibration":  (calibration_scores or {}).get(sid, 0.0),
        }
        comp = composite_score(s, **{k: v for k, v in comps.items()},
                                weights=weights)
        out.append((sid, comp, comps))
    out.sort(key=lambda x: -x[1])
    return out[:top_k]
