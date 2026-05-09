"""Reasoning-trace match metrics for eval_v2 (Step 9 placeholder).

Reasoning supervision is not yet populated in any loader; this
module reserves the metric surface so when reasoning-trace
predictions land, the eval pipeline can score them without code
churn.

Planned metrics
---------------

- **rule_set_overlap** — Jaccard between predicted and gold
  ``derivation_chain`` rules (which named grammatical rules fired)
- **justification_token_f1** — chrF-style fuzzy match on the
  ``justification`` text
- **alternative_count_match** — does the predicted reasoning trace
  surface the same number of alternatives as gold
- **transformation_logic_match** — exact match on the canonical
  transformation rule (e.g., "kana_completion: ism→raf, khabar→nasb")

Until predictors emit these, the metric returns zeros gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..data_v2.schema_v2 import Sentence


@dataclass
class ReasoningReport:
    n_gold_steps:                  int = 0
    n_predicted_steps:             int = 0
    n_matched_rules:               int = 0
    rule_set_overlap_jaccard:      float = 0.0
    n_correct_transformation_logic: int = 0
    transformation_logic_match_rate: float = 0.0


def reasoning_match(
    sentences: Iterable[Sentence],
    # In the future, the second arg will be a list of predicted
    # reasoning traces; for now we just inspect the gold side.
    *,
    predictions=None,
) -> ReasoningReport:
    """Reserved metric surface; activates when reasoning-trace
    predictors exist.

    Currently returns counts of gold reasoning steps so corpus
    quality can be tracked even before predictors land.
    """
    n_gold_steps = 0
    for s in sentences:
        n_gold_steps += len(s.reasoning_steps)

    return ReasoningReport(n_gold_steps=n_gold_steps)
