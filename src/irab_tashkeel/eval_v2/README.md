# Evaluation v2 (Step 13 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

The frozen baseline's evaluation is per-word case / role / marker /
fully on Gazelle and MASAQ. Useful, but insufficient for a system
that reasons over clauses, constructions, and discourse.

## New evaluation axes

| Axis | What it measures |
|---|---|
| clause-level correctness | does the predicted clause hierarchy match gold? |
| construction-level correctness | are detected constructions correct (boundary + family + head) |
| reasoning correctness | does the predicted reasoning trace match gold rules |
| consistency | do per-word predictions agree with the predicted parse |
| calibration | gap between confidence and correctness, broken down by construction |
| ambiguity robustness | does the system surface alternatives when multiple parses are plausible |
| long-range dependency accuracy | does the system get cross-clause case assignment right |
| discourse consistency | do pronoun resolutions and topic continuations align across sentences |
| nested-construction accuracy | accuracy on constructions of depth ≥ 2 |

## Evaluation suites

- **Hard subsets** drawn from existing datasets — the failing 0%
  Gazelle constructions (istithnāʾ, quranic_proxy) become first-
  class evaluation targets, not aggregate metrics.
- **Adversarial grammar sets** — sentences crafted to test
  long-range agreement, omitted-element reconstruction, ambiguous
  attachment.
- **Ambiguity stress tests** — sentences known to admit multiple
  defensible parses; the system is scored on whether it surfaces
  the alternatives.
- **Rare-construction challenge suites** — per-family targeted
  evaluations with adequate sample size for paired statistical
  tests.

## Inheriting the frozen baseline's eval

`evaluation/structural.py` (with the kāna-aware role extraction)
remains the gold extractor for legacy comparison. Any new
evaluator subclasses or extends it.

## Open design questions

- How is reasoning correctness measured without becoming
  brittle to surface-form variation?
- What is the minimum sample size per construction for
  paired statistical claims? (Frozen baseline floor: ±7 pp on
  n=134 binary metrics.)
- How are ambiguity-robustness scores aggregated across the
  evaluation set?
