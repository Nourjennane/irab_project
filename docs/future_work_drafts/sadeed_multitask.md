# Future-work paragraph — Sadeed / multi-task

Drop into the Future Work section of REPORT.md as-is, after any existing
future-work items. Do not modify wording — it is intentionally measured to
flag the related work without inflating into a contribution claim.

---

**Future work — Sadeed-style fine-tuning and multi-task learning.** Recent work on Arabic diacritization (Sadeed; Aldallal et al., 2025) demonstrates that small fine-tuned Arabic language models can achieve state-of-the-art results when paired with high-quality curated data. Our approach differs in two respects: we address i'rāb generation rather than diacritization, and we use retrieval-augmented generation rather than task-specific fine-tuning. However, the two tasks are structurally connected — the case marker predicted in i'rāb directly determines a word's final vowel — and our self-consistency analysis suggests that grammatical reasoning constrains diacritization in approximately 96% of cases. A natural extension of this work would explore a multi-task formulation that jointly optimizes diacritization and i'rāb objectives, or a Sadeed-style fine-tuning pipeline targeted specifically at i'rāb generation with hand-curated training data. We did not pursue these directions due to time and budget constraints, but note that the structural relationship between the two outputs makes this a promising research avenue.

---

## What this paragraph does well (per directive)

1. Cites Sadeed as related work — methodological-awareness signal for Hovy.
2. Acknowledges the structural connection between diacritization and i'rāb — shows understanding of the task space.
3. Gestures at multi-task learning — one of Hovy's listed research interests.

## What it doesn't do

- Doesn't promise something we didn't deliver.
- Doesn't inflate the project's scope.
- Doesn't conflict with the current evidence base.
- Doesn't require new experiments or data.
