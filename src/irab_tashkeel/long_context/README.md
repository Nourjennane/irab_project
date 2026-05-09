# Long-Range Reasoning (Step 5 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

The frozen baseline reasons locally — per-word predictions over a
single 320-token window with token-classification heads. Many
Arabic grammatical decisions require longer-range context:

- a kāna construction whose khabar is a multi-word nominal clause
  spanning ≥4 positions
- pronoun antecedent resolution across sentences
- omitted subject reconstruction in idiomatic Quranic verses
- cross-sentence topic continuation affecting case assignment

## Mechanisms to implement

- multi-hop dependency traversal over the grammar graph (Step 4)
- clause graph traversal (clause-id → governing clause-id chain)
- nested-sentence reasoning (embedded nominal/verbal clauses with
  their own iʿrāb structure that bubbles up to the matrix clause)
- discourse memory (cross-sentence state for pronoun antecedent
  resolution and topic continuation)
- omitted-element reconstruction (ḍamīr mustatir, omitted mubtada)
- cross-sentence consistency (e.g., maintained gender on a discourse
  topic)

## Open design questions

- Latency budget at inference: how many hops per word is acceptable?
- Trainable graph attention vs deterministic graph traversal —
  the frozen baseline showed Phase 3.1 (trainable relational
  attention) plateaued, suggesting deterministic traversal as the
  starting point.
- Discourse memory representation: per-sentence vector, structured
  topic-tracker, or persistent graph state?
