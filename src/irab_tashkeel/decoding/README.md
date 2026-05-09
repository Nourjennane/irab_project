# Structured Decoding (Step 8 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

**Important constraint:** this is **not** another trainable CRF.
The frozen baseline established CRF / output-bias / hierarchy
mechanisms plateau at this scale. Step 8 is global *inference-time*
optimisation, not new trainable parameters.

## Mechanisms

- **Beam search** over per-word predictions, with the beam
  scoring a *full sentence parse* against the grammar graph.
- **Candidate generation** — top-k joint (case, role, marker)
  per word, expanded into sentence-level parse candidates.
- **Grammar-consistency reranking** — rerank candidates by
  agreement between predicted roles and the dependency graph
  (e.g., a `mafoul_bih` should be the OBJ child of a verb).
- **Calibration-aware reranking** — penalise high-confidence
  predictions that conflict with the graph; promote moderate-
  confidence predictions that agree with multiple graph
  constraints.
- **Ambiguity resolution** — when multiple parses tie within
  a small score margin, surface the ambiguity in the explanation
  trace rather than hiding it.

## Inputs

- Phase 3-A logits (or its successor's) per word
- Construction objects (Step 3) detected on the sentence
- Grammar graph (Step 4)
- Reasoning trace candidates (Step 9)

## Output

- A full sentence parse: per-word (case, role, marker) + clause
  hierarchy + construction labels.
- An ambiguity flag list when multiple parses score within margin.
- A reasoning trace explaining the chosen parse.

## Open design questions

- Beam width: 5 / 10 / 25 trade-off vs latency.
- How is the grammar-consistency score normalised against the
  per-word log-prob score?
- Should the reranker have learned weights or only heuristic
  weights? (Reminder: trainable rerankers were the failure mode
  of Phase R2 — but Step 8 is supposed to be *non*-trainable
  inference.)
