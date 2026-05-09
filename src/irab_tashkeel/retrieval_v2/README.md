# Retrieval v2 (Step 12 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

The frozen baseline's retrieval (Phase R-C and Phase R2 in
`grammar_memory/`) was shallow: cosine over span embeddings,
symbolic family filter, soft logit bias. It produced 0.0 net
metric change with a corrected forward path. Retrieval v2 must
be qualitatively different.

## What retrieval v2 retrieves

Not just analogue *examples*. The retrieval index stores:

- grammatical analyses (full per-sentence iʿrāb structure)
- structural analogues keyed on the dep-tree subgraph + clause
  hierarchy, not just the surface span
- semantic structures (predicate-argument frame patterns)
- reasoning traces (justification chains from the reasoning
  module, Step 9)
- clause graphs (multi-clause skeletons from the grammar graph,
  Step 4)
- discourse patterns (topic-continuation templates)

## What retrieval v2 does NOT do

- soft logit bias (failed in Phase R-C and Phase R2)
- vote-based label override (failed in Phase R2)
- consensus-driven canonical-rule application (failed in Phase R2)

The new approach: retrieval informs the **decoder** (Step 8) by
proposing parse-graph candidates, not the per-word logits. The
decoder ranks candidates against the grammar graph; retrieval
contributes candidates, not bias.

## Required indices

| Index | Key | Stored | Used by |
|---|---|---|---|
| analysis index | sentence embedding + dep skeleton | full parse | decoder candidate generation |
| construction index | construction signature + dep subgraph | construction object | construction module (Step 3) |
| reasoning index | parse skeleton | reasoning trace | reasoning module (Step 9) |
| discourse index | discourse skeleton | rhetorical pattern | discourse module (Step 11) |

## Open design questions

- Multi-modal embedding (surface + dep + semantic) vs separate
  indices per modality?
- What is the retrieval pool's training-time / eval-time
  separation? The frozen baseline used a single pool which
  may have leaked across phases.
- How does retrieval interact with the curriculum (Step 7)? At
  early stages the pool may need to be filtered to construction-
  free analogues.
