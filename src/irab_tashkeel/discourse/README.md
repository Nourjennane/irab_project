# Discourse Reasoning (Step 11 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

Future gains on Quranic-style and classical Arabic require
discourse context: many decisions in those registers cannot be
made within a single sentence.

## Capabilities

- sentence linkage (which sentences form a discourse unit)
- topic continuation (what entity is the discourse "about")
- pronoun resolution / ḍamīr marjiʿ tracking
- discourse-aware grammatical disambiguation (e.g., a relative
  clause's antecedent may be in the previous sentence)
- rhetorical relation tracking (cause / contrast / elaboration)

## Why discourse matters

- Quranic verses often have implicit topic-continuation across
  ayāt that shifts case patterns (e.g., maintained subject =
  raf-marfū assignment)
- ḍamīr antecedent resolution is intra- or inter-sentential and
  determines the case the pronoun's antecedent should bear
- discourse relations alter the semantic role assignments of
  arguments

## Open design questions

- Cross-sentence batching at training time — how to feed multi-
  sentence context to a 320-token window? Concatenation with
  separator? Hierarchical encoder?
- Where does the discourse state live — in the grammar graph
  (Step 4) as persistent edges, or in a separate discourse-memory
  structure?
