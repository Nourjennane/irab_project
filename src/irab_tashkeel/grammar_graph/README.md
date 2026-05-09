# Grammar Graph Engine (Step 4 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

The frozen baseline operates on flat per-word predictions. The
next-generation system needs a unified graph representation that
makes long-range and cross-clause grammatical interactions explicit
and traversable.

## Nodes

- words
- phrases
- clauses
- constructions (kāna span, iḍāfa chain, relative clause, …)
- discourse units (sentences, paragraphs)

## Edges

- dependency (UD-style head-dependent, typed)
- semantic (predicate-argument)
- agreement (gender / number / definiteness pairing)
- governor-dependent (case-assignment edges, e.g. preposition → noun)
- clause-membership (clause-id → word-id, construction-id → clause-id)
- co-reference (pronoun → antecedent, ḍamīr → marjiʿ)
- discourse continuation (cross-sentence)
- rhetorical relations (cause, contrast, elaboration, …)

## Why a graph

Three classes of grammatical phenomena cannot be captured by per-word
prediction:

1. **Long-range agreement** — adjective-to-noun agreement across
   intervening modifiers requires walking the dep tree.
2. **Cross-clause case assignment** — the case of a word in an
   embedded clause may depend on a particle three clauses up.
3. **Discourse-level role assignment** — pronoun resolution and
   topic-continuation effects span sentences.

## Required interfaces

```python
class GrammarGraph:
    def add_word(self, idx: int, surface: str, morph: Dict): ...
    def add_dep_edge(self, head: int, dep: int, label: str, conf: float): ...
    def add_semantic_edge(self, predicate: int, arg: int, role: str): ...
    def add_agreement_edge(self, w1: int, w2: int, axes: List[str]): ...
    def add_construction(self, span, family, head_idx, children): ...
    def assign_to_clause(self, word_idx: int, clause_id: int): ...

    def walk_to_governor(self, word_idx: int, max_hops: int = 3) -> List[int]: ...
    def find_construction_at(self, word_idx: int) -> Optional[Construction]: ...
    def cross_clause_path(self, w1: int, w2: int) -> Optional[List[Tuple[int, str]]]: ...
```

## Integration points

- Construction module (Step 3) populates the construction nodes.
- Long-context module (Step 5) traverses the graph for multi-hop reasoning.
- Decoding module (Step 8) scores parse candidates against graph consistency.
- Eval module (Step 13) computes graph-level correctness metrics.

## Open design questions

- In-memory dict-of-edges vs `networkx` vs custom sparse tensor representation?
- How are edge weights / confidences propagated during inference?
- How does the graph interact with batch processing for training?
