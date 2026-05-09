# Construction Schemas (Step 3 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

This module will implement first-class construction objects with
explicit span / head / children / clause-membership / agreement /
ambiguity slots, replacing the surface-particle detection used in
the frozen baseline.

## Target construction families

- kāna sisters (12 particles)
- inna sisters (modal + assertive subgroups)
- istithnāʾ (illa, ism, ma-3ada, hasha)
- iḍāfa (single + multi-level)
- relative clauses (mawṣūl, definite + indefinite)
- conditionals
- vocatives (munādā)
- coordination (ʿaṭf)
- apposition (badal, ʿaṭf bayān)
- embedded nominal clauses
- embedded verbal clauses
- ellipsis (omitted subjects, omitted predicates)
- Quranic-specific constructions

## Required object structure

```python
@dataclass
class Construction:
    family: str                   # canonical name
    subgroup: str                 # particle group / variant
    span: Tuple[int, int]         # token-level start, end_excl
    head_idx: int                 # syntactic head within span
    children: List[int]           # dependent token indices
    clause_id: Optional[int]      # which clause this construction belongs to
    semantic_role: Optional[str]  # event-level role
    agreement_relations: List[Tuple[int, int]]   # word-pair agreement constraints
    ambiguity_score: float        # 0..1; higher = more competing parses
    alt_parses: List["Construction"]   # alternative analyses to score
```

The frozen baseline's `signature.py` is *not* this — it stores
flat detection records keyed by particle surface. The new
construction object has nested clause membership and explicit
ambiguity tracking.

## Open design questions

- How do constructions interact with the grammar graph (Step 4)?
  Likely: each construction is a sub-graph annotation over the same
  shared word/clause graph, not a duplicate structure.
- How does construction detection interact with parser confidence?
  Stanza UAS ≈ 84% means head/children may be wrong; the construction
  object should expose `parser_confidence` and provide a fallback
  detector (rule-based on surface particle + neighbourhood).
- How should `alt_parses` be enumerated? Likely top-k from a
  candidate generator at decode time (Step 8).
