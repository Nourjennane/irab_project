# Schema v2 — Canonical Supervision Specification

**Version:** 2.0.0
**Module:** `src/irab_tashkeel/data_v2/schema_v2.py`
**Status:** authoritative supervision format for the
`nextgen-grammatical-reasoning` branch.

This document is the source of truth for what every loader must
produce, what every trainer + evaluator must consume, and how the
on-disk JSONL is structured. Anyone adding a new source corpus,
new evaluation metric, or new training stage works against this
spec.

The schema is consciously verbose: the empirical Step 16 result
that **annotation sparsity dominates 93.5% of frozen-baseline
errors** means future work depends on making *more* annotation
fields, with *richer* provenance, trackable end-to-end. We optimise
for permanence, not minimalism.

---

## 1. Top-level: `Sentence`

A single training/evaluation/inference unit. Serialises to one
JSONL line.

```python
Sentence(
    schema_version: "2.0.0",          # pin to detect on-disk migrations
    sentence_id: str,                  # UUID
    raw_text: str,                     # unmodified surface
    normalized_text: str,              # NFC + diacritic-stripped + whitespace-collapsed

    tokens: List[Token],               # Layer A + B + C per-token labels
    spans: List[Span],                 # Layer B phrasal spans
    clauses: List[Clause],             # Layer B/C clause hierarchy
    constructions: List[Construction], # Layer C construction occurrences
    graph: Optional[GrammarGraph],     # Step 4 grammar graph
    discourse_links: List[DiscourseLink],  # Step 11 cross/intra-sentence links
    reasoning_steps: List[ReasoningStep],  # Step 9 explanation chains

    metadata: SentenceMetadata,        # provenance + quality
    curriculum: CurriculumMetadata,    # Step 7 curriculum scheduling
    completeness: AnnotationCompleteness,  # which Layers are populated
)
```

Every field is optional except `tokens`. A schema-conformant
sentence with an empty `constructions`, `graph`, `discourse_links`,
and `reasoning_steps` is fine — it represents the current frozen-
baseline level of supervision and provides a clean migration
target for new sources to replace fields incrementally.

---

## 2. Per-token labels: `Token`

```python
Token(
    index: int,                        # 0-based; primary key for cross-references
    surface: str,                      # original surface form (with diacritics)
    normalized: str,                   # arabic_normalize(surface) — for retrieval
    char_start: int, char_end: int,    # offsets into normalized_text

    # Layer A — Morphology
    morph: Morphology,                 # gender / number / person / definite / pos /
                                       # mood / voice / aspect / inflection / derivation
    pos: LabelTag,                     # canonical POS — separate axis from morph.pos
                                       # so multi-source POS can disagree

    # Layer B — Local syntax
    dep_head_idx: int,                 # 0-based; -1 = unset, -2 = root
    dep_label: LabelTag,               # UD DEPREL or comparable
    governor_pos: Optional[str],

    # Layer C — Iʿrāb
    case: LabelTag,                    # raf / nasb / jarr / jazm / mabni
    role: LabelTag,                    # canonical role taxonomy (25 + extensions)
    marker: LabelTag,                  # damma_visible / fatha_visible / ya / ...

    # Layer C — Semantic role (PropBank-style; future)
    semantic_role: LabelTag,

    notes: List[str],                  # free-form annotation notes
)
```

### LabelTag — every annotation carries provenance

```python
LabelTag(
    value: Optional[str],              # the canonical label, or None
    source: str,                       # provenance string (see §6)
    confidence: float,                 # 0..1
    alternatives: List[Tuple[str, float]],   # competing values (case ambiguity, ...)
    notes: str,                        # optional
)
```

`value=None` is meaningful: it distinguishes "this field was not
annotated by this source" from "this field was annotated with the
empty string". Loaders must use `LabelTag(value=None)` (the
default) for unannotated fields, never `LabelTag(value="")`.

---

## 3. Spans: `Span`

Phrase-level constituent annotation. Optional — a sentence with
only Layer A + B token-level annotation has `spans=[]`.

```python
Span(
    span_id: str,                      # UUID
    token_indices: List[int],          # contiguous token indices
    span_type: str,                    # "NP" / "VP" / "PP" / "SBAR" / ...
    head_idx: Optional[int],           # token index INTO the sentence
    source: str,
    confidence: float,
    notes: str,
)
```

---

## 4. Clauses: `Clause`

Clause-level structure with explicit nesting. Required for
Step 5 long-context reasoning and Step 11 discourse processing.

```python
Clause(
    clause_id: str,                    # UUID
    token_indices: List[int],          # tokens belonging to this clause
    clause_type: ClauseType,           # MATRIX / NOMINAL_EMBEDDED / RELATIVE / ...
    parent_clause_id: Optional[str],   # tree parent
    head_idx: Optional[int],
    role_in_parent: Optional[str],     # "khabar" / "naat" / "subj" / ...
    depth: int,                        # 0 = matrix, 1 = once-embedded, ...
    source: str,
    confidence: float,
)
```

---

## 5. Constructions: `Construction`

First-class construction objects — replacing the flat
`signature.py::ConstructionInstance` of the frozen baseline.

```python
Construction(
    construction_id: str,              # UUID
    family: str,                       # kana_sisters / inna_sisters / istithna / idafa /
                                       # idafa_multi / mawsool / quranic_proxy / ...
    subgroup: str,                     # particle group within family
    token_indices: List[int],
    head_idx: Optional[int],
    children_indices: List[int],
    particle_surface: str,
    clause_id: Optional[str],          # which clause this construction lives in
    semantic_role: Optional[str],
    agreement_relations: List[Tuple[int, int, List[str]]],
                                       # (token_a, token_b, axes)
    ambiguity_score: float,
    alternative_analyses: List[Dict],
    source: str,
    confidence: float,
    notes: str,
)
```

The `alternative_analyses` field carries competing parses for the
same span — this is the structural slot for Step 8 decoding to
surface ambiguity rather than hide it.

---

## 6. Provenance & quality: `SentenceMetadata`

```python
SentenceMetadata(
    domain: Domain,                    # MSA_NEWS / QURANIC / CLASSICAL / ...
    source: str,                       # corpus identifier (e.g. "distill_v2")
    source_id: str,                    # within-corpus row id
    annotation_quality: AnnotationQuality,
    parser_origin: str,                # primary parser (e.g. "stanza_ud")
    morph_origin: str,
    dep_origin: str,
    role_origin: str,
    marker_origin: str,
    construction_origin: str,
    reasoning_origin: str,
    license: str,
    ingestion_timestamp: str,          # ISO8601
)
```

### Annotation-quality tiers

| Tier | Description |
|---|---|
| `GOLD_HUMAN`              | hand-annotated by a domain expert |
| `GOLD_TREEBANK`           | from a published treebank (UD-PADT, CamelTB, EQT) |
| `SILVER_LLM_DISTILL`      | distilled from a teacher LLM (Haiku, Sonnet, …) |
| `SILVER_PARSER_HIGH_CONF` | parser output with confidence ≥ 0.8 |
| `BRONZE_PARSER_LOW_CONF`  | parser output with confidence < 0.8 |
| `BRONZE_HEURISTIC`        | rule-based / heuristic detection |
| `UNKNOWN`                 | unset; loaders must override |

### Provenance source strings (canonical)

| Source string | Meaning |
|---|---|
| `gold_human`             | hand annotation |
| `gold_treebank`          | treebank gold |
| `silver_llm_distill`     | teacher LLM distillation (set per teacher: `haiku_distill`, `sonnet_distill`) |
| `silver_stanza_ud`       | Stanza UD parser |
| `silver_madamira`        | MADAMIRA |
| `silver_camelparser`     | CamelParser2 |
| `bronze_heuristic`       | rule-based detector |
| `bronze_extractor_v2`    | the kana-aware structural extractor |

Loaders MUST use canonical source strings; do not invent new
sources without updating this table.

---

## 7. Curriculum metadata: `CurriculumMetadata`

Computed by post-loader passes in `data_v2.metadata.difficulty`,
not by loaders. Holds the structural-difficulty signals for the
Step 7 curriculum scheduler.

```python
CurriculumMetadata(
    difficulty_level: int,             # 1..7 (curriculum stage)
    dependency_depth: int,
    clause_depth: int,
    construction_count: int,
    nested_construction_count: int,
    ambiguity_score: float,
    semantic_pressure_score: int,      # 0..3
    discourse_complexity: float,
    sentence_length_tokens: int,
    nested_clause_count: int,
)
```

Curriculum stages (defined in `src/irab_tashkeel/curriculum/README.md`):

| Stage | Focus |
|---|---|
| 1 | pure morphology |
| 2 | local syntax |
| 3 | simple constructions |
| 4 | nested syntax |
| 5 | semantic interactions |
| 6 | discourse-sensitive structures |
| 7 | Quranic / classical complexity |

---

## 8. Annotation completeness: `AnnotationCompleteness`

```python
AnnotationCompleteness(
    has_morph: bool,
    has_dep: bool,
    has_role: bool,
    has_marker: bool,
    has_constructions: bool,
    has_clauses: bool,
    has_reasoning: bool,
    has_graph: bool,
    has_discourse: bool,
    has_alternative_parses: bool,
    fields_complete_pct: float,        # 0..1, fraction of (case, role, marker)
)
```

The eval engine uses these flags to stratify metrics by what the
gold actually contains — the key insight from Step 16 was that
~93.5% of frozen-baseline "errors" had at least one missing gold
field. Eval v2 will report metrics separately on the
`fields_complete_pct == 1.0` subset (the model's true error rate)
and the full set.

---

## 9. Graph + Discourse + Reasoning placeholders

Empty in most current loaders. Populated by future steps:

- **`graph: GrammarGraph`** — Step 4
- **`discourse_links: List[DiscourseLink]`** — Step 11
- **`reasoning_steps: List[ReasoningStep]`** — Step 9

The slots exist now so when those steps land, no schema migration
is needed; loaders just populate the existing fields.

---

## 10. JSONL on-disk format

```bash
# One Sentence per line; `to_json()` writes / `from_json()` reads
data_v2/annotated/<source>/train.jsonl
data_v2/annotated/<source>/val.jsonl
data_v2/annotated/<source>/test.jsonl
```

Empty fields and default `LabelTag(value=None)` records are
omitted in `to_dict()` to keep on-disk size manageable. Forward
migration in `_migrate()` handles older schema versions.

---

## 11. Loader contract

Every loader subclasses `BaseLoader` (in
`data_v2/loaders/base.py`):

```python
class MyCorpusLoader(BaseLoader):
    source_id          = "my_corpus_v1"
    domain             = Domain.MSA_NEWS.value
    annotation_quality = AnnotationQuality.SILVER_LLM_DISTILL.value
    parser_origin      = "stanza_ud"

    def iter_raw(self) -> Iterator[Dict]: ...
    def normalize_row(self, raw, idx) -> Optional[Sentence]: ...
```

Use the `@register_loader` decorator so the loader is discoverable
by `data_v2.loaders.base.get_loader(source_id)`.

---

## 12. Layer-completion roadmap

The schema is fully populated when these data sources land:

| Layer | Status | Target source |
|---|---|---|
| A morph (per-token) | partial | UD-PADT, MADAMIRA — both available |
| B local syntax (dep) | partial | Stanza UD — already used |
| B phrasal spans | empty | constituency conversion from UD (Step 4) |
| B/C clause hierarchy | empty | Stanza + clause-detector (Step 5) |
| C construction occurrences | partial | construction detector (Step 3) |
| C semantic role | empty | Arabic PropBank / SALMA |
| Graph | empty | grammar-graph engine (Step 4) |
| Discourse links | empty | discourse-annotated corpora (Step 11) |
| Reasoning steps | empty | LLM-generated traces + textbook scrapes (Step 9) |

---

## 13. Migration policy

Bump `SCHEMA_VERSION` when:

- Removing or renaming a field
- Changing the meaning of a field
- Adding a required field with no default

Adding optional fields with sensible defaults does NOT require a
version bump (it's forward-compatible).

When you bump the version, add migration logic in
`schema_v2._migrate()` and update this document's §13.
