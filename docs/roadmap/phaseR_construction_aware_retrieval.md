# Phase R — Construction-Aware Retrieval + Grammar Memory

> **The new core phase, replacing what was previously Phase 9 (grammar-
> memory expansion).** Per the project pivot 2026-05-09 after the
> per-construction baseline revealed that the bottleneck is *specific
> sparse construction families* (kāna 0% Gazelle fully, istithnāʾ 0%
> fully, calib gap −0.533 on kāna), not general weakness. Aim:
> transform the system from "structured classifier" into "hybrid
> neuro-symbolic grammatical reasoning system" via inference-time
> retrieval of analogous training-corpus constructions.

## 0. Strategic frame

The five-cell architectural case study + Phase 3-A's per-construction
baseline + the Phase #39 partial-positive together establish:

- **WORKS:** orthogonal information (morphology, dep features),
  distribution-matched supervision
- **FAILS / PLATEAUS:** hierarchy, conditioning, decoder restructuring,
  CRF reshaping, broad synthetic dominance

Phase R is the canonical "add new orthogonal information at inference
time" intervention. The retrieval pool *is* a new information source
(specific training-corpus instances similar to the test instance —
something the encoder cannot retrieve from its parametric memory).
Soft evidence, never hard-imposed.

## 1. Architecture overview

```
                   test sentence
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
       AraT5v2        Stanza UD      construction
       encoder         parser          detector
       (Phase 3-A)                   (kana/inna/...)
           │             │              │
           ▼             ▼              ▼
       per-word     dep features    construction
       features                     spans (start, end, family, particle)
                                          │
                                          ▼
                                construction signature
                                (family, particle, head_morph,
                                 head_deprel, …)
                                          │
                       ┌──────────────────┴───────────────┐
                       ▼                                  ▼
              symbolic retrieval                vector retrieval
              (filter by family +               (cosine top-k on
               particle group)                  encoder embedding,
                       │                         restricted to symbolic-
                       ▼                         filtered set)
                top-k analogues
                       │
                       ▼
              per-word retrieval prior
              (soft prior over case/role/marker
               aggregated from analogues, weighted
               by similarity scores)
                       │
                       ▼
              Phase 3-A logits + λ · log(prior)
                       │
                       ▼
                final prediction
                       │
                       ▼
              explanation trace:
              top-k retrieved examples,
              symbolic+vector scores,
              construction match strength
```

**Key design choices:**

- **Inference-time only** for the first pass. Phase 3-A stays the
  production checkpoint; retrieval is an inference-side wrapper. This
  matches the Phase 2 / 5 / 6 lesson: don't retrain the encoder around
  a moving target.
- **Hybrid symbolic + vector retrieval**. Symbolic filter (must match
  construction family + particle group) prunes the candidate pool;
  vector cosine ranking selects the most-similar analogues *within*
  that symbolic subset.
- **Soft additive bias on logits**, NOT a hard replacement. The
  retrieval prior is added to the Phase 3-A logits with a small
  multiplier `λ`. At `λ = 0` we recover Phase 3-A exactly.
- **Retrieval pool is the iʿrāb-supervised training corpus**
  (4,750 distill_v2 sentences). UD-PADT sentences have no iʿrāb
  labels and are not indexed. Phase 39 augmented sentences are
  excluded from the pool to avoid leakage between aug v2 experiments
  and retrieval (separately addressable).

## 2. Construction signature schema

Per-construction-instance record (one per construction occurrence in
the training corpus):

```python
@dataclass
class ConstructionInstance:
    instance_id: str               # e.g. "distill_v2_train_03421_word_2_4"
    sentence: str                   # full source sentence
    sentence_idx: int               # index into the training corpus
    construction: str              # one of FAMILIES (see §3)
    span: Tuple[int, int]          # word-level (start, end+1) of construction
    particle_surface: str           # e.g. "أصبح" or None
    particle_family: str            # e.g. "kana_inceptive" (subgroup of kana)
    head_morph: Dict[str, str]      # gender/number/definite/case of construction head
    head_deprel: str                # DEPREL of the construction head in dep tree
    head_governor_upos: str         # UPOS of the head's governor
    sentence_length: int
    items: List[Dict]               # full per-word labels for the span
    embedding: np.ndarray           # 768-d encoder embedding of the span (mean-pooled)
    confidence: float               # quality flag (gold completeness 0..1)
```

**Construction families and particle groups (frozen):**

```python
FAMILIES = {
    "kana_sisters": {
        "particle_groups": {
            "kana_completion":     ["كان", "صار", "أصبح", "أمسى", "أضحى", "بات", "ظل"],
            "kana_negation":       ["ليس", "ما زال", "ما برح", "ما فتئ", "ما انفك"],
        },
    },
    "inna_sisters": {
        "particle_groups": {
            "inna_assertion":      ["إن", "أن", "إنّ", "أنّ"],
            "inna_modal":          ["ليت", "لعل", "كأن", "لكن", "كأنّ", "لكنّ"],
        },
    },
    "istithna": {
        "particle_groups": {
            "illa":                ["إلا"],
            "istithna_noun":       ["غير", "سوى"],
            "ma_3ada_phrase":      ["ما عدا", "ما خلا"],
            "hasha":               ["حاشا"],
        },
    },
    "mawsool": {
        "particle_groups": {
            "definite_relative":   ["الذي", "التي", "الذين", "اللاتي", "اللواتي",
                                    "اللذان", "اللتان", "اللذين", "اللتين"],
            "indefinite_relative": ["من", "ما"],
        },
    },
    "idafa":         { "particle_groups": { "any": [] } },
    "idafa_multi":   { "particle_groups": { "any": [] } },
    "quranic_proxy": {
        "particle_groups": {
            "qad_idh":    ["قد", "إذ", "إذا"],
            "lamma":      ["لما", "لمّا"],
            "kullama":    ["كلما", "كلّما"],
            "hatta":      ["حتى"],
        },
    },
}
```

The particle subgrouping matters for retrieval quality: a kāna-completion
sentence (*أصبح*) shouldn't be retrieved as analogue for a kāna-negation
sentence (*ليس*) — they have different syntactic behaviour even though
both are technically "kāna sisters".

## 3. Retrieval signature similarity

Two-stage scoring:

### 3.1 Symbolic match (binary filter)

A candidate `c` matches a query `q` if:
- `c.construction == q.construction`, AND
- `c.particle_family == q.particle_family` (when both have particles), AND
- (optional) `c.head_morph["definite"] == q.head_morph["definite"]`
  if `q.head_morph["definite"]` is set

Anything failing the symbolic filter is dropped from the candidate pool.

### 3.2 Vector ranking (within symbolic-matched set)

Cosine similarity between `q.embedding` and `c.embedding` (768-d
encoder span embeddings). Top-k by cosine, k = 5 default.

**Retrieval confidence score:**

```
score(c, q) = α · symbolic_overlap(c, q) + (1 - α) · cosine(c.embedding, q.embedding)
```

Where `symbolic_overlap` is a 0..1 fraction of matching categorical
fields (head_morph, head_deprel, governor_upos). `α = 0.3` default
(symbolic small, vector dominant after the hard filter).

## 4. Retrieval prior aggregation

For a query span `q` with retrieved top-k `[c_1, …, c_k]`:

For each word position `i` in the query span, look at the
corresponding word position in each retrieved instance. Aggregate
soft priors over (case, role, marker) labels:

```python
prior_case[i, label] = Σ_j (score(c_j, q) · 1{c_j.items[i].case == label})
                       / Σ_j score(c_j, q)
```

Same for role and marker. Output: `prior_case ∈ R^{W × N_case}`,
`prior_role ∈ R^{W × N_role}`, `prior_marker ∈ R^{W × N_marker}` —
soft probability distributions over each label space.

## 5. Logit integration

Phase 3-A produces raw logits `(case_logits, role_logits, marker_logits)`.
Phase R adds:

```python
case_logits_final = case_logits + λ * log(prior_case + ε)
role_logits_final = role_logits + λ * log(prior_role + ε)
marker_logits_final = marker_logits + λ * log(prior_marker + ε)
```

`ε = 1e-6` for numerical stability. `λ` is a global hyperparameter
tuned on val set, default 0.3.

When no retrievals are found (small symbolic-filtered set), `λ` is
set to 0 for that span and Phase 3-A's prediction passes through
unchanged.

## 6. FAISS / vector integration

- Index type: `IndexFlatIP` (inner product = cosine on normalised
  vectors) for the first pass. Pool size ~14K instances → no need
  for IVF/PQ approximation; brute-force is fast enough.
- Embeddings: span mean-pool of Phase 3-A `pooled_irab` features,
  L2-normalised before indexing.
- Storage: one `.faiss` file + one `.jsonl` metadata file per
  retrieval pool. Stored in `data/grammar_memory/`.
- Per-construction-family separate sub-indices: when symbolic filter
  is `construction == "kana_sisters"`, only search the kana sub-index
  (avoids wasted comparisons against the istithnāʾ pool).

## 7. Memory storage format

```
data/grammar_memory/
    kana_sisters/
        instances.jsonl              # per-instance metadata
        embeddings.faiss             # FAISS index (kana subset)
    inna_sisters/
        instances.jsonl
        embeddings.faiss
    istithna/
        ...
    mawsool/
        ...
    idafa/
        ...
    quranic_proxy/
        ...
    _build_summary.json              # build provenance + counts
```

## 8. Build pipeline

```
data/morph_v1_dep/train.jsonl  (Phase 3-A training corpus, dep-parsed)
                │
                ▼  scripts/grammar_memory/build_memory.py
                │
                │  for each sentence:
                │    1. detect constructions (reuse eval_per_construction.py logic)
                │    2. for each construction span:
                │       - extract signature (particle, head_morph, head_deprel, …)
                │       - encode span with Phase 3-A encoder (mean-pool pooled_irab)
                │       - assign instance_id, write to family JSONL
                ▼
data/grammar_memory/{family}/instances.jsonl + embeddings.faiss
```

Compute estimate: 4,750 distill_v2 sentences × 1 forward pass
(GPU) ≈ 5 min. Per-sentence construction detection ≈ 1 ms (pure CPU
regex + role check). FAISS index build ≈ 10 sec for 14K vectors.
**Total: ~10 min HPC.**

## 9. Inference-time retrieval flow

```
def predict_with_retrieval(sentence, predictor, memory):
    # 1. Phase 3-A forward pass
    raw_pred = predictor.predict_sentence(sentence)
    pooled_irab = predictor.last_pooled_features  # cached for span encoding

    # 2. Detect constructions in this sentence
    constructions = detect_constructions(sentence, raw_pred.items)

    # 3. For each construction span: retrieve + bias
    biased_pred = raw_pred.copy()
    for (start, end, family, particle) in constructions:
        query_signature = build_signature(sentence, start, end, family, particle,
                                           pooled_irab[start:end])
        analogues = memory.retrieve(query_signature, k=5)
        if not analogues:
            continue
        prior_case, prior_role, prior_marker = aggregate_prior(analogues, end - start)
        biased_pred = apply_logit_bias(biased_pred, start, end,
                                        prior_case, prior_role, prior_marker, λ=0.3)

    return biased_pred, RetrievalTrace(constructions, analogues_per_span)
```

The `RetrievalTrace` is the explanation hook: it carries the
top-k retrieved sentences + their similarity scores + which
construction span they were used for. This becomes the basis for
Phase 11 (explanation engine).

## 10. Integration with Phase 3-A baseline

Phase R is **strictly additive** over Phase 3-A:

- `λ = 0` → Phase 3-A predictions byte-exact (no retrieval applied)
- `λ > 0` → soft retrieval bias on logits

No retraining. Phase 3-A checkpoint stays at `runs/phase3a_491240/final/`.
Phase R is implemented as a new predictor wrapper class
`RetrievalAugmentedPredictor(StructuredPredictor)` that takes a
retrieval-memory directory + `λ` and overrides `predict_sentence`.

## 11. Ablation plan

| Cell | Description | Phase 3-A baseline | What it tests |
|---|---|:---:|---|
| R-0 | λ=0 (Phase 3-A baseline) | reference | sanity check (must reproduce baseline) |
| R-A | symbolic filter only, no vector ranking | | does the symbolic family/particle filter alone help? |
| R-B | vector only, no symbolic filter | | does pure vector similarity help without symbolic constraint? |
| R-C | symbolic + vector, λ=0.3 (default) | | full system |
| R-D | symbolic + vector, λ=0.1 | | minimal retrieval bias |
| R-E | symbolic + vector, λ=1.0 | | aggressive retrieval bias |
| R-F | per-construction λ-tuning | | does the right λ vary by construction family? |

Run order: R-0 (sanity) → R-C (default full system) → R-A, R-B
(decompositions) → R-D, R-E (λ sweep) → R-F (per-construction λ if R-C ships).

## 12. Decision gate

Phase R-C ships as the new production INFERENCE pipeline (Phase 3-A
checkpoint unchanged) if:

1. **Per-construction kāna_sisters fully ≥ +5 pp** on Gazelle vs
   Phase 3-A baseline (currently 0%).
2. **Per-construction istithnāʾ fully ≥ +5 pp** on Gazelle vs
   baseline (currently 0%).
3. **Overall Gazelle case + role + fully within ±1 pp** of Phase 3-A
   baseline (no regression on majority constructions).
4. **MASAQ retention preserved** (within ±1 pp).
5. **Retrieval coverage**: ≥80% of test sentences containing target
   constructions had ≥1 valid retrieval.

If gate (1)+(2)+(3) pass: Phase R-C ships as default inference
pipeline. Phase 3-A checkpoint stays the underlying model.

If only (1) or (2) passes: Phase R ships per-construction (apply
retrieval only on the construction families that benefited).

## 13. Evaluation protocol

For each ablation cell:
1. Run `eval_per_construction.py` on Gazelle + MASAQ
2. Capture per-construction case/role/marker/fully + calibration gap
3. Diff vs Phase 3-A baseline (per construction)
4. Run `eval_phase4a.py` for the 4-stream + stress table (overall metrics)
5. **New: retrieval-hit analysis** — what fraction of test sentences
   had retrievals? What was the avg cosine to the top retrieval?

New analysis file: `runs/phaseR_<cell>/retrieval_hit_summary.json`
with fields:
- per-construction hit rate (% of test sentences with ≥1 retrieval)
- avg top-1 cosine
- distribution of retrievals-per-test-sentence
- per-construction confidence: avg `score(top_1, q)` across hits

## 14. Compute estimate

| Step | Time |
|---|---:|
| Build retrieval memory (one-time) | ~10 min HPC |
| Per ablation cell eval | ~5 min HPC |
| 7 ablation cells (R-0 through R-F) | ~35 min HPC |
| Per-construction analysis + writeup | ~30 min local |
| Paper integration (REPORT.md, REPORT.tex, PDF rebuild) | ~30 min |
| **Total: ~2 h** | |

Per the data-centric phase, this is small compute. Most time is in
designing the symbolic/vector layers and validating the retrieval
output quality.

## 15. Failure modes

1. **Retrieval pool too small for rare constructions.** Phase 3-A
   training corpus has only 4,750 sentences. After symbolic filter to
   kana_sisters, the pool may be ~100-200 instances total. Ablation
   plan: count per-family pool size in step (8); if any family has
   <50 instances, supplement from existing UD-PADT (UD has dep + morph
   but no iʿrāb — would need synthetic iʿrāb labels for retrieval-only
   instances; defer to Phase R+).

2. **Stanza dep noise propagates into retrieval signatures.** The 30%
   Stanza alignment failure documented in Phase 3 means many
   distill_v2 sentences have noisy dep info. Mitigation: store
   `confidence` per instance based on `has_dep` flag; weight retrieval
   scores by source confidence.

3. **Retrieval bias can hurt majority constructions.** If `λ > 0` for
   a sentence with no retrievals (i.e. a non-target construction), we
   already gate on that — the `apply_logit_bias` only fires when
   retrievals exist. But edge cases like wrong construction detection
   (false-positive: sentence flagged as kana but wasn't) could
   pollute predictions. Mitigation: detection precision audit on
   training set; per-construction false-positive rate must be <5%.

4. **Out-of-distribution test inputs.** Quranic test sentences may
   match no kana training instances (registers differ). This is fine
   — `λ = 0` for empty-retrieval spans means Phase 3-A passes through.

5. **λ overfitting on val set.** Val set is small (250 distill_v2 +
   small synthetic). Cross-construction λ-tuning risks overfit. We
   report results on a held-out 20% of val explicitly.

6. **Inference latency.** Per-sentence: 1 encoder forward + N FAISS
   queries (N = number of constructions). Should be <100ms. Acceptable
   for offline eval; for real-time deployment we'd cache embeddings.

## 16. Expected gains

Conservative target (per the decision gate):
- kana_sisters Gazelle fully: 0% → 5-15% (matches §5.4 0/7 gap)
- istithnāʾ Gazelle fully: 0% → 5-15%
- overall Gazelle fully: ≈ 25.2% → 26-28% (small overall gain from rare-class lift)
- MASAQ: similar small gains, no regression

Optimistic (if retrieval pool is rich enough):
- kana_sisters Gazelle fully: 0% → 25-40%
- istithnāʾ: 0% → 25-40%
- overall fully: 25.2% → 28-32%

Pessimistic (retrieval pool too sparse):
- kana_sisters: 0% → 0-5%
- ship per-construction; Phase 3-A stays as default for non-targeted
  constructions

## 17. Implementation order

1. **Construction signature module** (`src/irab_tashkeel/grammar_memory/signature.py`):
   `ConstructionInstance` dataclass + `build_signature(sentence, span, …)` +
   span-level encoder embedding helper. ~1h.

2. **Retrieval memory module** (`grammar_memory/memory.py`):
   FAISS-based per-family index, `add(instance)` / `retrieve(query, k)`,
   serialisation to disk. ~1h.

3. **Memory build pipeline** (`scripts/grammar_memory/build_memory.py`):
   Iterate distill_v2 train, detect constructions, encode spans, write
   to per-family JSONL + FAISS. ~30 min.

4. **HPC build sbatch** (`scripts/slurm/83_build_grammar_memory.sbatch`):
   ~10 min HPC to populate the memory.

5. **RetrievalAugmentedPredictor** (`grammar_memory/retrieval_predictor.py`):
   Wraps `StructuredPredictor`; computes signature on-the-fly during
   `predict_sentence`; queries memory; applies logit bias. ~1h.

6. **Ablation eval driver** (`scripts/grammar_memory/eval_phaseR.py`):
   Calls the existing `eval_per_construction.py` + new
   `retrieval_hit_summary` for each λ + each retrieval mode. ~30 min.

7. **R-0 through R-C runs on HPC** (~25 min sequential).

8. **Per-construction comparison + writeup** (~30 min local).

9. **Paper integration** (~30 min).

**Total wallclock**: ~6 h focused work (~5 h human / ~1 h HPC).

## 18. Out of scope for Phase R first pass

- **Training-time retrieval injection** (a future Phase R+: feed
  retrieved instances as additional input features during Phase 3-A
  retraining). Defer until R-C ships and we know retrieval helps.
- **UD-PADT synthetic iʿrāb labelling** for retrieval-pool augmentation.
- **Cross-corpus retrieval** (e.g., retrieve from MASAQ at test time
  for Gazelle inputs).
- **Hard symbolic constraints derived from retrieved evidence**
  (would re-introduce Phase 2-style joint-dynamics issues).
- **Retrieval-aware encoder pre-training**.
- **Approximate nearest-neighbour** (IVF/PQ) — not needed at 14K pool size.

## 19. Phase 11 explanation engine — built on top of Phase R

Once Phase R ships, the explanation engine becomes the natural next
phase:
- The `RetrievalTrace` already carries top-k retrieved examples per
  span
- Add a small templated rationale generator that says "this token is
  predicted as `khabar_kana` because the retrieved analogue
  `أصبح المعلمُ مجتهداً` shows `مجتهداً` taking the same role with
  cosine 0.87"
- Combines with morph + dep evidence (already available) to produce
  a multi-source rationale per prediction

This is the canonical "neuro-symbolic" output the project framing
calls for.

## 20. Ship sequencing

Phase R is the next major experiment. After it lands:
1. **R-C ships** (or per-construction subset) → REPORT.md + REPORT.tex update
2. **Phase 11 explanation engine** layered on Phase R's RetrievalTrace
3. **Aug v2 (Priority 2)** can run in parallel with Phase 11 — different
   surface (data side vs inference side); if both ship, they compose
4. **Dependency quality (Priority 4)** — affects retrieval signature
   quality, deferred behind R-C

Production lineage updates to: rev 2 → Phase 1 → Phase 3-A
(architecture-frozen) + Phase R retrieval (inference wrapper).
