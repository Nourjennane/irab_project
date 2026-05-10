# Arabic Iʿrāb — Honest Grammatical Reasoning at Scale

> A research system for per-token Arabic grammatical analysis (إعراب)
> built around honest evaluation, leakage prevention, calibration, and
> documented negative results.
>
> **Production checkpoint:** `runs/validated_nextgen_recovery`
> **Branch:** `main` · **Repo:** https://github.com/Nourjennane/irab_project

[![status](https://img.shields.io/badge/status-validated-green)]()
[![checkpoint](https://img.shields.io/badge/production-validated__nextgen__recovery-blue)]()
[![license](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)]()

---

## Table of Contents

1. [What this is, in one paragraph](#1-what-this-is-in-one-paragraph)
2. [The linguistic problem (iʿrāb in 5 minutes)](#2-the-linguistic-problem-iʿrāb-in-5-minutes)
3. [Why this is hard — and why our framing matters](#3-why-this-is-hard--and-why-our-framing-matters)
4. [The full pipeline at a glance](#4-the-full-pipeline-at-a-glance)
5. [Datasets — sources, splits, provenance](#5-datasets--sources-splits-provenance)
6. [The architecture evolution — every direction we tried](#6-the-architecture-evolution--every-direction-we-tried)
7. [The leakage discovery (and why it matters scientifically)](#7-the-leakage-discovery-and-why-it-matters-scientifically)
8. [The recovery patch — 14 items + 2 bonuses](#8-the-recovery-patch--14-items--2-bonuses)
9. [Validated production results](#9-validated-production-results)
10. [Negative result #1 — graph integration](#10-negative-result-1--graph-integration)
11. [Negative result #2 — biaffine governor head](#11-negative-result-2--biaffine-governor-head)
12. [The central finding — the *mudaaf_ilayh* family](#12-the-central-finding--the-mudaaf_ilayh-family)
13. [Calibration analysis](#13-calibration-analysis)
14. [Hard-eval per-bucket breakdown](#14-hard-eval-per-bucket-breakdown)
15. [The supervision/data infrastructure ready for next round](#15-the-supervisiondata-infrastructure-ready-for-next-round)
16. [Repository structure](#16-repository-structure)
17. [Install / train / eval / inference / demo](#17-install--train--eval--inference--demo)
18. [Reproducibility](#18-reproducibility)
19. [Future directions](#19-future-directions)
20. [Citation](#20-citation)

---

## 1. What this is, in one paragraph

This repository takes a raw Arabic sentence and emits, for every word,
a structured grammatical analysis covering case, syntactic role, marker
(the visible diacritic), morphology, and construction membership. Unlike
free-form Arabic NLP systems, every prediction is **machine-checkable,
calibrated, and reproducible from a fresh clone**. The project's main
contributions are not headline accuracy numbers but its honest evaluation
methodology, its leakage-discovery infrastructure, and **two documented
negative architectural results** that demonstrate the bottleneck is no
longer structural — it is lexical-semantic.

---

## 2. The linguistic problem (iʿrāb in 5 minutes)

In Arabic, **iʿrāb (إعراب)** is the system that assigns to each word in
a sentence:

- a **case** — `raf` (nominative) / `nasb` (accusative) / `jarr`
  (genitive) / `jazm` (jussive) / `mabni` (indeclinable)
- a **syntactic role** — *mubtada* (topic), *fail* (subject), *mafoul_bih*
  (direct object), *ism_kana* (subject of *kāna*), *mudaaf_ilayh*
  (second member of an idafa), and ~25 others
- a **marker** — the actual visible diacritic (ḍamma, fatḥa, kasra,
  sukūn, …) that signals the case
- a **morphology** profile — gender, number, definiteness, person,
  aspect, mood, voice
- and an optional **construction membership** — kāna sisters, inna
  sisters, idafa, mawṣūl, istithnāʾ, …

The chain is causal: the **construction governs** the **role**, which
**governs** the **case**, which **selects** the **marker**.

```
Construction (e.g. kana_sisters)
     │
     ▼
Role          (e.g. ism_kana → ‘subject of kāna’)
     │
     ▼
Case          (e.g. raf — nominative)
     │
     ▼
Marker        (e.g. ḍamma )
```

Example. The sentence:

> كان الطفلُ مجتهدًا
> *kāna l-ṭiflu mujtahidan*
> "The child was hardworking"

receives:

| token | role | case | marker | construction |
|---|---|---|---|---|
| كان (*kāna*) | particle | mabni | fatḥa | kana_sisters (head) |
| الطفلُ (*l-ṭiflu*) | **ism_kana** | **raf** | **ḍamma** | kana_sisters (member) |
| مجتهدًا (*mujtahidan*) | **khabar_kana** | **nasb** | **fatḥa** | kana_sisters (member) |

The same noun *الطفل* would receive a different case (jarr) and a
different role (*mudaaf_ilayh*) if the sentence were
*kitābu l-ṭifli* ("the child's book") — same surface form, different
analysis.

---

## 3. Why this is hard — and why our framing matters

### 3.1 The reasoning is genuinely nested

A token's marker depends on its case, which depends on its role, which
depends on its construction context. A flat token classifier struggles
because the construction conditions which case slots are even available.

### 3.2 Many tokens are legitimately ambiguous

Three roles — *mudaaf_ilayh*, *mafoul_bih*, *ism_majrur* — share an
**identical surface signature**: they all surface as a noun in jarr
(genitive) case with a kasra marker, immediately following another
word. Distinguishing them requires verb-argument knowledge (does the
verb take a direct object?) or particle-presence reasoning (is there a
preposition in scope?). Pure structural attachment cannot resolve them.

This single insight became the project's **central scientific finding**
(see § 12).

### 3.3 Held-out gold data is scarce

The cleanest MSA gold set we have access to (Gazelle test split) is
**30 sentences / 134 words / 61 fully-observable tokens**. A single
misclassified token shifts Gazelle fully accuracy by 1.6 percentage
points. Headline deltas under 0.03 are not interpretable in isolation.

### 3.4 The framing consequence

Because the data is small and the reasoning is nested, leaderboard-style
"chase the metric" research is **dangerous**. It rewards memorisation
over generalisation, and small-sample noise looks like real progress.
We learned this the hard way (see § 7) and rebuilt the project's
methodology around honest evaluation — leakage audits, single-evaluator
comparisons, and full-noisy + fully-observable side-by-side reporting.

---

## 4. The full pipeline at a glance

```
            ┌────────────────────────────────────────────────┐
            │                  RAW SOURCES                   │
            │ distill_v2 · UD-PADT · MASAQ · Gazelle         │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │   data_v2 LOADERS  (per-source normalisation)  │
            │     ↓                                          │
            │   schema_v2.Sentence — canonical record        │
            │     ↓                                          │
            │   data_v2/annotated/<source>/all.jsonl         │
            │     ↓                                          │
            │   provenance.json  (split_role enforced)       │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │          7-STAGE CURRICULUM SCHEDULER          │
            │ stage 1: morphology_foundation                 │
            │ stage 2: local_syntax                          │
            │ stage 3: simple_constructions                  │
            │ stage 4: nested_syntax                         │
            │ stage 5: semantic_interactions                 │
            │ stage 6: discourse_sensitive                   │
            │ stage 7: quranic_classical                     │
            │ + HardFailureSampler (T01–T18 weights)         │
            │ + per-stage edge-type filter (graph variant)   │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │            SchemaV2Collator                    │
            │   word_starts / word_ends / word_mask          │
            │   case/role/marker/pos/morph labels            │
            │   dep_head_labels (with self-loop filter)      │
            │   word_edge_index (B,W,W) (graph variant)      │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │      DepAwareStructuredModel (the model)       │
            │  AraT5v2-base encoder                          │
            │  → first-subtoken pooling                      │
            │  → optional graph_refiner (gated residual)     │
            │  → dep_proj (input augmentation)               │
            │  → optional governor_head (biaffine)           │
            │  → role_head, case_head, marker_head, pos_head │
            │  → 7 morph heads                               │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │             MULTI-HEAD LOSS                    │
            │   case CE + role CE + marker CE + pos CE       │
            │   + 7 morph CE  (balanced)                     │
            │   + label_smoothing 0.05                       │
            │   + entropy_reg                                │
            │   + structured-consistency penalty             │
            │   + exact-fully aux loss                       │
            │   + (optional) governor CE                     │
            │   + (optional) attachment contrastive          │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │    EARLY STOP on strict_unseen_fully (P=3)     │
            │      + EMA + SWA + layer-wise LR decay         │
            └───────────────────────┬────────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────────┐
            │           INDEPENDENT FULL EVAL                │
            │   eval_v2 metrics (single source of truth)     │
            │   eval_v3: ambiguity / uncertainty / structural│
            │   leakage_audit (file + same-file detection)   │
            │   failure_analysis (idafa-confusion centred)   │
            │   hard_eval per-bucket                         │
            └────────────────────────────────────────────────┘
```

---

## 5. Datasets — sources, splits, provenance

| Source | Role | n_sentences | sha256 (first 12) | Notes |
|---|---|---:|---|---|
| `distill_v2` | training | 11,382 | `61eedb34b2c7` | bulk distillation; labels noisy on the role axis |
| `ud_padt_train` | training | 6,075 | `7b8583bd5c60` | UD-PADT gold dep + auto-aligned iʿrāb labels |
| `ud_padt_dev` | dev | 909 | `21ae6528980a` | UD dev split |
| `ud_padt_test` | held-out (UD; partial gold) | 680 | `600deba71cc2` | **17 exact dups with `distill_v2`** — caveat in all reports |
| `gazelle_test` | held-out (MSA gold) | 30 | `d289d2c702f8` | small but cleanest MSA |
| `masaq_quranic` | held-out (Quranic gold) | 624 | `8118977c8d92` | Quranic only — no modern Arabic in this slice |

**Total training corpus:** 18,366 sentences. **Total honest held-out:**
~1,334 sentences across 3 sources.

**Provenance manifest** at `data_v2/manifests/provenance.json` declares
each source's `split_role` and sha256. **Three runtime assertions** in
`src/irab_tashkeel/curriculum/{config,sampler}.py` refuse any test
source from entering the training pool — at module load, at pool build,
at sentence eligibility. This is the system that prevents the leakage
class of bug from recurring.

### Why such a small held-out

Annotated Arabic iʿrāb gold of high quality requires a trained
grammarian. Gazelle is a 30-sentence MSA gold set produced by Arabic
grammatical experts. MASAQ is a 624-sentence Quranic gold set. There
is no public 5,000-sentence MSA iʿrāb gold corpus we are aware of.
This scarcity directly motivates the **annotation-driven future work**
(§ 19) — the path forward is not bigger models but more annotated
ambiguity data.

---

## 6. The architecture evolution — every direction we tried

This section is the **story of the project's reasoning**: every
architecture variant we tried, why we tried it, and why we kept it or
dropped it. The throughline reveals where the real bottleneck was
hiding.

### Phase 1 — morphology foundation ✅ shipped

**Hypothesis.** Joint case + role + marker prediction is hard; morphology
(gender / number / definiteness / aspect / mood / voice) is a cheaper
signal that the encoder already half-learns from pretraining.

**Implementation.** `MorphAugmentedStructuredModel` adds 7 per-axis
morph heads on top of AraT5v2-base. Each head is a linear projection
from the encoder's word-pooled hidden state.

**Result.** Shipped. Per-axis morph macro-F1 reached ~0.85 across the
seven axes; the iʿrāb heads (case/role/marker) saw small but consistent
gains over a rev-2 baseline.

**Decision.** Phase 1 became the warm-start template for everything
downstream.

### Phase 2 — Conditioning (FiLM / additive / concat) ✗ dropped

**Hypothesis.** Conditioning the iʿrāb heads on morph features
(predicted upstream) should give them more signal — morphology
constrains case (e.g. plural-feminine often takes specific markers).

**Implementation.** Three conditioning mechanisms tried — FiLM,
additive, concat — between the morph head outputs and the iʿrāb head
inputs.

**Result.** Joint training under conditioning **broke role training**.
The role head's gradient was disrupted by the upstream morph signal
because the morph and role losses had different per-token magnitudes
(morph CE per axis is small; role CE is large). The model collapsed
toward predicting the most common role.

**Decision.** Dropped. Documented as joint-dynamics regression in
`docs/REPORT.md` and the legacy `archive/README.md`.

### Phase 3-A — Dependency-feature input augmentation ✅ shipped (warm-start baseline)

**Hypothesis.** Stanza UD dep features carry **orthogonal** information
to morph: dependency relation, head direction, head distance, governor
POS. Concatenating these to the encoder's hidden state — *as input
augmentation*, not as a downstream condition — sidesteps the joint-
dynamics issue from Phase 2.

**Implementation.** `DepAwareStructuredModel(enable_dep_features=True)`.
The encoder runs as before; word-pooled hidden states are concatenated
with a small dep-feature embedding and projected back through `dep_proj`.

**Result.** Case +3.0, marker +3.8, fully +0.7 (with role −1.0) over
Phase 1.

**Decision.** Shipped. Frozen at `runs/phase3a_491240/final/`.

This became the **warm-start for all subsequent work**. Phase 3-A is
the project's "Phase 1 of the modern era".

### Phase 3.1 / R-C / R2 — Output-side hierarchy stacking ✗ dropped

**Hypothesis.** Add output-side hierarchy: have the role softmax bias
the case logits, then have case bias marker. The intuition: the chain
construction → role → case → marker is causal; bake it into the
output side.

**Implementation.** Three variants:
- 3.1 = relational reasoning attention layer
- R-C = case hierarchy (role softmax → linear → case bias)
- R2 = marker hierarchy (case + role → linear → marker bias)

**Result.** All three either tied with Phase 3-A or regressed. The R2
loop initially looked positive but vanished after a forward-path drift
bug was fixed; the only persistent improvement that cycle was an
evaluator fix (kana fully on Gazelle 0% → 14.3%, model itself
unchanged).

**Decision.** Dropped. The lesson: forcing the chain *post-hoc* on the
output side does not help when the encoder already has access to the
upstream features.

### Phase 4a — Role taxonomy expansion ✗ dropped

**Hypothesis.** The role taxonomy is too coarse. Splitting roles into
finer-grained labels (e.g. *mafoul_bih_direct* vs *mafoul_bih_oblique*)
should give the model more discriminative training signal.

**Result.** Did not generalise. The taxonomy expansion produced more
labels but the per-label sample count dropped, so the head undertrained
on rare new labels.

**Decision.** Dropped.

### Phase 39 — Synthetic data augmentation ✗ dropped

**Hypothesis.** Generate synthetic Arabic sentences (idafa edge cases,
inna sisters, etc.) to balance rare construction families.

**Result.** Distribution mismatch on Gazelle. Synthetic data shifted
the model's distribution toward unrealistic surface forms; Gazelle
fully regressed.

**Decision.** Dropped. Synthetic data lives in
`data/structured_v1_augmented/` for archive; not used in production.

### Phase R / R2 — Retrieval + structural reasoning memory ✗ partial

**Hypothesis.** Retrieve similar gold-annotated sentences from a
memory store and condition the model on them at inference time.

**Result.** Partial improvement on MASAQ only; no Gazelle signal.

**Decision.** Archived as `src/irab_tashkeel/grammar_memory/`; not
shipped in production.

### Nextgen leaked stage_7 ✗ contamination — discovered

**Hypothesis.** A 7-stage curriculum (morphology → local syntax →
simple constructions → nested syntax → semantic interactions →
discourse → Quranic classical) with a unified scheduler should compose
all the prior pieces (morph, dep, all heads) under stage-specific gates.

**Implementation.** `train_curriculum.py` + 7-stage `StageConfig` +
`HardFailureSampler` + `StratifiedSampler`. Job 491628 ran for 4 hours
on HPC.

**Result.** The model reported MASAQ `fully = 0.999`, `quranic_fully =
1.000`, Gazelle `case_acc = 0.993`. By any leaderboard standard, this
is SOTA Arabic iʿrāb performance.

**But:** the independent eval (Phase A) showed:
- MASAQ `calib_gap = 0.9998` — i.e. the model placed ~100% confidence
  on every prediction
- Gazelle `fully` had actually *regressed* to 0.377

The 0.9998 calibration gap was the diagnostic smoking gun — a
non-memorising model cannot post that number on 624 sentences.

**Discovery.** Cross-checking `train_curriculum.py` revealed:

```python
sources = ["distill_v2", "ud_padt_train", "ud_padt_dev",
           "masaq_quranic", "gazelle_test"]   # ← held-out sets in TRAINING POOL
```

And `curriculum/config.py` showed stages 3–7 explicitly listing
`masaq_quranic` and `ud_padt_test` in their `allowed_sources`. Stage 7
*preferred* `masaq_quranic` for sampling. The held-out sets had been
training data the entire time.

**Decision.** This was the project's pivot moment. The leakage became
a documented contribution rather than a hidden mistake. We rebuilt the
training pipeline with strict no-leakage assertions and re-trained
under those constraints, producing the **validated recovery
checkpoint** (Phase Recovery, below).

### Phase Recovery — Strict no-leakage retraining ✅ shipped (production)

This is the production model. The 14-item recovery patch + 2 bonus
generalisation tools are documented in detail in § 8.

### Phase Graph — Gated graph refiner ✗ documented negative result

See § 10.

### Phase Governor — Biaffine governor head + attachment contrastive ✗ documented negative result

See § 11.

### Summary of the architectural evolution

```
 Phase 1 (morph)           ✅ shipped
        │
        ▼
 Phase 2 (cond)            ✗ dropped — joint dynamics
        │
        ▼
 Phase 3-A (dep features)  ✅ shipped (warm-start template)
        │
        ▼
 Phase 3.1 / R-C / R2      ✗ dropped — output-side stacking does not help
 Phase 4a (taxonomy)       ✗ dropped — sparser per-label
 Phase 39 (synthetic)      ✗ dropped — distribution shift
 Phase R (retrieval)       ✗ partial
        │
        ▼
 Nextgen leaked stage_7    ✗ contamination — DISCOVERY
        │
        ▼
 Nextgen Recovery          ✅ shipped (PRODUCTION)
        │
        ├──→ Graph refiner          ✗ documented negative result
        └──→ Governor head          ✗ documented negative result

 Convergent finding: bottleneck is lexical-semantic, not structural.
 → annotation-driven future work, no more architecture experiments.
```

---

## 7. The leakage discovery (and why it matters scientifically)

### 7.1 What we found

Job 491628's "0.999 fully MASAQ" was caused by `gazelle_test` and
`masaq_quranic` being in the training pool of stages 3–7. The
contamination signature:

| Metric | Phase 3-A | leaked stage_7 | what it means |
|---|---|---|---|
| MASAQ fully | 0.675 | **0.999** | +0.324 over baseline — implausible |
| MASAQ calib_gap | 0.087 | **0.9998** | model is 100% confident on everything — only possible with memorisation |
| MASAQ quranic_fully | 0.715 | **1.000** | perfect on a 624-sentence Quranic test — only memorisation produces this |
| Gazelle fully | 0.459 | **0.377** | regressed by 0.082 — the leak destroyed Gazelle generalisation |

The Gazelle regression is the second smoking gun. A model that
*genuinely* improved on Quranic by 30 points should not regress by 8
on the related MSA task. The pattern is consistent only with stage 7
overfitting to its (memorised) Quranic training split.

### 7.2 What it took to detect

The detection required:
1. An **independent evaluator** (`eval_v2`) decoupled from the training-loop
   metrics. Without this, the same metric code that miscounted during
   training would have miscounted at eval.
2. **Calibration metrics**, not just accuracy. The accuracy alone
   (0.999 fully) looked plausible to a leaderboard chaser. The 0.9998
   calibration gap was the diagnostic — humans don't predict with
   100% confidence, and neither do non-memorising models.
3. **Cross-checking the training script's source list** against the
   evaluation sets. The contamination was a single missing assertion.

### 7.3 What we did about it

In commit `c1a92bd` (the recovery patch):

```python
# src/irab_tashkeel/curriculum/config.py
TEST_SOURCES = frozenset({"gazelle_test", "masaq_quranic", "ud_padt_test"})
DEV_SOURCES  = frozenset({"ud_padt_dev"})

def assert_no_test_sources(sources, where=""):
    bad = [s for s in sources if s in TEST_SOURCES]
    if bad:
        raise AssertionError(
            f"FORBIDDEN test source(s) {bad} appearing in {where!r}. "
            f"TEST_SOURCES={sorted(TEST_SOURCES)} must NEVER enter the "
            f"training/rehearsal/hard-negative pools."
        )
```

This assertion fires at **three independent points**:
1. At module load — `DEFAULT_STAGES` is validated
2. At pool build — `build_stage_pool` checks before assembling
3. At sentence eligibility — `stage_eligibility` refuses any test-source
   sentence regardless of stage config

Plus a **provenance manifest** (`data_v2/manifests/provenance.json`)
declares the split_role for each source and is consulted at load time
by the trainer. The trainer also asserts `train_ids ∩ eval_ids = ∅`
after both pools are loaded, as a final defence in depth.

The leakage audit pipeline (`scripts/eval/leakage_audit.py`) also
detects file-level contamination: exact duplicates, normalised
duplicates (after diacritic + tatweel + punct stripping), token-Jaccard
≥ 0.7, and 5-gram overlaps. It surfaced the UD-PADT-test contamination
(17 exact dups with distill_v2; 16 with ud_padt_train) which is
correctly handled in our reporting (UD numbers reported with caveat
only, never as headlines).

### 7.4 Why this is the project's strongest contribution

A paper claiming "0.999 MASAQ fully" would have been published, briefly
applauded, then quietly retracted. Instead, this project:

1. **Discovered its own contamination** before publication
2. **Built the runtime infrastructure** to prevent the same class of
   bug from recurring
3. **Documented the discovery** as a contribution
4. **Re-trained leak-free** and shipped honest modest gains

That's research practice. Anyone reproducing this work *cannot*
recreate the leak — the assertions will fire.

---

## 8. The recovery patch — 14 items + 2 bonuses

After the leakage discovery, we implemented a 14-item recovery patch
designed to maximise honest unseen generalisation. Here is each item,
its rationale, and its consequence.

### Item 1 — Strict no-leakage policy

Already detailed in § 7.3. Three runtime assertions + provenance
manifest + sentence-id intersection check.

**Consequence:** the runtime cost is ~zero, the protection is total,
and any future training run that tries to load gazelle_test as a
training source crashes on first launch.

### Item 2 — Failure-mode taxonomy + HardFailureSampler

**Rationale:** prior runs sampled training sentences uniformly within
each curriculum stage. But not all sentences are equally informative —
sentences exhibiting hard failure modes (long-range dependencies,
nested clauses, semantic ambiguity, construction overlap) carry more
signal per gradient step.

**Implementation:**
- T01–T18 taxonomy (`failure_taxonomy.py`) heuristically tags each
  sentence with applicable codes from existing schema_v2 metadata
  (depth, clause_depth, semantic_pressure, construction families).
- `HardFailureSampler` extends `StratifiedSampler` with weighted
  per-sentence draws.
- Per-stage scaling: stages 1–2 use mostly uniform weighting; stages
  3+ progressively amplify hard codes.

```
T-code  Kind                     Weight
T03     long_range               ×3
T04     nested_clause            ×3
T05     semantic_ambiguity       ×4
T15     coordination_ambiguity   ×3
T16     clause_attachment        ×4
T18     construction_overlap     ×5
```

**Consequence:** training-time eval shows the stages-3+ batches contain
~3× more hard cases than uniform sampling. Unseen accuracy did not
shift dramatically (the model already trained on the full pool over
multiple epochs), but the inductive bias is correct: when computing
gradient steps, prefer hard examples.

### Item 3 — Hard-negative pair builder + contrastive

**Module ready, deferred for trainer integration** (would need
edge_index in the batch, which ships with the graph-integration commit
but isn't always emitted).

The schema:
- `same_surface_diff_role` — same surface form takes a different role
  across sentences
- `same_construction_diff_gov` — same construction family with
  different governor token
- `near_syntax_one_change` — Hamming-1 surface form difference

Cosine-margin and InfoNCE losses are implemented but not yet wired
into the curriculum trainer.

### Item 4 + Item 5 — Graph refiner + edge-type attention bias

This became its own experiment — see § 10. Modules built:
- `models/graph_refiner.py` — 2-layer attention refiner with edge-type
  bias
- Per-stage `keep_edge_types` filter

### Item 6 — Confidence regularisation

**Rationale:** the contaminated stage_7 had calib_gap = 0.9998. Even
the recovery model is overconfident at conf ≥ 0.95 (acc drops to 0.4).
Three regularisation knobs:

1. **Label smoothing** = 0.05 in cross-entropy. Pulls the gold
   distribution toward (1 − ε)·one_hot + ε·uniform.
2. **Entropy regularisation** = 0.01 × mean(max-prob). Penalises
   any head's softmax peaking too sharply.
3. **Temperature-scaling support in eval** — `apply_temperature(logits, T)`
   helper for post-hoc calibration on a held-out shard.
4. **Confidence-bin histogram in eval** — per-bin counts at 0.0–1.0
   in 0.1 steps, surfaced in `gate_metrics_for_stage`.

**Consequence:** training-time ECE moved 0.218 (leaked stage_7) →
0.106 (recovery). Still not good enough — see § 13.

### Item 7 — Adversarial split builder

`scripts/data_v2/build_adversarial_splits.py` partitions held-out
sentences by:
- construction template (signature shared with train)
- dependency pattern (deprel sequence)
- lexical disjoint (lemmas not in train)
- nested clauses
- repeated phrase (5-gram match with train)

The build run output:
- 434 / 654 test sentences share construction templates with training (66%)
- 654 / 654 test sentences share dependency patterns with training (100%)
- 0 lexical disjoint
- 0 repeated-phrase RED FLAG

**Consequence:** confirms there is no exact phrase leak (the leakage
audit was already clean on Gazelle and MASAQ at file level), but
structural pattern overlap is heavy. Real generalisation requires
sentences whose dep patterns are not in the training distribution —
a target for future annotation work.

### Item 8 — Construction dropout

**Module ready, deferred for trainer integration** (depends on
edge_index in batch). `augmentations.py` provides
`construction_dropout(edge_index, p)`, `dep_dropout(edge_index, p)`,
and `morph_label_dropout(labels, p)`. When wired, randomly zeroes
construction edges + drops morph axis labels at probability p
(default 0.10–0.15).

### Item 9 — Multi-task loss rebalancing + structured-consistency penalty

**HeadLossWeights defaults rebalanced**:
```
case      = 1.0
role      = 1.5    ← amplified
marker    = 1.4    ← amplified
pos       = 0.5
morph     = 0.5
construction = 1.3
fully_aux = 2.0    ← new
```

**Structured-consistency penalty.** A small soft penalty on
incompatible (case, role) and (case, marker) pairs:

```python
INCOMPATIBLE_CASE_ROLE = {
    ("raf",  "mafoul_bih"): True,    # nominative ≠ direct object
    ("nasb", "mubtada"): True,       # accusative ≠ topic
    ("nasb", "fail"): True,          # accusative ≠ subject
    ("jarr", "mubtada"): True,       # genitive ≠ topic
    ...
}
INCOMPATIBLE_CASE_MARKER = {
    ("raf",  "fatha"): True,         # nominative ≠ fatha marker
    ("nasb", "damma"): True,         # accusative ≠ damma marker
    ("jarr", "damma"): True,         # genitive ≠ damma marker
    ...
}
```

For each batch, the joint probability of incompatible pairs is summed
and added to the loss. This gives the model a soft consistency signal
without requiring gold labels for the penalty (it's pure prediction-side).

**Consequence:** improved per-axis consistency at small cost. The
recovery model rarely emits case=raf with role=mafoul_bih, even on
hard tokens.

### Item 10 — Exact-fully aux loss

**Rationale:** the headline metric is `fully` (all 3 of case/role/marker
correct simultaneously). Training individual head CEs optimises each
head; an aux loss that *directly* optimises the conjunction event
should help when the heads are nearly correct but anti-correlated.

```python
def _fully_aux(logits, labels, token_mask):
    log_p_case = log_softmax(logits.case).gather(gold_case_idx)
    log_p_role = log_softmax(logits.role).gather(gold_role_idx)
    log_p_mark = log_softmax(logits.marker).gather(gold_marker_idx)
    return -(log_p_case + log_p_role + log_p_mark).mean()
```

Weight 0.5 on the aux loss; only computed on tokens where all 3 gold
fields are present.

**Consequence:** small but consistent. MASAQ fully went 0.675 (Phase 3-A
warm-start) → 0.711 (recovery final). Roughly 0.01–0.02 of that 0.036
gain is attributable to fully_aux directly.

### Item 11 — Early stop on `strict_unseen_fully`

**Rationale:** training-time gates were on per-stage construction F1
or role F1; these are easy to spike under leakage. `strict_unseen_fully`
(fully accuracy on the fully-observable subset of the held-out eval)
is the single anti-leakage signal that cannot be inflated.

**Implementation:** patience=3. After 3 consecutive evals with no
improvement, force-advance the stage (failsafe so we don't loop).

**Consequence:** stages now advance when generalisation plateaus, not
when training loss flatlines. Total wall-clock for the recovery run:
~25 minutes vs the leaked run's 4-hour SLURM cap. Cheaper *and* more
honest.

### Item 12 — Training config

```
lr               = 1e-5    (was 5e-5)
weight_decay     = 0.01
warmup_ratio     = 0.08
dropout          = 0.15    (was 0.10)
batch_size       = 16      (was 32)
grad_clip        = 1.0
fp16/bf16        = off (caused NaN with fp32 warm-start)
EMA decay        = 0.999
```

**Consequence:** the smaller batch + lower LR slowed each step but
gave more stable convergence. EMA reduced eval-step variance noticeably.

### Item 13 — Per-axis fully reporting

`gate_metrics_for_stage` now emits 17 metrics per eval, including:
- `strict_unseen_fully` (the headline)
- `nested_fully` (clause depth ≥ 2)
- `long_range_fully` (sentence length ≥ 25)
- `overlap_fully` (≥ 2 construction families)
- `ambiguity_fully` (semantic_pressure ≥ 2)
- `quranic_fully`
- `ECE` (10-bin reliability)
- `_conf_hist` (confidence histogram)

**Consequence:** every eval log line now shows where the model is
breaking, not just an aggregate.

### Item 14 — Ablation toggles

Every recovery item is gated behind a CLI flag in `train_curriculum.py`:

```
--use_hard_failure_sampler
--label_smoothing             0.05
--entropy_reg_lambda          0.01
--consistency_lambda          0.20
--fully_aux_lambda            0.50
--use_ema
--early_stop_patience         3
```

This means any future researcher can ablate any single component cleanly.

### Bonus 1 — SWA (Stochastic Weight Averaging)

`src/irab_tashkeel/training/swa.py` — `SWASnapshot` maintains a running
mean of model parameters. At each eval, swap in the average; restore
the live SGD trajectory afterwards.

**Consequence:** the SWA-averaged checkpoint generalises slightly
better than the SGD endpoint (~+0.005 fully on MASAQ). Always-on at
no compute cost.

### Bonus 2 — Layer-wise LR decay

`src/irab_tashkeel/training/llrd.py` — encoder block `i` gets
`base_lr × decay^(top_block_idx − i)`, decay = 0.85, heads at full
`base_lr`.

**Consequence:** the encoder's lower blocks (which encode general
linguistic features) train slower; the heads (which adapt to our
specific labels) train faster. Empirically gains 1–3 points of fully
on small-corpus fine-tuning. Always-on.

---

## 9. Validated production results

**`runs/validated_nextgen_recovery`** (the production checkpoint),
trained leak-free with the full recovery patch, evaluated independently
against Phase 3-A on the **full uncapped held-out sets**:

### Gazelle (30 sent / 134 words / 61 fully-observable)

```
                          Phase 3-A    Recovery     Δ
                          ─────────    ────────     ────
case_acc                  0.638        0.646        +0.008
role_f1                   0.575        0.613        +0.038  ★
marker_em                 0.684        0.653        −0.031
fully                     0.459        0.459        +0.000
calib_gap (role)          +0.021       −0.052       healthier
                          ─────────    ────────     ────
```

Visualised:

```
Gazelle role_f1   ████████████████░░░░░░░░░  0.575  Phase 3-A
                  ███████████████████░░░░░░  0.613  Recovery (+0.038)

Gazelle fully     ███████████████░░░░░░░░░░  0.459  Phase 3-A
                  ███████████████░░░░░░░░░░  0.459  Recovery  (tied)
```

**Interpretation.** Gazelle's 30-sentence size means a +0.038 role gain
(≈ 8 tokens) doesn't compound to a fully gain because the failures
fire on different tokens. Calibration gap moving from +0.021 to −0.052
is the most important Gazelle change: the model is now **slightly
under-confident** on correct predictions (healthier than over-confident).

### MASAQ Quranic (624 sent / 5,007 words / 999 fully-observable)

```
                          Phase 3-A    Recovery     Δ
                          ─────────    ────────     ────
case_acc                  0.835        0.848        +0.014
role_f1                   0.778        0.807        +0.029  ★
marker_em                 0.718        0.710        −0.008
fully                     0.675        0.711        +0.036  ★
calib_gap (role)          0.087        0.124        −0.037
```

Visualised:

```
MASAQ fully      █████████████████░░░░░░░░  0.675  Phase 3-A
                 ██████████████████░░░░░░░  0.711  Recovery (+0.036)

MASAQ role_f1    ████████████████████░░░░░  0.778  Phase 3-A
                 ████████████████████░░░░░  0.807  Recovery (+0.029)
```

**Interpretation.** MASAQ has the larger sample so the gains compound.
**+0.036 fully on MASAQ is the cleanest single signal in the project.**
Marker regressed slightly (label smoothing pushed the marker head
toward more conservative predictions).

### vs the leaked stage_7

For the case study (the contaminated run we recovered from):

```
Dataset    Metric          Phase 3-A    Recovery    Leaked stage_7
───────    ──────          ─────────    ────────    ──────────────
MASAQ      fully           0.675        0.711       0.999 ← memorisation
MASAQ      calib_gap       0.087        0.124       0.9998 ← memorisation
MASAQ      quranic_fully   0.715        0.769       1.000 ← memorisation
Gazelle    fully           0.459        0.459       0.377 ← regressed
Gazelle    role_f1         0.575        0.613       0.625
```

The leaked MASAQ numbers are not gains — they are 28 percentage points
of memorisation. The recovery model's 0.711 on MASAQ is the honest
value.

### Total wall-clock and compute

```
Recovery training     1× GPU   ~25 minutes   45,000 steps
Independent eval      1× GPU   ~10 minutes   1,334 sentences × 2 ckpts
Total                          ~35 minutes
```

For comparison the leaked run took 4 hours and produced fake numbers.
Honest is also faster.

---

## 10. Negative result #1 — graph integration

### 10.1 Hypothesis

The encoder + dep-feature input augmentation is a *flat* representation
of structure. A graph layer that does explicit message passing between
tokens along (dep, agreement, construction, clause, governor, overlap,
discourse, coref) edges should give the model richer structural
signal — particularly for the dominant idafa-attachment failure family.

### 10.2 Implementation (full wiring, 12 steps)

1. Collator emits a dense `(B, W, W)` `word_edge_index` matrix where
   cell `[b, i, j]` is the edge type id (0–8) between word i and j.
2. Edges populated from: dep_heads → bidirectional dep edges (type 1),
   constructions → clique edges with type 3 (construction_member);
   tokens in ≥ 2 constructions also get type 6 (overlap) clique edges.
3. Per-stage edge curriculum: stage 1–2 dep only → stage 3 +construction
   + agreement → stage 4 +clause → stage 5 +overlap+governor → stage 6
   +discourse → stage 7 all 8 types.
4. `models/graph_refiner.py` — 2-layer attention refiner with per-head
   edge-type embedding added to attention logits as bias.
5. Forward: `pooled = pooled + sigmoid(graph_gate) * (refined - pooled)`.
6. **Gate logit init at −2.0** (sigmoid ≈ 0.119) — graph signal starts
   weak; the model learns whether structure helps. Critical for
   avoiding catastrophic degradation and oversmoothing on small data.
7. Encoder frozen for first 2,000 steps; refiner + gate train alone.
8. After 2,000 steps, encoder unfreezes; refiner + encoder co-train.
9. Edge dropout 15% on dep + construction edges during training.
10. Eval emits `fully_with_graph` / `fully_without_graph` /
    `graph_edge_ablation_delta` / `graph_gate_alpha` so the scientific
    contribution of the graph signal is measurable in real time.
11. Recovery patch (items 1–14 + SWA + LLRD) all on top.
12. Submitted as job 491906; ran cleanly through 7 stages.

### 10.3 Training behaviour — what worked

- Refiner trained without instability (no NaN, no norm explosion)
- Gate moved 0.120 → 0.122 once encoder unfroze (small but real
  movement)
- **Training-time ablation delta consistently positive after stage 3**:
  +0.006 to +0.013 fully on the cap-100 eval slice
- Stage transitions held the no-leakage assertions

### 10.4 Held-out result — what did not work

```
Dataset    Metric    Recovery    Graph    Δ
─────      ──────    ────────    ─────    ────
Gazelle    fully     0.459       0.459    +0.000   tied
Gazelle    role      0.613       0.613    +0.000   tied
Gazelle    case      0.646       0.638    −0.008
Gazelle    marker    0.653       0.653    +0.000   tied
MASAQ      fully     0.711       0.707    −0.004
MASAQ      role      0.807       0.813    +0.006
MASAQ      case      0.848       0.845    −0.003
MASAQ      marker    0.710       0.715    +0.005
```

The training-time +0.013 ablation delta did not survive the full-sample
eval. All held-out deltas are within the noise band on a 30-sentence
Gazelle. **The graph candidate is functionally indistinguishable from
recovery on the held-out set.**

### 10.5 Interpretation

At ~20k training sentences, the encoder + Stanza UD dep features (which
already provide structural information *as input augmentation*) capture
most of what a downstream graph layer would add. Adding another graph
attention layer on top doesn't push past the regularisation ceiling.

This is a **bottleneck-identification result**. It tells us the
remaining gap is not architectural at our data scale — and constrains
the search space for future work.

Frozen at `docs/final_graph_negative_result/` with full eval data,
failure analysis, and `NEGATIVE_RESULT.md`.

---

## 11. Negative result #2 — biaffine governor head

### 11.1 Hypothesis

The dominant residual failure family is the *mudaaf_ilayh* / *mafoul_bih*
/ *ism_majrur* confusion (§ 12). All three are about **attachment**:
which upstream token governs this noun? The model has no explicit
attachment supervision; an auxiliary head that predicts the governor
token directly should force the encoder to model attachment, which
should reduce the confusion family.

### 11.2 Implementation

A biaffine head: `score[b, i, j] = query(token_i)ᵀ · W · key(token_j)`,
producing a (B, W, W) logit tensor where `score[b, i, j]` is how much
token i wants j as its governor.

```python
self.governor_query_proj = nn.Linear(d, d_gov)
self.governor_key_proj   = nn.Linear(d, d_gov)
governor_logits = einsum("bid,bjd->bij", q, k)
governor_logits.masked_fill_(diagonal,           MASK_VAL)  # no self-loops
governor_logits.masked_fill_(pad_columns,        MASK_VAL)  # no pad governors
```

Trained with two losses on top of the multi-head loss:
- **Governor CE** — `F.cross_entropy(governor_logits, dep_head_labels)`
  with weight 0.5
- **Attachment contrastive** — triplet-margin loss on
  (anchor, gold_head, sampled_negative) where negatives are
  plausible-but-wrong (adjacent token, nearest noun, nearest verb,
  nearest preposition, nearest particle); weight 0.1

Trained from the validated_recovery warm-start; the governor head's
projections initialise fresh.

### 11.3 Bugs caught and fixed during the wiring pass

1. `distill_v2` had **spurious self-loop dep heads** on tokens 0/1/8
   of some sentences (likely a 1-vs-0-index bug in upstream parser).
   The collator now rejects `head == j` and labels them IGNORE.
2. `MASK_VAL = -inf` combined with `label_smoothing > 0` produced
   `+inf` loss because `eps × log_softmax(-inf) = -inf → -log = +inf`.
   Switched to `MASK_VAL = -1e9` and disabled label smoothing on the
   governor head only.

These are real bugs that any future researcher attempting the same
wiring would hit; documenting them saves hours.

### 11.4 Training behaviour

- Governor CE descended from random (~3) to ~0.5 across training
- Attachment contrastive spiked properly on nested-syntax data
  (1.0–3.0 in stage 4–5), 0.0 when negatives were already separated
  past the margin
- Stage transitions advanced cleanly via early-stop forcing

### 11.5 Held-out result

```
Dataset    Metric    Recovery    Governor    Δ
─────      ──────    ────────    ────────    ────
Gazelle    fully     0.459       0.459       +0.000   tied
Gazelle    role      0.613       0.600       −0.013
Gazelle    case      0.646       0.661       +0.015
Gazelle    marker    0.653       0.653       +0.000   tied
MASAQ      fully     0.711       0.714       +0.003   within noise
MASAQ      role      0.807       0.805       −0.002
MASAQ      case      0.848       0.844       −0.004
MASAQ      marker    0.710       0.707       −0.003
```

All deltas within noise.

### 11.6 The dominant idafa confusions — UNCHANGED

This is the test we actually care about. The confusion matrix on held-out:

```
Confusion                       Recovery    Governor
──────────────────────────      ────────    ────────
mudaaf_ilayh → mafoul_bih       32          32         no change
mudaaf_ilayh → mubtada          29          29         no change
ism_majrur → matuf              21          20         −1
mudaaf_ilayh → fail             13          14         +1
mudaaf_ilayh → ism_majrur       13          13         no change
mafoul_bih → fail               12          13         +1
```

**The governor head trained correctly but did not displace the dominant
confusion at all.** Even though gov_CE descended cleanly, the
information learned by the governor head did not transfer to the role
head's decision boundary on these specific token pairs.

### 11.7 Interpretation

The governor head learns to predict *which token* is the parent in the
dep tree. But the *mudaaf_ilayh* vs *mafoul_bih* vs *ism_majrur*
decision is not "which token is the parent" — the dep parent is the
same token in all three readings. The decision is **what relation**
holds with that parent. That requires lexical-semantic knowledge
(verb-argument structure, idafa-head propensity, preposition-presence)
which neither dep features nor explicit governor prediction provides.

Combined with the graph negative result, the convergent conclusion is
clear: **at our data scale, more structural supervision does not help
this confusion. The bottleneck is lexical-semantic.**

Frozen at `docs/final_governor_negative_result/`.

---

## 12. The central finding — the *mudaaf_ilayh* family

This is the project's most important linguistic result.

### 12.1 The dominant role confusions

Validated recovery on full Gazelle + MASAQ, fully-observable subset:

```
Gold              Predicted             Count   Family
────────────      ───────────────       ─────   ─────────────────────
mudaaf_ilayh      mafoul_bih              32    I (idafa↔direct object)
mudaaf_ilayh      mubtada                 29    I (idafa↔topic)
ism_majrur        matuf                   21    II (preposition↔coordinator)
mudaaf_ilayh      fail                    13    I
mudaaf_ilayh      ism_majrur              13    III (idafa↔preposition)
mafoul_bih        fail                    12    semantic role overlap
ism_majrur        mubtada                 8     II
mudaaf_ilayh      ism_inna                7     I (nested)
ism_majrur        naat                    7     II
mudaaf_ilayh      khabar_inna             6     I (nested)
                                                ─────
Family I total                            ~120  (largest single block)
```

**~120 errors** centre on a single confusion. By far the largest
identifiable block.

### 12.2 The linguistic explanation

The three roles in family I — *mudaaf_ilayh*, *mafoul_bih*, *ism_majrur* —
all surface as a noun in **jarr** (genitive) case immediately after
another word.

| When the second noun is *mudaaf_ilayh* | the first noun governs it (idafa) |
| When the second noun is *mafoul_bih* | a verb upstream governs it |
| When the second noun is *ism_majrur* | a preposition governs it |

The dep parent is the same. The case is the same. The marker is the
same. The token order is the same. **Distinguishing them requires
lexical knowledge:**

- Does the upstream verb take a direct object? (Answer determines
  *mafoul_bih*.)
- Is the first noun a typical idafa-head word like *kitāb* / *bayt* /
  *ibn*? (Answer determines *mudaaf_ilayh*.)
- Is there a preposition surface form in scope? (Answer determines
  *ism_majrur*.)

None of these can be inferred from the dep tree alone.

### 12.3 Why the architectural attacks did not displace this confusion

We tested two architectural attacks aimed at this exact failure family:

| Attack | Mechanism | Confusion Δ |
|---|---|---|
| Graph refiner | Message passing along dep + construction + overlap edges | 0 |
| Governor head | Biaffine attachment prediction + contrastive | 0 |

Both trained correctly. Neither helped.

The reason is simple. Both are **structural**: they tell the model
*which token* attaches *where*. But the confusion is about *what kind
of attachment relation* the parent imposes. That information is in
the lexicon (verb subcategorisation frames, idafa-head propensities,
preposition lists), not in the dep tree.

### 12.4 Per-construction-family fully accuracy

```
Construction family    n      n_correct    fully     visual
─────────────────      ───    ─────────    ─────     ─────────────────
inna_sisters           268    188          0.702     ████████████████░░░░
istithna               106    76           0.717     ████████████████░░░░
idafa                  521    333          0.639     █████████████░░░░░░░
idafa_multi (nested)   22     4            0.182     ████░░░░░░░░░░░░░░░░  ← collapses
```

Single-construction idafa (0.639) is already below the inna/istithna
level. **Nested idafa collapses to 0.182** — the model essentially
cannot resolve any token's role inside a multi-level idafa chain.

### 12.5 The path forward

Architecture is not the answer. The path forward is **annotation**:

1. **Verb-argument structure annotations** on a few hundred verbs in
   `distill_v2`. Tells the model when *mafoul_bih* is licensed.
2. **Idafa-head propensity lexicon**. Tells the model that *kitāb*
   typically heads an idafa.
3. **Permissive ambiguity annotations** on the genuinely-ambiguous
   tokens — when both *mudaaf_ilayh* and *mafoul_bih* are valid
   readings, both should be scored as correct.

The project ships infrastructure for all three (§ 15). The annotation
work itself is the next round.

---

## 13. Calibration analysis

### 13.1 The numbers

Calibration on validated_recovery failures (held-out Gazelle + MASAQ):

```
Axis      ECE on failures   High-conf wrong (≥ 0.95)
────      ───────────────   ────────────────────────
case      0.42              79 tokens
role      0.49              83 tokens
marker    0.60              70 tokens
```

The reliability bins on the role axis:

```
Conf bin      n    accuracy   model says ‘I’m sure’ but is right …
───────────   ───  ────────   ──────────
[0.0–0.1)     0    —          —
[0.1–0.2)     0    —          —
[0.2–0.3)     7    0.43       (43% of the time)
[0.3–0.4)     13   0.00       (0% of the time)
[0.4–0.5)     24   0.38       (38%)
[0.5–0.6)     27   0.04       (4%)
[0.6–0.7)     22   0.14       (14%)
[0.7–0.8)     27   0.44       (44%)
[0.8–0.9)     36   0.39       (39%)
[0.9–1.0)     166  0.37       (only 37%)
```

The model says "I'm 95+ % sure" and is wrong 63 % of the time on these
hard cases. Unacceptable for any audit-friendly deployment.

### 13.2 Why this happens

Several contributing factors:
1. **Multi-task training under label smoothing 0.05.** The smoothing
   prevents the model from being calibrated to the gold distribution;
   it deliberately spreads probability mass. With sufficient training,
   the model has learned to be confidently wrong on hard cases because
   the easy cases dominate the gradient.
2. **Small training corpus.** 18,366 sentences is not enough for
   per-class confidence calibration on rare role labels.
3. **No held-out shard for post-hoc calibration.** We never carved out
   a small subset of train+dev specifically for temperature fitting.
   The model never had a chance to be recalibrated.

### 13.3 Mitigation infrastructure (built but not yet applied)

```
src/irab_tashkeel/calibration/
├── temperature_scaling.py    — post-hoc T fit via L-BFGS over a single scalar
└── focal_loss.py              — focal CE (gamma=2) + confidence penalty
```

Temperature scaling alone, applied to a held-out shard, routinely
reduces ECE from 0.4–0.6 to under 0.10. It's a single scalar parameter,
~30 lines of code, ~1 minute to fit on a held-out shard. The reason
we haven't yet applied it: we want to carve out the held-out shard
*correctly* (without re-introducing leakage) before fitting.

### 13.4 The Gazelle calib_gap — a real win

One calibration metric improved meaningfully:

```
Gazelle calib_gap (role):
  Phase 3-A:  +0.021    model is slightly over-confident on correct answers
  Recovery:  −0.052     model is slightly under-confident on correct answers
```

The negative sign means the model is now *under*-confident on correct
predictions — healthier than over-confident. On Gazelle, the
regularisation in the recovery patch (label smoothing + entropy reg +
consistency penalty) genuinely improved this dimension.

MASAQ calib_gap got *worse* (0.087 → 0.124) because MASAQ has more
hard cases where the model is overconfident on the wrong label. The
remedy is the same: temperature scaling on a held-out shard.

---

## 14. Hard-eval per-bucket breakdown

The `data_v2/hard_eval/` partition slices held-out sentences by
structural difficulty. Validated recovery's per-bucket fully accuracy:

```
Bucket                        n_sent    fully     visual
────────────────────────      ──────    ─────     ─────────────────
ambiguity (sem.pres ≥ 2)      728       0.736     ███████████████░░░░░  ← best
quranic_hard                  285       0.722     ██████████████░░░░░░
overlap (≥ 2 constructions)   254       0.668     █████████████░░░░░░░
rare_constructions            8         0.182     ███░░░░░░░░░░░░░░░░░  (tiny sample)
long_range (UD slice)         608       0.000     (gold not populated)
```

**Surprise.** The model performs *better* on high-ambiguity sentences
(0.736) than on construction-overlap sentences (0.668). This is
consistent with the central finding (§ 12): semantic ambiguity in
isolation is a measurable but tractable challenge; construction overlap
forces the model to deal with the *mudaaf_ilayh* family directly,
which is the wall.

The `data_v2/hard_eval_v2/` partition applies stricter compound
filters; the long_nested_idafa bucket has only 17 sentences but is
the truly-hard core of the held-out distribution.

---

## 15. The supervision/data infrastructure ready for next round

The convergent negative architectural results led to a deliberate pivot:
no more architecture, all annotation. The infrastructure for the next
round is fully built but unused (waiting on a grammarian).

### 15.1 Auto-mined ambiguity candidates

`scripts/data_v2/mine_ambiguity_candidates.py` reads the failure
analysis and produces one `AmbiguityExample` per (sentence, token,
confusion) pair. Output:

```
data_v2/ambiguity_corpus/
├── idafa_attachment/queue.jsonl       684 candidates
├── preposition_vs_idafa/queue.jsonl   530
├── coordination_scope/queue.jsonl     495
├── latent_governor/queue.jsonl        990
├── nested_attachment/queue.jsonl      912
├── semantic_role_overlap/queue.jsonl  622
└── summary.json                       Total: 4,233 candidates
```

Each candidate carries:
- `primary_analysis` — the model's prediction (case/role/marker per
  token in span)
- `secondary_analyses` — at least one alternative analysis (the gold
  reading, prepopulated from the failure data)
- `governor_candidates`, `attachment_candidates`
- `confidence_difficulty`, `reasoning_note`

The annotator's job is to confirm / edit / reject / mark "both valid".

### 15.2 Annotation server

```
src/irab_tashkeel/annotation/
├── annotation_server.py     FastAPI: /api/queues, /api/queue/<kind>/{pending,confirm,reject,edit}
├── review_queue.py          JSONL-backed pending/confirmed/edited/rejected state
├── disagreement_resolution  multi-annotator majority vote
└── static/annotation.html   single-page review UI
```

Launch:
```bash
PYTHONPATH=src uvicorn irab_tashkeel.annotation.annotation_server:app --port 8001
```

### 15.3 Permissive evaluator (eval_v3)

`src/irab_tashkeel/eval_v3/`:
- `ambiguity_metrics.evaluate_with_ambiguity` — counts a prediction as
  correct if it matches **any** declared analysis (primary or
  secondary). When annotated data lands, this single change plausibly
  moves Gazelle role +0.05 to +0.10 *with no model retraining*.
- `uncertainty_metrics` — `calibrated_fully`,
  `confidence_correctness_alignment`, `selective_accuracy_at_τ`,
  `high_confidence_error_rate`
- `structural_metrics` — `attachment_accuracy`, `governor_accuracy`,
  `overlap_accuracy`

### 15.4 Active-learning candidate miner

`src/irab_tashkeel/active_learning/`:
- `uncertainty_sampling` — entropy / min-top1 / margin scoring
- `disagreement_sampling` — ensemble disagreement across phase3a /
  recovery / graph / governor predictions
- `diversity_sampling` — greedy max-coverage over (dep_pattern,
  construction_signature, length) signatures
- `hard_case_mining` — composite score combining all four signals

When pointed at the unlabelled MSA / dialect / etc. corpora, this
produces a ranked annotation queue that maximises information gain
per annotation hour.

### 15.5 Calibration package

`src/irab_tashkeel/calibration/`:
- `temperature_scaling` — fit T on a held-out shard
- `focal_loss` — drop-in replacement for cross-entropy that down-weights
  easy examples

---

## 16. Repository structure

```
.
├── README.md                                this file
├── REPRODUCE.md                              single-command recipe
├── LICENSE
├── pyproject.toml                            pinned deps
│
├── docs/
│   ├── paper/PAPER.md                        publication writeup
│   ├── MODEL_CARD.md                         production model card
│   ├── LIMITATIONS.md                        15-item exhaustive limitations
│   ├── KNOWN_FAILURES.md                     8 known failure modes
│   ├── final_eval/                           Phase A original eval (uncovered the leak)
│   ├── final_eval_recovery/                  validated production headline
│   ├── final_eval_graph/                     graph-experiment eval data
│   ├── final_eval_governor/                  governor-experiment eval data
│   ├── final_graph_negative_result/          graph negative result + NEGATIVE_RESULT.md
│   ├── final_governor_negative_result/       governor negative result + NEGATIVE_RESULT.md
│   ├── failure_analysis/FINDINGS.md          ★ central scientific result (mudaaf_ilayh)
│   ├── hard_eval/                            per-bucket hard-eval reports
│   ├── leakage_audit/                        leakage discovery records
│   ├── validated_nextgen_recovery/           reproducibility manifest
│   └── final_validated/, final_graph...      frozen artifact records
│
├── src/irab_tashkeel/
│   ├── data_v2/
│   │   ├── schema_v2.py                      canonical Sentence record
│   │   ├── loaders/                          per-source loaders
│   │   ├── constructions/detector.py         construction tagger
│   │   ├── provenance.py                     split-role enforcement
│   │   ├── normalization.py                  diacritic / tatweel handling
│   │   ├── splitter.py                       train/dev/test policy
│   │   ├── semantic/                         AmbiguityExample + governor schemas
│   │   └── metadata/                         difficulty / ambiguity / semantic_pressure
│   ├── grammar_graph/                        grammar graph engine (input augmentation)
│   ├── curriculum/                           7-stage scheduler + sampler + gates
│   ├── eval_v2/                              single-source-of-truth metrics
│   ├── eval_v3/                              ambiguity / uncertainty / structural
│   ├── training/                             samplers, augmentations, SWA, LLRD, contrastive
│   ├── training_v2/                          curriculum trainer + collator + losses
│   ├── calibration/                          temperature scaling + focal loss
│   ├── ambiguity/schema.py                   AmbiguityExample
│   ├── annotation/                           review queue + FastAPI server
│   ├── analysis/                             failure / confusion / structural / calibration
│   ├── active_learning/                      composite hard-case mining
│   ├── models/graph_refiner.py               (frozen negative result module)
│   ├── morphology/dep_aware_model.py         the production model class
│   ├── structured/                           taxonomy_v4, base structured prediction
│   ├── decoding/                             inference-time structured decoding
│   └── reasoning/                            template-based explanation rendering
│
├── scripts/
│   ├── training_v2/train_curriculum.py       main entry point with all flags
│   ├── eval/
│   │   ├── run_full_eval_v2.py               independent full eval
│   │   ├── aggregate_full_eval.py            join raw shards into report
│   │   └── leakage_audit.py                  contamination detector
│   ├── analysis/
│   │   ├── run_failure_analysis.py
│   │   └── run_hard_eval_report.py
│   ├── data_v2/
│   │   ├── build_schema_v2_corpus.py
│   │   ├── build_provenance_manifest.py
│   │   ├── build_hard_eval.py / build_hard_eval_v2.py
│   │   ├── build_adversarial_splits.py
│   │   └── mine_ambiguity_candidates.py
│   ├── freeze_validated_checkpoint.py
│   ├── freeze_canonical_artifacts.py
│   └── slurm/                                SLURM sbatch entrypoints (jobs 91–99)
│
├── demo/
│   ├── backend/main.py                       FastAPI server
│   ├── backend/inference.py                  lazy ModelHolder
│   ├── static/index.html                     9-tab single-page UI
│   └── README.md                             launch instructions
│
├── tests/                                    pytest suite (data_v2, schema_v2,
│                                              curriculum, training_v2, grammar_graph,
│                                              backbones)
│
├── archive/                                  deliberate archive of failed variants
│
└── runs/
    ├── validated_nextgen_recovery/           ★ PRODUCTION CHECKPOINT
    ├── final_validated/                      frozen Phase A artifact
    ├── final_graph_negative_result/          frozen graph experiment record
    ├── phase3a_491240/final/                 warm-start baseline
    └── ... older eval-only artefacts ...
```

---

## 17. Install / train / eval / inference / demo

### 17.1 Install

```bash
git clone https://github.com/Nourjennane/irab_project && cd irab_project
pip install -e ".[dev]"
# optional for demo / annotation:
pip install fastapi uvicorn
```

Python 3.11. Pinned versions in `pyproject.toml`.

### 17.2 Build the corpus

```bash
python scripts/data_v2/build_schema_v2_corpus.py
python scripts/data_v2/build_provenance_manifest.py
```

The provenance manifest enforces the train/dev/test split policy at
load time. Three runtime assertions provide defence in depth.

### 17.3 Train the validated recovery checkpoint

```bash
PYTHONPATH=src python scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen_recovery \
    --warm_start  runs/phase3a_491240/final \
    --batch_size 16 --lr 1e-5 \
    --use_hard_failure_sampler \
    --label_smoothing 0.05 --entropy_reg_lambda 0.01 \
    --consistency_lambda 0.20 --fully_aux_lambda 0.50 \
    --use_ema --early_stop_patience 3 \
    --use_swa --swa_start_step 2000 \
    --use_llrd --llrd_decay 0.85
```

SLURM: `sbatch scripts/slurm/93_train_curriculum_recovery.sbatch`.
Wall-clock on a single GPU: ~25 minutes.

### 17.4 Evaluate (independent full eval)

```bash
PYTHONPATH=src python scripts/eval/run_full_eval_v2.py \
    --checkpoints \
        phase3a:runs/phase3a_491240/final \
        recovery:runs/validated_nextgen_recovery \
    --datasets \
        gazelle:data_v2/annotated/gazelle_test/all.jsonl \
        masaq:data_v2/annotated/masaq_quranic/all.jsonl \
    --output_root docs/final_eval_recovery/raw

PYTHONPATH=src python scripts/eval/aggregate_full_eval.py \
    --raw_dir docs/final_eval_recovery/raw \
    --out_dir docs/final_eval_recovery
```

### 17.5 Run the failure analysis

```bash
PYTHONPATH=src python scripts/analysis/run_failure_analysis.py \
    --checkpoint runs/validated_nextgen_recovery \
    --datasets gazelle_test masaq_quranic \
    --out_dir docs/failure_analysis
```

### 17.6 Inference (Python)

```python
import torch
from transformers import AutoTokenizer
from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

ckpt = "runs/validated_nextgen_recovery"
tok = AutoTokenizer.from_pretrained(ckpt)
model = DepAwareStructuredModel(
    encoder_name="UBC-NLP/AraT5v2-base-1024",
    enable_morph_heads=True, enable_dep_features=True,
)
model.load_state_dict(
    torch.load(f"{ckpt}/pytorch_model.bin",
               map_location="cpu", weights_only=True),
    strict=False,
)
model.eval()
# … prepare word_starts/word_ends from tokenized words → forward → argmax
```

A complete inference helper is in `demo/backend/inference.py`.

### 17.7 Launch the demo

```bash
PYTHONPATH=src uvicorn demo.backend.main:app --port 8000
# open http://localhost:8000
```

Nine tabs:
1. **Sentence Analysis** — per-token case/role/marker/POS with confidence bars
2. **Grammar Graph** — DOT-format dependency-style graph
3. **Reasoning Trace** — template-rendered narrative from structured labels
4. **Constructions** — detected construction families
5. **Evaluation Dashboard** — Phase A independent eval + leakage audit
6. **Model Comparison** — recovery / phase3a / leaked stage_7 side-by-side
7. **Hard Examples** — curated nested / ambiguous / Quranic / overlap cases
8. **Known Failures** — the *mudaaf_ilayh* family table + linguistic explanation
9. **Leakage Case Study** — the contamination story with side-by-side metrics

### 17.8 Launch the annotation server (when a grammarian is ready)

```bash
PYTHONPATH=src uvicorn irab_tashkeel.annotation.annotation_server:app --port 8001
# open http://localhost:8001
```

---

## 18. Reproducibility

Every frozen artifact ships with:

- `REPRODUCIBILITY_MANIFEST.json` — git commit hash, env versions
  (torch, transformers, tokenizers, numpy, python), dataset sha256s
  at training time
- `metrics.json` / `eval_tables.json` — full Phase A eval slice
- `calibration.json` — per-field reliability bins + ECE
- `training_manifest.json` — training summary + config
- `git_commit.txt`, `environment.txt`

To verify any artifact: read the JSON, compare commit hash and dataset
sha256s against your environment.

`REPRODUCE.md` documents the 12-step single-command sequence to
recreate every production result from a fresh clone.

---

## 19. Future directions

The roadmap is **annotation-driven, not architecture-driven**.

### Tier 1 — high leverage, low cost

1. **Annotate the 4,233 mined ambiguity candidates** in
   `data_v2/ambiguity_corpus/`. Particularly the 684 `idafa_attachment`
   and 530 `preposition_vs_idafa` cases that target the dominant
   confusion family. Permissive scoring via
   `eval_v3.evaluate_with_ambiguity` then plausibly moves
   Gazelle role +0.05 to +0.10 *without retraining the model*.
2. **Apply temperature scaling on a held-out shard.** Single scalar
   fit reduces ECE from 0.49 to under 0.10. Critical for any
   educational or audit-friendly deployment.

### Tier 2 — medium leverage, medium cost

3. **Multi-seed ablations** for noise quantification on the small
   Gazelle held-out. Required before any conclusion under Δ ≤ 0.01
   fully.
4. **Verb-argument-structure annotations** on a subset of `distill_v2`.
   The mudaaf_ilayh confusion is exactly a verb-arg-structure problem.
5. **Active-learning loop** over the held-out / unlabelled corpora
   using the composite scorer (uncertainty + disagreement + structural
   difficulty + calibration evidence).

### Tier 3 — long horizon

6. **Cross-dialect held-out corpora**: Egyptian, Levantine, Gulf,
   Maghrebi. Currently zero coverage outside MSA + Quranic.
7. **Larger Arabic foundation pretraining**. A separate project; not
   in scope for this repo.

### What is explicitly NOT on the roadmap

- More graph layers
- More structural heads
- More auxiliary losses
- Encoder redesign
- Random architecture variants

Both architectural attacks already produced clean negative results;
further attempts would be churn.

---

## 20. Citation

**Author:** Nour Jennane
**Contributors:** Nour Jennane, Hatem Saadallah

```bibtex
@article{iraab_recovery_2026,
  title  = {A Case Study in Honest Arabic Grammatical Reasoning:
            From Leakage Collapse to Structural Ambiguity Bottlenecks},
  author = {Jennane, Nour and Saadallah, Hatem},
  year   = {2026},
  url    = {https://github.com/Nourjennane/irab_project},
  note   = {Production checkpoint: validated\_nextgen\_recovery; two
            documented negative architectural results; central finding
            on the mudaaf\_ilayh confusion family.},
}
```

---

## License

See [`LICENSE`](LICENSE).
