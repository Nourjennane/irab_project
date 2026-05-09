# Phase R2 — Retrieval-Guided Structural Reasoning

> **The new core direction after Phase R-C closed.** Phase R established
> that retrieval finds structurally similar examples (MASAQ idafa_multi
> +8.5 fully proves the mechanism is real) but **shallow soft-bias
> blending is insufficient for catastrophic Gazelle failures** (kāna 0%
> → 0%, calib gap −0.533 → −0.884). The retrieval itself is validated;
> what's missing is **explicit structural reasoning over the retrieved
> evidence**.

## 0. Strategic frame — what Phase R proved and didn't

**Proved:** retrieval transfers useful structural information when the
retrieval-pool distribution matches the test distribution (MASAQ:
+0.7 overall, +8.5 idafa_multi specifically). The pool is genuinely
new orthogonal information vs Phase 3-A's parametric memory.

**Did not solve:** the catastrophic Gazelle constructions (kāna 0%,
istithnāʾ 0%). These are not failing because the model lacks examples
— the retrieval pool has 540 kāna instances and finds them. They are
failing because:

1. The model's encoder produces logits where the *correct* iʿrāb
   labels have very low probability for these test sentences.
2. Adding `λ · log(prior)` of magnitude ~0.3 cannot flip an argmax
   when the gap between correct and predicted label is large.
3. The retrieval prior gets averaged across top-k, which can dilute
   the signal when the retrieved examples have heterogeneous patterns
   (e.g. some kāna_completion vs kāna_negation in the same retrieval).

**The bottleneck is now: REASONING OVER RETRIEVED STRUCTURES.**

The model needs to explicitly:
- Recognise the construction family
- Extract the structural transformation rule from retrieval consensus
- Apply the rule to the current sentence symbolically
- Generate a reasoning trace explaining the transformation

This is **construction grammar applied as inference-time correction**,
guided by retrieval consensus rather than hand-coded rules.

## 1. Anti-pattern (per the user's 2026-05-09 redirects)

**DO NOT:**
- Add new decoder tricks
- Add a learnable conditioning layer
- Add another CRF
- Add another taxonomy split
- Add output-bias variants (Phase 5/6 closed)
- Add hard symbolic training constraints (Phase 8 deleted)

**DO:**
- Build explicit structural reasoning over retrieved constructions
- Compare dependency patterns between current and retrieved
- Compare role transitions
- Model grammatical operations as transformation rules
- Generate reasoning traces

## 2. Architecture overview

```
                test sentence
                      │
         ┌────────────┼─────────────────┐
         ▼            ▼                 ▼
    Phase 3-A     Stanza UD        construction
    forward      parser            detector
    (logits +    (dep tree)
    pooled_irab)
         │            │                 │
         └────────────┴─────────┬───────┘
                                ▼
                  Phase R memory retrieval
                  (per-construction top-k)
                                │
                                ▼
              ┌────────────────────────────────┐
              │  STRUCTURAL REASONER          │
              │  (per-construction-family)    │
              │                                │
              │  1. align current ↔ retrieved │
              │     by surface + morph         │
              │  2. extract transformation:   │
              │     consensus over top-k      │
              │     for each aligned position  │
              │  3. validate consensus:       │
              │     ≥3/5 retrievals agree?    │
              │  4. compute confidence:        │
              │     consensus rate × cosine   │
              │  5. emit transformation rule   │
              └────────────────────────────────┘
                                │
                                ▼
              ┌────────────────────────────────┐
              │  CONFIDENCE GATING             │
              │                                │
              │  if confidence ≥ τ_high:      │
              │     OVERRIDE Phase 3-A pred    │
              │     for the span (symbolic)    │
              │  elif confidence ≥ τ_med:     │
              │     STRONG additive bias       │
              │     (λ × 5)                    │
              │  else:                         │
              │     FALLBACK to Phase 3-A      │
              │     (no override / weak bias)  │
              └────────────────────────────────┘
                                │
                                ▼
                final structured prediction
                + reasoning trace
                  ([span, transformation, confidence,
                    consensus_rate, retrieved_examples])
```

The key new module is the **structural reasoner**, which sits between
retrieval and prediction. It's per-construction-family because each
family has its own canonical transformation pattern.

## 3. Construction-specific reasoners

Per-family reasoner objects implement a common interface:

```python
class ConstructionReasoner(Protocol):
    family: str
    def reason(
        self,
        query_span: List[Dict],            # current sentence span items
        retrieved: List[RetrievalHit],     # top-k from Phase R memory
        query_dep_subtree: DepSubtree,     # current dep tree restricted to span
    ) -> ReasoningOutput:
        ...

@dataclass
class ReasoningOutput:
    predicted: List[Dict]                  # one prediction per word position in span
    confidence: float                      # 0..1 — how strong is the consensus?
    consensus_rate: float                  # fraction of retrievals that agree
    rule: str                              # human-readable rule, e.g.
                                            # "kana_completion: ism→raf, khabar→nasb"
    aligned_indices: List[Tuple[int, int]] # query word i ↔ retrieved word j
    reasoning_trace: str                   # natural-language explanation
```

### 3.1 KanaReasoner (highest priority — 0% Gazelle fully)

Construction: kāna + ism + khabar (3-word span).

Transformation rule (from grammar):
- particle: role=`fil`, case=`mabni`, marker depends on particle conjugation
- ism (subject): role=`ism_kana`, case=`raf`, marker depends on noun morph
- khabar (predicate): role=`khabar_kana`, case=`nasb`, marker depends on noun morph

The reasoner:
1. Aligns current span [particle, ism, khabar] with retrieved spans of
   the same length and same particle group
2. Reads off the labels from each retrieved instance for each aligned
   position
3. Computes consensus: at position 1 (ism), what fraction of retrievals
   agree on `(role=ism_kana, case=raf)`?
4. If consensus ≥ 3/5 (or fraction ≥ 0.6) on each position: emit the
   construction-grammar rule with high confidence
5. Marker is computed *symbolically* from the current word's morphology
   (gender/number/definiteness), not from retrieval consensus. This
   handles the morph mismatch between retrieved and current words.

### 3.2 IstithnaReasoner

Construction: main_clause + illa-particle + mustathna.

Transformation rules:
- particle (إلا): role=`harf_other`, case=`mabni`
- mustathna AFTER إلا: role=`mafoul_other`, case=`nasb` (positive context)
- particle غير/سوى: role=`mafoul_other`, case=`nasb`, AND مضاف (head of iḍāfa)
- noun AFTER غير/سوى: role=`mudaaf_ilayh`, case=`jarr`

The reasoner detects which sub-pattern by particle surface, applies
the appropriate rule. Retrieval is used to validate the rule applies
to the current sentence's morphological context.

### 3.3 MawsoolReasoner

Construction: head_noun + mawsool + relative_clause.

Transformation rules:
- mawsool (الذي / التي / etc.): role=`other` (or `naat` if attributive),
  case=`mabni`, marker=`sukun` typically
- the role is determined by what the relative pronoun MODIFIES — this
  is the structural reasoning: look at the head noun's role in the
  matrix clause, then determine the mawsool's role accordingly

### 3.4 InnaReasoner (lower priority — baseline already 27.3%)

Construction: inna-particle + ism + khabar (3-word span).

Transformation rule (mirror of kāna with reversed cases):
- particle: role=`harf_nasb` or `harf_other`, case=`mabni`
- ism inna: role=`ism_inna`, case=`nasb`
- khabar inna: role=`khabar_inna`, case=`raf`

### 3.5 QuranicProxyReasoner

Construction: qad/idh/lamma + verb + (subj)

Transformation rules:
- qad: role=`harf_other`, case=`mabni`, marker=`sukun`; following verb
  unchanged (qad doesn't shift the verb's case/marker)
- idh: role=`dharf` (temporal adverb), case=`mabni`
- lamma: similar to idh, requires perfect verb following

### 3.6 Reasoners NOT yet built (defer)

- iḍāfa: already at 33% Gazelle fully, well-handled by Phase 3-A. No
  reasoner needed.
- iḍāfa_multi: same — already at 30% Gazelle, +8.5 on MASAQ from R-C.

## 4. Structural alignment algorithm

Given a query span `q_items` and a retrieved instance `c_items`, align
their word positions:

**Surface-position alignment** (default for 3-word particle constructions):
- Position-by-position alignment when both spans have the same length
- particle aligns to particle (always)
- subsequent positions align by relative index

**Morphology-aware alignment** (when spans have different lengths):
- Match query word i to retrieved word j minimizing morph distance:
  `sum(I[q.gender != c.gender] + I[q.number != c.number] + I[q.definite != c.definite])`
- Use a simple greedy assignment (Hungarian for >5 words; not needed
  for our short spans)

**Dependency-path alignment** (for 4+ word spans like iḍāfa chains):
- Compare dep paths from particle/head to each word
- Align words with same dep path length + same DEPREL sequence

For Phase R2 first pass, **start with surface-position alignment only**.
Defer morphology-aware and dep-path alignment to Phase R2.1 if needed.

## 5. Transformation extraction (consensus over top-k)

For each aligned position `i` in the query span:

```
case_votes[i]   = Counter(c.items[align(i)].case   for c in retrieved if align(i) is valid)
role_votes[i]   = Counter(c.items[align(i)].role   for c in retrieved if align(i) is valid)
marker_votes[i] = Counter(c.items[align(i)].marker for c in retrieved if align(i) is valid)

predicted_case[i]   = argmax(case_votes[i])
predicted_role[i]   = argmax(role_votes[i])
consensus_case[i]   = top_count / total_count    # fraction of retrievals agreeing on the top vote
consensus_role[i]   = ...
consensus_marker[i] = ...
```

**Marker is special**: instead of voting, compute marker symbolically
from the word's morphology + predicted case (using existing schema):

```
predicted_marker[i] = derive_marker_from_morph(
    word=q.items[i],
    case=predicted_case[i],
    gender=q.items[i].gender,
    number=q.items[i].number,
    definite=q.items[i].definite,
)
```

This handles the surface-form mismatch between retrieved and current
sentences (where the retrieved morph differs from current morph).

## 6. Confidence + gating

```python
overall_confidence = (
    mean(consensus_case[i] for i in span) * 0.4 +
    mean(consensus_role[i] for i in span) * 0.4 +
    mean(top_k_cosine_scores) * 0.2
)
```

Three gating tiers:
- `τ_high = 0.75` → SYMBOLIC OVERRIDE: replace Phase 3-A predictions
  for the construction span with the reasoner's predictions.
- `τ_med = 0.50` → STRONG additive bias: λ × 5 = 1.5 multiplier on the
  log-prior bias from R-C.
- below τ_med → FALLBACK: keep Phase 3-A predictions for this span.

The high-confidence override is the new mechanism vs Phase R-C. It's
allowed because it's NOT a learnable conditioning layer or a decoder
trick — it's an explicit symbolic transfer gated by consensus
strength, generated at inference from retrieval data.

## 7. Reasoning trace generation

For each construction span the reasoner emits a natural-language trace:

```
Span [0, 3] = "أصبح الطالبُ مجتهداً"
    Detected: kana_sisters / kana_completion ("أصبح")
    Retrieved 5 analogues, top cosine 0.84:
      1. "كان الكتابُ مفيداً"          (cosine 0.84, sym 0.67)
      2. "أصبح المعلمُ مخلصاً"         (cosine 0.81, sym 1.00)
      3. ...
    Consensus:
      pos 0 (particle): role=fil(5/5), case=mabni(5/5)
      pos 1 (ism):      role=ism_kana(5/5), case=raf(5/5), marker=damma_visible(4/5)
      pos 2 (khabar):   role=khabar_kana(5/5), case=nasb(5/5), marker=tanween_fath(3/5)
    Confidence: 0.93 (≥ τ_high) → SYMBOLIC OVERRIDE
    Rule: "kana_completion: particle is mabni, ism is raf, khabar is nasb;
           markers derived from word morphology"
```

This trace is the basis for Phase 11 (explanation engine).

## 8. Inference flow

```python
def predict_with_structural_reasoning(sentence, base_pred, memory, reasoners):
    # 1. Phase 3-A forward
    raw_pred = base_pred.predict_sentence(sentence)
    pooled_irab = base_pred.last_pooled  # cached

    # 2. Detect constructions
    constructions = detect_constructions_in_record(
        record_from_pred(raw_pred)
    )

    final = raw_pred.copy()
    traces = []

    for cspan in constructions:
        family = cspan["construction"]
        reasoner = reasoners.get(family)
        if reasoner is None:
            continue   # families without reasoners (idafa) skip

        # 3. Retrieve
        query = build_signature(record_from_pred(raw_pred), cspan, sentence_idx=-1)
        span_emb = pooled_irab[cspan.span[0]:cspan.span[1]].mean(dim=0)
        hits = memory.retrieve(query, span_emb, k=5)
        if len(hits) < 3:
            continue   # not enough analogues for consensus

        # 4. Reason
        reasoning = reasoner.reason(
            query_span=raw_pred.items[cspan.span[0]:cspan.span[1]],
            retrieved=hits,
            query_dep_subtree=None,   # phase R2 first pass: no dep-tree alignment
        )

        # 5. Apply per gating tier
        if reasoning.confidence >= TAU_HIGH:
            final = apply_override(final, cspan.span, reasoning.predicted)
        elif reasoning.confidence >= TAU_MED:
            final = apply_strong_bias(final, cspan.span, reasoning.predicted, lambda_=1.5)

        traces.append((cspan, reasoning))

    return final, traces
```

## 9. Implementation surface

New files:
- `src/irab_tashkeel/grammar_memory/structural_reasoner.py`
  - `ReasoningOutput` dataclass
  - `ConstructionReasoner` base class
  - `KanaReasoner`, `IstithnaReasoner`, `MawsoolReasoner`,
    `InnaReasoner`, `QuranicProxyReasoner`
  - `derive_marker_from_morph(word, case, morph)` symbolic helper
- `src/irab_tashkeel/grammar_memory/structural_predictor.py`
  - `StructuralReasoningPredictor` (extends RetrievalAugmentedPredictor)
  - Replaces the soft-bias step with reasoner-driven override / strong
    bias / fallback
- `tests/test_structural_reasoner.py` — unit tests for each reasoner
  on synthetic spans
- `scripts/structured/eval_phaseR2.py` — runs the new predictor on
  Gazelle + MASAQ with per-construction breakdown + reasoning trace
  dumps

Modified files:
- None to existing models (pure inference-side wrapper)
- None to Phase 3-A checkpoint
- None to Phase R memory build pipeline (re-uses existing memory)

## 10. Ablation plan (focused, gate-driven)

| Cell | Description | Gate role |
|---|---|---|
| R2-0 | sanity (all reasoners disabled, falls back to Phase 3-A) | must reproduce baseline |
| R2-A | KanaReasoner only (kāna sisters override + strong-bias) | did kāna fix? |
| R2-B | KanaReasoner + IstithnaReasoner | did istithnāʾ also fix? |
| R2-C | All target reasoners (kana, istithna, mawsool, inna, quranic) | full system |
| R2-D | R2-C with τ_high=0.6, τ_med=0.4 (looser thresholds) | gate-sensitivity |
| R2-E | R2-C with consensus-only (no symbolic marker derivation) | does morph-derived marker help? |

Run order: R2-0 (sanity) → R2-A (kāna gate) → R2-B (istithnāʾ gate)
→ R2-C (full) → R2-D / R2-E (sensitivity, optional).

## 11. Decision gate

Phase R2 ships if:

1. **Per-construction kāna_sisters fully ≥ +10 pp** on Gazelle vs
   Phase 3-A baseline (currently 0% → ≥10%). The +5 pp soft-target
   from Phase R didn't budge; for R2 we want a clean step change.
2. **Per-construction istithnāʾ fully ≥ +10 pp** on Gazelle (currently 0%).
3. **Overall Gazelle case + role + fully within ±1.5 pp** of Phase 3-A.
   Slightly looser than R-C's ±1 pp because per-construction reasoners
   may shift the surface metrics slightly even when targeted gains
   are real.
4. **MASAQ retention ≥ Phase 3-A** (no regression on the surface where
   R-C already showed mechanism gains).
5. **Per-construction calibration gap improves** for kāna (currently
   −0.533) and istithnāʾ (currently −0.291). At minimum, both must
   move toward zero or positive.

If 1+2+3+4+5 all pass: Phase R2 ships as the new inference pipeline
on top of Phase 3-A. Production lineage: rev 2 → Phase 1 → Phase 3-A
+ R2 reasoner.

If only 1 passes (kāna fixed, istithnāʾ not): ship per-construction
(KanaReasoner production-on, others archival).

## 12. Failure modes

1. **Insufficient retrieval consensus**: top-k retrievals disagree
   too much (no consensus ≥ 3/5). Mitigation: monitor consensus rate
   in trace; if <40% of construction spans achieve consensus, the
   reasoning approach is structurally too weak and we'd need either
   bigger memory or different alignment.

2. **Symbolic marker derivation fails**: the morph features for the
   current word don't fit a clean symbolic rule (e.g., `und` morph).
   Mitigation: per-marker fallback table; for `und` morph default to
   the consensus-voted marker.

3. **Override hurts iḍāfa** (a regression on the well-handled
   majority class). Mitigation: don't build an iḍāfa reasoner;
   the gating naturally falls back to Phase 3-A.

4. **High-confidence override gets it wrong on edge cases**: a kāna
   sentence where the test instance has unusual morph that the
   retrieval pool doesn't cover. Mitigation: confidence threshold
   scaling — symbolically-derived marker requires both consensus AND
   morph clarity (`und` features lower the confidence).

5. **Inference latency**: reasoning + retrieval per construction span.
   Should still be <200ms per sentence for our sizes. Profile if
   exceeds 500ms.

6. **Test contamination**: if any retrieval finds an instance that's
   actually in the training corpus AND is suspiciously close to the
   test sentence (test leakage). Mitigation: assert that the eval
   surface (Gazelle, MASAQ) is held out from the retrieval pool. The
   pool is `data/morph_v1_dep/train.jsonl` which is distill_v2 train
   only — no Gazelle, no MASAQ. Already guaranteed.

## 13. Compute estimate

| Step | HPC | Local |
|---|---:|---:|
| Build reasoners + tests | — | ~2h |
| Build StructuralReasoningPredictor | — | ~1h |
| R2-0 sanity (Gazelle + MASAQ) | ~5min | — |
| R2-A KanaReasoner | ~5min | — |
| R2-B + Istithna | ~5min | — |
| R2-C all reasoners | ~5min | — |
| R2-D sensitivity (optional) | ~5min | — |
| Per-construction analysis + writeup | — | ~1h |
| Paper integration | — | ~1h |
| **Critical path: R2-A KanaReasoner only** | **~10min** | **~3h** |
| **Full R2-C ship** | **~30min** | **~5h** |

The compute is small. The work is in carefully implementing the
reasoners. Total wallclock: ~5 h focused work + paper integration.

## 14. Expected gains

Conservative (target):
- kāna_sisters Gazelle fully: 0% → 15-30% (consensus-driven override
  on high-confidence kāna constructions; ~half of them should hit
  consensus given the 540-instance pool)
- istithnāʾ Gazelle fully: 0% → 15-30%
- Overall Gazelle fully: 25.2% → 27-30% (small lift from rare-class
  improvements)
- MASAQ: at least retain Phase R-C's +0.7 fully gain

Optimistic (best case):
- kāna_sisters Gazelle fully: 0% → 50-60% (most kāna constructions
  achieve clean consensus and override correctly)
- istithnāʾ: 0% → 30-50%
- Overall: 25.2% → 30-33%

Pessimistic:
- Consensus rate is too low for most construction spans → R2 falls
  back to Phase 3-A predictions on most constructions → results match
  baseline. R2 ships only the kāna reasoner if it works specifically.

## 15. Phase 11 (explanation engine) absorption

Phase 11 (explanation engine) is now naturally absorbed into Phase R2:
the `reasoning_trace` per span is the explanation. Each prediction
comes with:
- The detected construction family
- The top-k retrieved analogues (sentence + cosine + symbolic overlap)
- The transformation rule applied (e.g., "kana_completion: ism→raf,
  khabar→nasb")
- The consensus rate per word position
- The confidence tier (override / strong-bias / fallback)

A simple report generator can render these traces as Arabic-prose
rationales for each prediction. The "explanation engine" is no longer
a separate phase — it's the default output of Phase R2.

## 16. Out of scope for Phase R2 first pass

- Dependency-tree alignment (defer to R2.1 if needed)
- Morphology-aware alignment (Phase R2.1)
- Hungarian matching for long spans (Phase R2.1)
- Per-construction λ tuning (irrelevant — R2 uses gating tiers, not λ)
- Training-time integration of reasoning signal (would require
  retraining; Phase 3-A stays frozen per the architectural case study)
- Cross-construction reasoning chains (e.g. kāna-inside-istithnāʾ
  composition) — defer

## 17. Project status after Phase R2 ships

If R2-A passes: kāna sisters fixed → first Gazelle catastrophic
construction unblocked since Phase 3.

If R2-C ships: kāna + istithnāʾ + mawṣūl all fixed →
the per-construction breakdown table in REPORT.md goes from "0/0/0"
on the catastrophic constructions to non-zero across the board.

The project would have transitioned from:
- "structured classifier" (rev 2)
to:
- "encoder + dep features" (Phase 3-A)
to:
- "encoder + dep features + retrieval-guided structural reasoning"
  (Phase 3-A + R2)

This is the **hybrid neuro-symbolic grammatical reasoning system**
the project framing has been pointing at since the 2026-05-08 redirect.

## 18. Suggested improvements based on findings (beyond Phase R2)

Three additional levers consistent with the research thesis that the
Phase 3-A → R-C → R2 trajectory suggests:

### 18.1 Dependency-path-aware retrieval

Phase R uses single-vector cosine on span-mean-pooled embeddings. A
better retrieval signal: **dep-tree subgraph similarity**. Match
sentences by:
- DEPREL path from sentence root to construction head
- governor + governed UPOS sequences
- dep edge labels in a 2-hop neighborhood

This is more structural than embedding cosine and aligns with the
"explicit grammatical operation" direction. Estimated lift on top of
R2: another ~2-5 pp on rare constructions if implemented.

Defer to Phase R3 if R2-A passes the gate.

### 18.2 MSA-news distribution-targeted retrieval pool curation

The Phase R-C and Phase 39 results both showed Gazelle distribution
mismatch as the failure mode. The retrieval pool is from
distill_v2, which has its own surface distribution. **Curating the
pool** to be MSA-news-style (filter to sentences syntactically close
to Gazelle's surface patterns) might lift Gazelle without hurting
MASAQ.

Implementation: train a small MSA-vs-Quranic style classifier on the
distill_v2 corpus, weight retrieval scores by MSA-style probability,
or build separate Gazelle-targeted and MASAQ-targeted sub-pools.

### 18.3 Inference-time Stanza dep parsing

Phase 3 currently passes zero `dep_emb` at inference (the predictor
doesn't run Stanza on Gazelle inputs in the current iteration —
documented in Phase 3 §14.2 inference debug episode). Running Stanza
at inference would feed real dep features and likely improve
construction detection accuracy + structural alignment.

This is small data-engineering, no model changes. Estimated lift:
~1-3 pp via cleaner construction detection. Defer to Phase R2.1.

### 18.4 SUGGESTION (implement now): tighter retrieval pool curation
on Phase R-C BEFORE building R2

The Phase R-C result shows kāna calibration *worsens* (−0.533 →
−0.884). One hypothesis: the retrieval pool's kāna instances are
biased toward certain particles (e.g., synthetic kāna heavy in
"أصبح", "كان") that don't match Gazelle's actual instances.

Quick diagnostic before building R2:
- Inspect the 540 kāna instances in `data/grammar_memory/kana_sisters/instances.jsonl`
- Particle distribution: how many "كان" vs "أصبح" vs others?
- Source distribution: how many from distill_v2 vs from Phase 39
  synthetic? (Phase 39's synthetic kāna at 22% mix may be polluting
  the pool with template-narrow examples)

If the pool is dominated by Phase 39 synthetic kāna, **rebuilding the
pool from distill_v2 train ONLY (no synthetic) might already improve
the retrieval quality** — without needing R2's structural reasoning.
This is a 5-minute fix and worth running before committing to R2's
~5h implementation.

This is the most valuable concrete improvement I can suggest from
the findings: **diagnose and re-curate the retrieval pool first.**
If pool curation alone lifts kāna meaningfully, R2 might not need
all the structural-reasoning machinery. If it doesn't, R2's
explicit reasoning is the right next investment.
