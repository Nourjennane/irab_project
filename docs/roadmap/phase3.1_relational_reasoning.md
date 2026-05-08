# Phase 3.1 — Relational Reasoning Expansion

> **The new main scaling direction.** Phase 3-A (rev 2 + Phase 1 morph
> + Stanza UD dep as static input augmentation) shipped as production
> with case +3.0, marker +3.8, *fully* +0.7 vs Phase 1. The four-cell
> architectural case study (Phases 4a + 2 + 5 + 6, all rearrangements
> of the same supervision, all plateau or regress) confirmed that the
> next productive lever is **richer relational reasoning over the dep
> structure**, not more decoder hierarchy or stronger conditioning.

## 1. Strategic frame

After the Phase 3 result + the Phases 5, 6 negatives (closed
2026-05-08), the empirical pattern is robust: **at 296M / 6 epochs,
rearranging the same supervision plateaus or regresses; orthogonal
information sources unlock gain.** Phase 3 introduced UD dependency
edges — relational signal that morph + role taxonomy literally cannot
express — and that is what worked.

The natural next step is to do *more* with that relational signal,
not to add more rearrangement of the existing predictions. Phase 3-A
treats dep edges as **static input embeddings** (DEPREL one-hot +
HEAD direction + HEAD distance + governor's POS, concatenated and
projected). Phase 3.1 explores **dynamic relational reasoning** over
the dep structure: information propagates along the tree, neighbours
inform each other, clause boundaries gate context.

## 2. Anti-goals (per user 2026-05-08 redirect)

The following are explicitly **out of scope**:

- More FiLM variants (Phase 2 closed)
- Stronger CRFs (Phase 7 cancelled mid-flight by the redirect)
- Deeper output hierarchy (Phases 5, 6 closed)
- Additional hard symbolic constraints (the rev 2 four constraints
  stay; the 5-constraint stress test stays a documented negative)
- Joint training of morph heads with the new relational layers; morph
  heads stay **frozen** by default

The Phase 2 joint-dynamics finding (joint training of morph heads
under conditioning corrupts their representation) is the operational
reason for keeping morph frozen here.

## 3. Four candidate mechanisms

The relational reasoning branch should explore mechanisms that:
- Take the existing Phase 3-A `pooled_irab` features and dep edge data
  as input
- Output a richer per-word feature `pooled_rel ∈ R^768` that the iʿrāb
  decoders consume in place of `pooled_irab`
- Add ≤ 5M parameters (small relative to 296M encoder; keeps the
  intervention interpretable)
- Are individually toggleable so each can be ablated cleanly

### 3.1 Relation-aware self-attention (primary candidate)

A single self-attention layer over the per-word features, with
attention biased by dep edge type:

```
Q, K, V = Linear(pooled_irab)                  # standard QKV projections
edge_bias[i, j] = dep_edge_embed(deprel(i,j))  # per-edge type bias, learnable
attn[i, j] = softmax((Q[i]·K[j])/√d + edge_bias[i, j])
pooled_rel[i] = Σ_j attn[i, j] · V[j]
```

The dep edge type `deprel(i, j)` is computed from the offline-parsed
dep tree: if word `j` is the head of word `i`, `deprel(i, j) =
DEPREL(i)`; if `j` is governed by `i`, `deprel(i, j) = REVERSE_DEPREL(j)`;
otherwise (no direct dep edge), use a special `<no_edge>` bias.

**Why this is the primary candidate:** it directly biases attention
by the relational structure that morph + taxonomy can't express, on
top of the orthogonal-info gain Phase 3 already established. ~1M
parameters for a single attention layer. Identity-init (edge_bias
zero, Q/K/V copied from a small projection of `pooled_irab`) keeps
step 0 byte-equivalent to Phase 3-A.

### 3.2 Lightweight graph message passing

One round of message passing over the dep tree:

```
for each word i:
    parent_msg[i]   = MLP(pooled_irab[parent(i)])  if parent exists else 0
    children_msg[i] = mean(MLP(pooled_irab[c]) for c in children(i))
    pooled_rel[i] = pooled_irab[i] + parent_msg[i] + children_msg[i]
```

**Why this is informative:** explicitly aggregates the head's feature
into the dependent (e.g., a verb's feature reaches its subject; a
preposition's feature reaches the noun it governs). Tests whether
explicit head-aware aggregation beats attention's implicit
weight-learning.

### 3.3 Clause-level context propagation

For each clause root (DEPREL=root or DEPREL=conj), broadcast the root's
feature to all descendants:

```
clause_root[i] = nearest_root_ancestor(i)        # by dep tree traversal
clause_emb[i]  = MLP(pooled_irab[clause_root[i]])
pooled_rel[i]  = pooled_irab[i] + clause_emb[i]
```

**Why:** Arabic verb-initial sentences carry case assignment from the
verb to all argument nominals; this gives every word in the clause
direct access to the verb's feature. Cheaper than full message
passing.

### 3.4 Syntactic neighborhood aggregation

Pool features from a 1-hop dep neighbourhood (parent + children +
siblings):

```
neighborhood[i] = {parent(i)} ∪ children(i) ∪ siblings(i)
pooled_rel[i] = pooled_irab[i] + mean(MLP(pooled_irab[j]) for j in neighborhood[i])
```

Like §3.2 but adds siblings (other children of the same parent). For
verbs with multiple arguments this propagates argument-to-argument
information through the shared parent.

## 4. 4-cell ablation grid

| Variant | Mechanism | Params | What it tests |
|---|---|---:|---|
| Phase 3.1-A | relation-aware self-attention (§3.1) | ~1M | primary — bias attention by dep edge type |
| Phase 3.1-B | message passing (§3.2) | ~1.5M | does explicit head/children aggregation beat learned attention? |
| Phase 3.1-C | clause-level propagation (§3.3) | ~600K | is clause-root context the dominant relational signal? |
| Phase 3.1-D | syntactic neighborhood (§3.4) | ~600K | does sibling info matter beyond parent + children? |

All layered on top of Phase 3-A (production checkpoint). Morph heads
**frozen** in all variants. Phase 3.1-A is the gate run.

## 5. Schema for dep tree access

The current Phase 3 schema gives per-word `(deprel, head_idx,
governor_upos)`. Phase 3.1 needs the **full dep tree per sentence**:
parent index for every word, children indices for every word, edge
type per edge.

This is computable from the existing `data/morph_v1_dep` corpus —
each record's per-word `head_idx` defines the parent; children are
the inverse. We add a one-pass preprocessing step in the dataset that
builds `parents: List[int]`, `children: List[List[int]]` per sentence
and serialises them as part of the encoded record.

No new Stanza parse needed; the dep info is already there.

## 6. Identity initialisation (per-mechanism)

All four mechanisms must start byte-equivalent to Phase 3-A at step 0:

- **§3.1 attention:** edge_bias init zero; Q/K/V projection init to
  reproduce identity output (output = V[i], V projection identity).
  After learning, attention can deviate from identity.
- **§3.2 message passing:** parent_msg/children_msg MLPs init to zero
  output (final-layer weights zero). pooled_rel = pooled_irab at step 0.
- **§3.3 clause propagation:** clause_emb MLP init to zero output.
- **§3.4 neighborhood:** same.

## 7. Decision gate

Soft two-of-three vs Phase 3-A baseline (56.7 / 41.3 / 44.8 / 20.1):
ship the variant if it beats Phase 3-A on ≥ 2 of {case, role-F1,
fully} with no regression > 1.0 pp.

The hypothesis (which the four-cell pattern motivates): mechanism that
adds *new relational computation* beyond static dep embeddings unlocks
gain. The mechanisms differ in computational expressiveness — §3.1 is
most expressive (attention learns its own edge weighting), §3.4 is
least (fixed neighbourhood with mean pool). If §3.4 ships but §3.1
doesn't, the relational signal is already saturable by simple
aggregation; if §3.1 ships and §3.4 doesn't, learned attention is
doing something simple aggregation can't.

## 8. Frozen morph heads (operational)

To freeze morph heads during Phase 3.1 training:

```python
for f in MORPH_FEATURES:
    if f in model.morph_heads:
        for p in model.morph_heads[f].parameters():
            p.requires_grad_(False)
```

Plus `morph_loss_weights = {f: 0.0 for f in MORPH_FEATURES}` so morph
losses don't contribute to the total loss. The encoder is still
shared, so encoder gradients flow from iʿrāb losses; the morph
representation is no longer optimised, eliminating the Phase 2
joint-dynamics drift.

This is a minor change to `train.py`: a new config flag
`freeze_morph_heads: bool` (default False; True for Phase 3.1
configs).

## 9. Reversibility

Phase 3.1 ships in one new module + additive flags:
- `src/irab_tashkeel/morphology/relational_reasoning.py` — new
- `dep_aware_model.py` — extends with `enable_relational_reasoning:
  Optional[str]` taking values `{"attn", "mp", "clause", "nbhd",
  None}`; default None keeps Phase 3-A byte-identical.
- `train.py` — adds `enable_relational_reasoning` + `freeze_morph_heads`
  config keys.

With both flags off, the graph is byte-identical to Phase 3-A.

## 10. Implementation order

1. Dataset extension to compute parent/children per sentence
2. `relational_reasoning.py` with all four mechanisms; pure-tensor
   unit tests (mirror Phase 3 test pattern)
3. Model integration in `DepAwareStructuredModel`
4. `freeze_morph_heads` flag in `train.py`
5. Configs + sbatches for each of the 4 ablation cells
6. HPC smoke test (50 sentences + frozen morph)
7. Run 1 (Phase 3.1-A, attention) — gate
8. If gate passes, ship as new production; run 3.1-B/C/D as
   diagnostics for the writeup
9. Paper integration with the relational-reasoning row + four-cell
   case study extension

Estimated wallclock: ~3-4 hours of code + ~30 min HPC per cell + eval.
Total: half a day to ship + write up.

## 11. Out of scope for Phase 3.1 itself

- More than one round of message passing (Phase 3.2 candidate)
- Cross-sentence reasoning (out of scope for per-sentence iʿrāb)
- Hand-coded relational rules (would re-introduce hard constraints)
- Re-running Stanza at inference (separate inference-quality lever)
- Joint training of morph heads (Phase 2 documented this is harmful)

## 12. Run 1 + 2 results — Phase 3.1 closes as negative

**Phase 3.1 does NOT ship. Phase 3-A remains production.** Both
training-mode variants fail the soft gate:

| Variant | morph | case | role-F1 | marker | fully | gate |
|---|---|---:|---:|---:|---:|:---:|
| Phase 3-A baseline | trained | **56.7** | **41.3** | **44.8** | **20.1** | — |
| Phase 3.1-A (491284) | frozen | 56.0 (−0.7) | 38.7 (−2.6) | 41.8 (−3.0) | 18.7 (−1.4) | ✗ |
| Phase 3.1-A2 (491290) | unfrozen | 56.0 (−0.7) | 41.0 (−0.3) | 41.0 (−3.8) | 17.2 (−2.9) | ✗ |

The unfrozen variant recovered role-F1 (the frozen-morph
configuration was over-conservative — the encoder was originally
trained with morph supervision flowing and freezing it prevented the
encoder from adapting to support the new attention layer) but
regressed marker + fully more sharply.

### 12.1 Mechanism interpretation

Both variants confirm the same finding: **the encoder representation
Phase 3-A learns from the static dep features already saturates the
dep information at this corpus size and training budget**. Adding a
relation-aware attention layer downstream just redistributes existing
prediction mass without adding new information. This is the same
shape as Phases 5, 6: redistribution of existing signal under joint
training tends to hurt the heads most sensitive to small input
distribution changes (here, marker and the *fully* aggregate, which
require all four heads to be simultaneously correct).

The frozen-morph variant's role-F1 collapse (−2.6 pp) was a separate
artifact: the encoder couldn't adapt because morph gradient was
removed, so the representation drifted in only the iʿrāb-loss
direction, hurting role-discrimination disproportionately. Unfreezing
morph fixed the role-F1 issue but didn't recover marker or fully.

### 12.2 The architectural lesson sharpens further

The original four-cell case study (Phases 4a + 2 + 5 + 6 = same-
supervision rearrangements all plateau or regress; Phase 3 = new info
gain) is now extended to FIVE cells with Phase 3.1 added:

| Phase | Intervention | New info? | Result |
|---|---|:---:|---|
| 4a | 25 → 34 taxonomy | ✗ (same labels, more granular) | plateau |
| 2 | morph→iʿrāb conditioning | ✗ (same supervision rearranged) | regress |
| 5 | role→case output bias | ✗ (re-uses role pred) | slight regress |
| 6 | case+role→marker output bias | ✗ (re-uses case+role pred) | larger regress |
| **3** | **UD dep edges (static input)** | **✓ relational signal (orthogonal)** | **gain** |
| **3.1** | **relation-aware attention** (rich relational reasoning over the same dep tree) | **✗ rearranges Phase 3's existing dep signal** | **regress** |

The five-cell pattern delivers an even sharper conclusion: **at 296M
/ 6 epochs, ANY downstream-of-encoder mechanism that operates on
already-incorporated information plateaus or regresses, regardless
of whether the mechanism is encoder-side conditioning (Phase 2),
input-side static augmentation (Phase 3), output-side hierarchical
decoders (Phases 5, 6), or input-side dynamic attention (Phase 3.1).
Only the introduction of NEW information (Phase 3 dep features vs
the morph + taxonomy baseline) unlocks gain.**

The implication for the next phases: do NOT add more architectural
mechanisms downstream of Phase 3-A's encoder. The next productive
levers must add NEW information sources or NEW training signals:

- **#39 (rare-construction synthetic augmentation)**: adds new
  training data for under-covered constructions
- **Phase 9 (grammar memory expansion)**: adds new lexical
  knowledge via the symbolic constraint reranker
- **Inference-time Stanza parsing**: addresses the Phase 3 inference
  distribution mismatch by giving the predictor real dep features
  on test inputs (currently inference passes zero dep_emb)
- **Cleaner Stanza alignment** (drop the 50% match threshold) +
  **gold UD-PADT dep on the morph half** — both add more dep
  coverage, which is the only signal that has ever moved the needle

### 12.3 Ship decision (final)

Phase 3.1 ships **as a documented negative result**. Code stays in
the codebase under `enable_relational_reasoning: None` default
(byte-equivalent to Phase 3-A). The relation-aware attention module
+ five-cell case study extension are valuable empirical evidence
for the "orthogonal info > rearrangement" generalization at this
scale.

Phase 3-A (rev 2 + Phase 1 morph + static Stanza dep features)
remains the production checkpoint. The next architectural cycle
focuses on data engineering (Phase 9 / #39 / cleaner Stanza), not
more downstream mechanisms.

## 13. After Phase 3.1 closes — next directions (revised)

The five-cell case study makes the priority order clearer:

1. **#39 — rare-construction synthetic augmentation** (highest priority).
   §5.4 of REPORT.md documents that EXCEPTION + KANA_SISTERS construct
   types are 0/9 and 0/7 across ALL systems including the closed
   frontier. This is a clear "missing training data" gap, not a
   missing-architecture gap. Augmentation here is the canonical case
   of "add new info" rather than "rearrange existing".

2. **Cleaner Stanza dep coverage**. Phase 3 succeeded with 70%
   alignment success. Dropping the 50% match threshold to 25% would
   raise coverage to ~85%; adding gold UD-PADT dep on the
   morph-only half adds another ~6,600 sentences with perfect dep.
   Both are pure data-engineering and add information.

3. **Inference-time Stanza parsing**. Phase 3 inference currently
   passes zero dep_emb (the predictor doesn't run Stanza on Gazelle
   inputs). Running Stanza at inference would give the iʿrāb heads
   the same dep-augmented input distribution they were trained on.
   This is a deployment-time fix, not a training-time intervention.

4. **Phase 9 — grammar memory expansion** (lower priority, requires
   Arabic linguistic decisions). Extends the symbolic constraint
   reranker with new lexicons (mawsool relative pronouns, modal
   verbs, demonstratives). Adds symbolic knowledge that the encoder
   doesn't have parametric access to.

5. **Phase 11 — explanation engine** (capstone). Generates rationale
   prose conditioned on the predictions; orthogonal to the
   supervision plateau (it's a new output modality, not new input).

The original Phase 3.1 plan (3.1-B/C/D — message passing,
clause-level, neighborhood) is **archived without execution**: the
five-cell pattern strongly predicts they will plateau too, since
all are rearrangements of the same relational signal Phase 3-A
already captures statically. Running them would only add more
documented negatives without changing the conclusion. Code surface
left in `relational_reasoning.py` for reference; the alternative
mechanisms are not implemented.

Per the user's 2026-05-08 redirect, the priority order after Phase
3.1 ships is:

1. **Rare-construction synthetic augmentation** (#39): generate
   training examples for EXCEPTION + KANA_SISTERS where every system
   currently fails (§5.4 of REPORT.md). Adds new training data.
2. **Richer syntactic supervision**: e.g., dependency role
   prediction as an auxiliary head — different from per-word DEPREL,
   it's a per-edge role label.
3. **Dependency-aware retrieval keys for grammar memory** (Phase 9
   reframed): build a retrieval index over the training corpus keyed
   by dep tree fragments, retrieve at inference for similar
   constructions.

These three are heavy data-engineering efforts but are the directly
productive next steps once Phase 3.1's architecture is stable.
