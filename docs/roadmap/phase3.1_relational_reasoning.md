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

## 12. After Phase 3.1 ships — next directions

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
