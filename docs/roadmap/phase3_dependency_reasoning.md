# Phase 3 — Dependency-Aware Reasoning

> Independent signal source: UD dependency edges (DEPREL + HEAD topology)
> contribute relational information that morphology + taxonomy cannot
> capture. Motivated by the Phase 4a substitutability finding and the
> Phase 2 joint-dynamics finding, which together exhausted morph +
> taxonomy reshaping at 296M / 6 epochs.

## 1. Motivation — what Phase 4a + Phase 2 ruled out

Two findings now constrain the next architectural lever:

1. **Phase 4a substitutability** (`docs/roadmap/phase4_taxonomy.md`):
   parallel multi-task supervision on morphology and 25→34 role
   granularity each independently lift Gazelle role-F1 by +5–7 pp,
   but combining them yields only +5.7 pp (vs the +12.2 pp linear
   sum). They capture largely the *same* representational gain at the
   encoder level.

2. **Phase 2 joint-dynamics** (`phase2_soft_morphology_conditioning.md`):
   hierarchical conditioning (FiLM, additive, FiLM detached) on the
   morph head outputs ALSO regresses Phase 1 at 296M / 6 epochs. The
   regression is driven by *joint optimisation dynamics* (morph head
   representation drifts under joint training, iʿrāb heads chase the
   moving target), not by the multiplicative gating mechanism.

Together these say: **rearranging the existing morph + taxonomy
supervision inside the same encoder bottleneck is exhausted at this
scale.** The next productive lever needs to add information that
morphology and taxonomy literally cannot express.

**Dependency edges fit that bill.** UD-PADT annotates each token with a
HEAD index (which token it depends on) and a DEPREL label (the
relation type — `nsubj`, `obj`, `nmod`, `amod`, `case`, `conj`,
etc.). Arabic iʿrāb is fundamentally a *relational* analysis: a noun
is `fail` (subject) because it relates to a verb in a specific way; a
noun is `mudaaf_ilayh` because it's the second member of an *iḍāfa*
construction; a noun is `ism_majrur` because it's governed by a
preceding preposition. Morphology features (Case=Acc, Gender=Fem) and
the role taxonomy (25 or 34 labels) cannot express *which other token*
is the governor — but the DEPREL + HEAD pair does.

The Phase 3 hypothesis: **adding UD dep features (DEPREL one-hot +
relative HEAD direction/distance) as input features to the iʿrāb
decoders unlocks role discrimination that morphology and taxonomy
together cannot.**

## 2. Architecture

```
                input_ids                         (B, T)
                    │
                    ▼
           AraT5v2-base encoder                   (B, T, 768)
                    │
                    ▼
       first-subtoken word pool                   (B, W, 768)   = h
                    │
        ┌───────────┼───────────────────────────────────┐
        │                                                │
        ▼                                                ▼
  morph heads (7)         dep features (per word)
  ├── gender   (3)        ┌── DEPREL embedding   (D_dep dims)   ─┐
  ├── number   (4)        │   (one of 37 UD labels)              │
  ├── definite (4)        ├── HEAD direction     (3: left/right/root)
  ├── person   (4)        ├── HEAD distance log-bucket (5 buckets)
  ├── aspect   (3)        └── governor's POS      (6 dims)       │
  ├── mood     (5)                                                │
  └── voice    (3)                                                │
                                                                  │
                    h ⊕ dep_features  (concat, B, W, 768+M_dep)  ◄┘
                                │
                                ▼
                    small projection (B, W, 768)
                                │
                                ▼
                    case (5)   role (25)   marker (18)   pos (6)
```

Dep features enter as **input augmentation**, not as a conditioning
module. The iʿrāb decoders see `[h ; dep_features]` projected back to
768 dim before any case/role/marker head consumes it. This is closer
to the additive-bias mechanism from Phase 2 (which preserved Phase 1
on case + marker + fully) than to FiLM. **Dep features are static
inputs computed offline by Stanza/UD parser**, NOT learned heads with
joint gradient flow — so the Phase 2 joint-training-dynamics issue
does not apply.

The dep-feature embeddings (DEPREL, HEAD direction, HEAD distance,
governor POS) are the only new learnable parameters introduced. Total
extra parameters: ≈ `37 × D_dep + 3 × 16 + 5 × 16 + 6 × 16 + 768 ×
(768+M_dep)` ≈ 600K, negligible vs 296M encoder.

## 3. Per-word dep features (frozen schema)

For each word in the sentence:

| Feature | Source | Vocab | Notes |
|---|---|---|---|
| `deprel` | UD-PADT col 8 / Stanza dep | 37 | one of `nsubj`, `obj`, `nmod`, `amod`, `case`, `conj`, `cc`, `det`, `mark`, `aux`, `cop`, `xcomp`, `acl`, `advcl`, `advmod`, `appos`, `compound`, `flat`, `fixed`, `obl`, `iobj`, `expl`, `vocative`, `discourse`, `dislocated`, `csubj`, `ccomp`, `dep`, `parataxis`, `goeswith`, `list`, `orphan`, `punct`, `root`, `nummod`, `clf`, `<unk>` |
| `head_direction` | sign of (HEAD − self) | 3 | `left` (governor precedes), `right` (governor follows), `root` (no governor) |
| `head_distance_bucket` | log-bucket of \|HEAD − self\| | 5 | `0=root`, `1=adj`, `2-3=near`, `4-7=mid`, `≥8=far` |
| `governor_upos` | UPOS of word at HEAD index | 6 | canonical POS (`noun`, `verb`, `particle`, `pronoun`, `adjective`, `other`) |

`<unk>` for missing/parser-failure cases. Embeddings are `D_dep=32`
for DEPREL and `16` for the others. M_dep = 32 + 16 + 16 + 16 = 80.

## 4. Data pipeline

UD-PADT comes with dep annotations. The distill_v2 corpus does NOT.
Two routes:

### 4a. Phase 3a (UD-PADT-only training)

Train and evaluate on UD-PADT alone. Existing `morph_v1` corpus
already merges UD-PADT (morph-supervised) with distill_v2 (iʿrāb-
supervised). Phase 3a uses *only the UD-PADT half*: morph + dep info
present, iʿrāb labels missing.

**Catch:** UD-PADT does not have iʿrāb supervision. We can't directly
train iʿrāb heads on UD-PADT data. Phase 3a is therefore *not viable
on its own* — it would produce a morph + dep classifier with iʿrāb
heads frozen at random init.

### 4b. Phase 3b (distill_v2 + Stanza-parsed deps) [primary]

Run Stanza's UD parser over the entire distill_v2 corpus offline.
Cache the (DEPREL, HEAD index, HEAD POS) per word in
`data/structured_v1_dep/`. This adds a `dep_features` field to each
training example. The pipeline:

```
data/structured_v1/{train,val}.jsonl                (existing iʿrāb-supervised)
                    │
                    ▼  scripts/morphology/parse_deps.py
                       (Stanza UD parser, ~30 min for 7K sentences)
                    │
                    ▼
data/structured_v1_dep/{train,val}.jsonl            (+ dep_features field)
                    │
                    ▼  merge_corpora.py + DepAwareDataset
                    │
                    ▼
DepAwareStructuredModel                             (Phase 3b model)
```

UD-PADT half retains its native dep features. Distill_v2 half gets
Stanza-predicted deps (Stanza's reported UAS on Arabic UD ≈ 84%, so
the dep features will be noisy, but the *signal* is there).

### 4c. Sanity probe — predicted vs gold dep on UD-PADT

Run Stanza on UD-PADT test split. Compare predicted DEPREL vs gold
DEPREL: target ≥ 80% accuracy. If significantly lower, the parsed
distill_v2 features will be too noisy and we'd need to fall back to a
held-out human-parsed Arabic dep corpus.

## 5. Ablation matrix

3-cell, on top of Phase 1 baseline (rev 2 + 7 morph heads):

| Variant | morph heads | dep features | what it tests |
|---|:---:|:---:|---|
| Phase 3-A — Phase 1 + dep features | ✓ | ✓ | full hypothesis: morph + dep complementary |
| Phase 3-B — dep features alone | ✗ | ✓ | does dep alone beat Phase 1, or only with morph? |
| Phase 3-C — Phase 1 only (control) | ✓ | ✗ | re-runs Phase 1 baseline (sanity check) |

This is a *3-cell* matrix, not 2×2. Phase 3-A is the primary; B and C
are controls. We do not run a `no morph + no dep` cell — that's just
rev 2.

## 6. Decision gate

Same strict gate as Phase 2 (case ≥ 53.0, role-F1 ≥ 43.0, fully ≥ 19.4
on Gazelle, vs Phase 1 baseline 53.7 / 42.3 / 41.0 / 19.4).

Phase 3-A ships as the new production checkpoint *only if* it beats
Phase 1 on **at least two of {case, role-F1, fully}** while not
regressing any of them by more than 1.0 pp.

If Phase 3-A passes role-F1 (≥ 43.0) but flat-or-regress on fully,
it ships as opt-in only. If it regresses both, we drop dep features
and the substitutability + joint-dynamics findings stand as a
ceiling claim for the 296M / 6-epoch operating point.

## 7. Reversibility + production hygiene

Phase 3 changes ship in two new files (`dep_features.py`,
`dep_aware_model.py`) and *additive* changes to `train.py` (new
`enable_dep_features` config key with default `False`). With
`enable_dep_features=False`, the model graph is byte-identical to
Phase 1. Reverting is `enable_dep_features: false` in the config.

The Stanza-parsed corpus is committed as a separate dataset
(`data/structured_v1_dep/`) so Phase 1 / Phase 4a / Phase 2 corpora
are untouched. Existing checkpoints continue to load.

## 8. Smoke test (50-sentence)

Before any 6-epoch retrain we run a 50-sentence + 5-UD-PADT-sentence
2-epoch smoke on a single GPU node to verify:

- Stanza-parsed `dep_features` field round-trips through `DepAwareDataset`;
- `DepAwareStructuredModel` produces case/role/marker logits on a
  forward pass with `dep_features=None` AND with `dep_features` set
  (no shape errors);
- Per-feature embedding gradients flow back from `L_irab`;
- At-init behaviour: with dep-feature embeddings set to zero and the
  `(h+dep)` projection initialised as `[I; 0]` (concat → 768 = first
  768 dims of input → preserved), iʿrāb logits at step 0 should be
  byte-identical to Phase 1.

## 9. HPC schedule

3 retrains × ≈ 2.5 h on the stud partition's MIG 4g.40gb slice
(Phase 1 6-epoch retrain takes 8 min; dep features add minor
forward overhead). Total ≈ 30 min sequential plus ~30 min Stanza
parsing of distill_v2.

Order:
1. Pre-process: Stanza-parse distill_v2 → `data/structured_v1_dep/` (~30 min)
2. Smoke test (15 min)
3. Run 1: Phase 3-A — Phase 1 + dep features (gate run)
4. Run 2: Phase 3-B — dep features alone (no morph)
5. Run 3: Phase 3-C — Phase 1 only (sanity control; matches existing Phase 1 result if seeds aligned)

Run 1 is the gate. If it regresses Phase 1, Runs 2 + 3 still inform
the writeup.

## 10. Evaluation

Same 4-stream metric breakdown as Phase 2 (`scripts/slurm/70b_eval_phase2_light.sbatch`),
adapted to Phase 3. Lightweight eval (Gazelle gate + 4-stream stress +
UD-PADT morph + UD-PADT dep accuracy) for all 3 runs. Full eval (with
MASAQ + constraints) only on the ship candidate.

New auxiliary metric: **dep-prediction accuracy on UD-PADT test**, to
verify dep features are not damaged by the joint training.

## 11. Risk register

- **Stanza UD parser noise.** Stanza's Arabic UAS ≈ 84%, so 16% of
  per-word HEAD indices and DEPREL labels in distill_v2 will be wrong.
  This is the dominant noise source. If Phase 3-A regresses, the
  diagnostic check is whether using **gold UD-PADT dep features only**
  on a UD-only subcorpus shows a stronger signal.
- **Substitutability with morphology.** It's possible dep features
  are themselves substitutable with morph features at this encoder
  scale (DEPREL `nsubj` and Case=Nom are correlated). Phase 3-B
  (dep alone, no morph) addresses this directly.
- **MWT / tokenisation alignment.** Stanza tokenises differently from
  the AraT5v2 SentencePiece. Need to align Stanza tokens to the
  word-level units the iʿrāb heads operate on. The existing UD-PADT
  pipeline already handles MWT collapsing (last-segment fallback);
  Stanza output uses the same UD CoNLL-U format so the same loader
  applies.
- **Dep features could leak gold info on UD-PADT.** If we evaluate on
  UD-PADT and the dep features are gold dep, we're double-counting.
  The Gazelle eval is unaffected (Gazelle has no dep gold), but any
  UD-PADT eval that uses gold deps as input must be reported with
  that caveat.

## 12. Out of scope for Phase 3

- Graph attention layer over dep edges (Phase 3.5 — defer behind
  Phase 3-A result).
- Joint dep+iʿrāb training where dep is also a learned head (would
  recreate the Phase 2 joint-dynamics issue).
- Dep features for the POS head (POS is independent of dep relations
  in the canonical 6-class POS schema).
- v4 taxonomy + dep features composition test (Phase 3 stays on v3
  unless 3-A passes the gate).

## 13. Open questions to revisit after Run 1

- Is the Stanza-parsed dep noise the bottleneck? (Diagnostic:
  Phase 3-A on UD-PADT-only val using gold dep should be cleaner than
  Phase 3-A on distill_v2 val using Stanza dep.)
- Does dep + morph compose past the substitutability wall?
  (Diagnostic: Phase 3-A combined Δ vs Phase 1 + Phase 3-B
  individually-summed Δ — same form as the Phase 4a substitutability
  table. If linear sum holds, dep is genuinely orthogonal info.)
- Does dep feature interact with constraint reranking? (Test with
  constraints on/off for the production candidate.)
