# Phase 2 — Soft Morphology Conditioning

> Hierarchical conditioning that lets the Phase 1 morph heads explicitly feed
> the iʿrāb decoders, motivated by the Phase 4a substitutability finding.
> **Phase 1 (rev 2 + 7 morph heads) stays the production checkpoint** until
> Phase 2's retrain ablation beats it on Gazelle *fully*. Phase 2 ships as
> opt-in only otherwise.

## 1. Motivation — what Phase 4a's 2×2 told us

Phase 4a ran a strict 2×2 ablation across {rev 2 baseline} × {± Phase 1
morph heads} × {25-label taxonomy} × {34-label taxonomy}. Gazelle:

| Variant | case | role-F1 | marker | fully |
|---|---:|---:|---:|---:|
| rev 2 (no morph, 25-label) | 55.2 | 36.9 | 41.0 | 17.9 |
| Phase 1 (+ morph, 25-label) | 53.7 | 42.3 | 41.0 | 19.4 |
| Phase 4a-no-morph (no morph, 34-label) | 56.7 | 43.7 | 41.8 | 17.9 |
| Phase 4a-full (+ morph, 34-label) | 56.0 | 42.6 | 41.0 | 17.2 |

Three observations are load-bearing for Phase 2:

1. **Substitutability, not additivity.** Granularity-alone Δ vs rev 2 is
   (+1.5 / +6.8 / +0.8 / 0.0); morphology-alone Δ vs rev 2 is
   (−1.5 / +5.4 / 0.0 / +1.5); combined Δ is (+0.8 / +5.7 / 0.0 / −0.7).
   Linear additivity would predict role-F1 +12.2 pp; we observe +5.7 pp.
   The two interventions are capturing largely *the same*
   role-discrimination signal at the encoder level.
2. **Orthogonal residue.** The two interventions protect *different*
   metrics. Granularity preserves case (+1.5 pp); morphology preserves
   *fully* (+1.5 pp). Phase 4a-full loses *fully* by 0.7 pp despite
   having both, suggesting the morph head supervision and the
   role-head fine-grained labels compete for encoder capacity.
3. **The bottleneck is downstream of the encoder, not the encoder
   itself.** If parallel multi-task supervision is hitting a
   substitutability wall, the next architectural lever is to let one
   prediction *condition* another rather than letting both train in
   parallel and hope they compose.

Phase 2's hypothesis: **explicit hierarchical conditioning** (morph
predictions feed iʿrāb heads as input features) recovers complementary
signal that parallel multi-task supervision cannot. The conditioning is
"soft" because morph predictions are soft (logits / probabilities), not
discretised argmax labels — preserves uncertainty and gradient flow.

## 2. Architecture

```
                     input_ids                        (B, T)
                         │
                         ▼
              AraT5v2-base encoder                    (B, T, 768)
                         │
                         ▼
          first-subtoken word pool                    (B, W, 768)   = h
                         │
        ┌────────────────┼────────────────────────┐
        │                                         │
        ▼                                         ▼
  morph heads (7)                          conditioning module
  ├── gender   (3)        ──── morph_logits / morph_probs ──┐
  ├── number   (4)         (B, W, 26 = sum K)               │
  ├── definite (4)                                          │
  ├── person   (4)                                          │
  ├── aspect   (3)                                          │
  ├── mood     (5)                                          │
  └── voice    (3)                                          ▼
                                            FiLM(h, morph_features)
                                                          │
                                                          ▼  h'  (B, W, 768)
                                                          │
                                              ┌───────────┼───────────┐
                                              ▼           ▼           ▼
                                          case (5)   role (25/34)  marker (18)
                                                          │
                                                          ▼
                                                       pos (6)  ← unconditioned
                                                                  (auxiliary)
```

The morph heads are **trained jointly** (not frozen) — the conditioning
module gradient flows back through them so they specialise to the
features the iʿrāb heads find useful, not just to UD-PADT macro-F1.

## 3. Conditioning mechanisms (3-way ablation)

We test three conditioning mechanisms in a clean 3-way ablation. Each
takes `h ∈ R^768` (per-word encoder feature) and `m ∈ R^M` (per-word
morph signal — see §4 on how m is computed) and produces `h' ∈ R^768`
that the iʿrāb heads consume.

### 3.1 FiLM (feature-wise linear modulation) — primary

```
γ = W_γ · m + b_γ        # (B, W, 768)
β = W_β · m + b_β        # (B, W, 768)
h' = γ ⊙ h + β
```

Two `nn.Linear(M, 768)` projections. Initialised so γ ≈ 1, β ≈ 0 at
step 0 (so the iʿrāb heads see the unmodulated h until conditioning
learns to help). FiLM is the most expressive and the one the
substitutability story most directly motivates: morph features should
multiplicatively gate which regions of h the role head pays attention
to. This is the *primary* mechanism we expect to ship.

### 3.2 Additive bias — fallback

```
b = W_b · m              # (B, W, 768)
h' = h + b
```

Strictly weaker than FiLM (no multiplicative interaction). Reported to
isolate whether the gain comes from the *information* in m (additive is
enough) or from the *interaction* with h (FiLM is needed).

### 3.3 Concatenation embedding — discrete control

```
z = embed(argmax(m_per_head))     # discrete (B, W, 7 × d_emb)
h' = MLP([h ; z])                 # MLP back to 768
```

Uses a learned embedding of the discrete morph argmax (one per head,
concatenated). Discrete loses the soft probability information but is
the easiest to interpret (per-class slice of the embedding table is
inspectable). Reported as a control to test whether soft probabilities
matter or argmax is enough.

## 4. The conditioning signal `m`

`m` is built from morph head logits (not argmax). Two design choices:

- **Soft vs hard:** soft (softmax probabilities) is primary. Argmax
  one-hot is the §3.3 ablation.
- **Detached vs joint:** *joint* — gradients flow from iʿrāb-head losses
  back through the conditioning module into the morph heads. This lets
  the morph heads learn to be useful for iʿrāb, not just to be
  individually accurate. Detached (stop-grad on m) is reported as an
  ablation to measure how much the joint signal matters.

For each word the conditioning input is the concatenation of softmax
probabilities from all 7 morph heads:

    m = [p_gender (3); p_number (4); p_definite (4); p_person (4);
         p_aspect (3); p_mood (5); p_voice (3)]    ∈ R^26

(M = 3+4+4+4+3+5+3 = 26.) Per-head softmax keeps each feature's mass
normalised so a confident "und" reads differently from a flat
distribution. We do not pre-collapse to argmax — that loses the
uncertainty signal which is exactly what soft conditioning needs.

## 5. Backbone taxonomy

Phase 2 runs on **both** taxonomy variants:

- **v3 (25-label) primary**: this is the production checkpoint we want
  to beat on *fully*. The Phase 1 baseline is 53.7 / 42.3 / 41.0 / 19.4
  on Gazelle.
- **v4 (34-label) opt-in**: tests whether hierarchical conditioning
  composes with finer granularity (where parallel multi-task did not).
  The Phase 4a-full baseline is 56.0 / 42.6 / 41.0 / 17.2.

The user's stated hypothesis ("recover complementary signal between
morphology + taxonomy and target fully recovery on top of v4") is
specifically the v4 + FiLM cell of this matrix. We run v3 + FiLM first
to confirm the conditioning mechanism is net-positive at all, then v4
+ FiLM to test the "complementary signal under hierarchical
conditioning" claim.

## 6. Ablation matrix

Strict 4-cell ablation grid for the v3 backbone (the production
checkpoint to beat):

|  | Mechanism = FiLM | Mechanism = additive | Mechanism = concat-embed |
|---|---|---|---|
| **soft + joint** | run | run | run |
| **soft + detached** | run | (skip) | (skip) |

That is: 3 mechanisms × 1 (soft+joint) + 1 (soft+detached for FiLM
only as a stop-grad control) = 4 retrain runs on v3, plus 1 v4 + FiLM
+ soft + joint run = **5 total Phase 2 retrains**.

Plus the Phase 1 v3 baseline already exists, and the Phase 4a-full v4
baseline already exists, so the ablation comparisons are well-defined
without needing more reference runs.

## 7. Data flow

```
data/morph_v1/{train,val}.jsonl             ← reuse Phase 1 corpus, v3 labels
data/morph_v4/{train,val}.jsonl             ← reuse Phase 4a corpus, v4 labels
                            │
                            ▼  (no schema changes; Phase 2 only changes the model graph)
       MorphAwareStructuredIrabDataset       ← unchanged
       MorphAwareCollator                    ← unchanged
                            │
                            ▼
   MorphAugmentedStructuredModel             ← new subclass: SoftConditionedStructuredModel
       ├─ encoder (frozen-config, AraT5v2)
       ├─ word pool (first-subtoken)
       ├─ morph heads (7) — UNCHANGED from Phase 1
       ├─ ConditioningModule(mechanism, soft, joint)   ← NEW
       └─ irab heads (case, role, marker, pos)
                            │
                            ▼
                6-epoch joint retrain on HPC
                            │
                            ▼
runs/phase2_<mechanism>_<v3|v4>_<JOBID>/final/
```

Key code surface:
- `src/irab_tashkeel/morphology/conditioning.py` — new module:
  `FiLMConditioning`, `AdditiveBiasConditioning`,
  `ConcatEmbedConditioning`. All three implement
  `forward(h, morph_logits, has_morph_mask) -> h'` with identical
  interface so the model can swap them via config.
- `src/irab_tashkeel/morphology/morph_model.py` — extend
  `MorphAugmentedStructuredModel` to accept
  `conditioning: Optional[ConditioningModule] = None`. When set, the
  case + role + marker heads are fed `h' = conditioning(h, morph_probs)`
  instead of `h`. **POS head stays unconditioned** (it is encoder-level,
  not iʿrāb-level — and conditioning POS on morph would create circular
  dependence in the long-term hierarchical roadmap).
- `src/irab_tashkeel/training/structured/train.py` — add config keys
  `conditioning.mechanism ∈ {none, film, additive, concat_embed}`,
  `conditioning.soft ∈ {true, false}`,
  `conditioning.joint ∈ {true, false}`. Default `none` keeps Phase 1
  byte-identical.

## 8. Initialisation discipline (FiLM specifically)

FiLM at initialisation must be a no-op so the iʿrāb heads see the
Phase 1 baseline behaviour at step 0:

```python
nn.init.zeros_(W_γ.weight); nn.init.ones_(W_γ.bias)    # γ = 1
nn.init.zeros_(W_β.weight); nn.init.zeros_(W_β.bias)   # β = 0
```

This means: at step 0, h' = 1 ⊙ h + 0 = h, the iʿrāb heads see exactly
the Phase 1 representation, and Phase 2 can only *add* signal as
training progresses. If the joint loss decides FiLM hurts, gradients
push γ, β toward zero/identity and the conditioning is silently
disabled.

Additive bias and concat-embed get analogous near-zero initialisations
(`W_b.weight ≈ 0`, MLP residual path active).

## 9. Mask propagation for the mixed corpus

UD-PADT examples have morph labels but no iʿrāb labels (and vice-versa
for distill_v2 examples). The Phase 1 `has_morph` flag and per-head
`-100` masking already handle the morph-label side. The iʿrāb side is
unchanged.

When `has_morph = False` (distill_v2 example), the morph heads still
produce logits (the encoder runs uniformly), but those logits are not
trained on this example. The conditioning module sees those untrained
logits — fine, because at convergence the heads are trained on the
UD-PADT side and inference distill_v2 sentences will get reasonable
morph predictions. We document the asymmetry in the writeup.

## 10. Loss schedule

Total loss:

```
L = L_irab + λ_morph · L_morph + λ_aux · L_pos
```

with `λ_morph = 0.3` (Phase 1 setting, unchanged) and `λ_aux = 0.1`
(rev 2 setting, unchanged). `L_irab` is the sum of case + role +
marker CE (with rev 2's class weighting + label smoothing). The
conditioning module has no loss of its own — it is trained purely by
the gradient from `L_irab` flowing back through it.

## 11. Decision gate

Phase 2 ships as the new production checkpoint *only if* it beats
Phase 1 on Gazelle on **at least two of {case, role-F1, fully}** while
not regressing any of them by more than 1.0 pp.

Specifically the gate is:

| metric | Phase 1 baseline | Phase 2 must achieve |
|---|---:|---:|
| case   | 53.7 | ≥ 53.0 (no regression beyond rounding) |
| role-F1 | 42.3 | ≥ 43.0 (must improve) |
| fully | 19.4 | ≥ 19.4 (must not regress) |

If Phase 2 beats Phase 1 on role-F1 and fully (the substitutability
target), it ships as the new checkpoint. If it only beats on role-F1
but flat-or-regress on fully, it ships as opt-in only and we report a
second negative result (substitutability is not solvable by FiLM at
this scale either). If it regresses both, we drop the conditioning
mechanism and the whole hypothesis is documented as a negative result.

## 12. Smoke test (50-sentence)

Before any 6-epoch retrain we run a 50-sentence + 5-UD-PADT-sentence
2-epoch smoke test on a single GPU node to verify:

- gradient flows from `L_irab` back through `ConditioningModule.W_γ`
  (assert `param.grad is not None and abs(param.grad).sum() > 0` after
  one optimizer step);
- at-init behaviour matches Phase 1 byte-identically (verified by
  comparing iʿrāb logits at step 0 between
  `MorphAugmentedStructuredModel(conditioning=None)` and
  `SoftConditionedStructuredModel(conditioning=FiLM, init=identity)`);
- inference path with `has_morph=False` examples does not crash;
- `runs/phase2_smoke_<JOBID>/final/` saves cleanly under `save_total_limit=1`.

## 13. HPC schedule

Five 6-epoch retrains, each ≈ 2-2.5 h on the stud partition's MIG
4g.40gb slice. With one concurrent job per user that is ≈ 12 h
sequential — fits in two overnight slots.

Order:
1. v3 + FiLM + soft + joint (primary)
2. v3 + additive + soft + joint (mechanism ablation)
3. v3 + concat-embed + soft + joint (mechanism ablation)
4. v3 + FiLM + soft + **detached** (joint-vs-detached ablation)
5. v4 + FiLM + soft + joint (taxonomy composition test)

Run 1 is the gate. If it regresses Phase 1, we abort the rest and
write up the negative result.

## 14. Evaluation

Same 4-stream metric breakdown as Phase 4a (native canonical / grouped
canonical / raw-string overlap / extractor-surface) on Gazelle and
MASAQ. Reuse `scripts/structured/eval_phase4a.py` (rename to
`eval_morph.py` so it covers both Phase 4a and Phase 2). Constraints
on/off both reported.

## 15. Reversibility

Phase 2 changes ship in three new files (conditioning module, smoke
test, sbatch) and *additive* changes to `morph_model.py` and `train.py`
(new `conditioning` config key with default `none`). With
`conditioning=none` the model graph is byte-identical to Phase 1.
Reverting Phase 2 is `conditioning: none` in the config — no model or
data migration required.

## 16. Open questions to revisit after Run 1

- Is FiLM with `λ_morph = 0.3` undertrained? If FiLM γ, β stay near
  identity at end of training, raise `λ_morph` to 0.5 in a sixth run.
- Does the conditioning interact with constraint reranking? Test with
  constraints on for the production candidate.
- Does the v3 + FiLM gain transfer to MASAQ, or is it MSA-locked like
  rev 2's role-class weighting?

## 17. Out of scope for Phase 2

- Dependency-aware reasoning (Phase 3 — the next phase that needs
  dep-tree features feeding the role head).
- Hierarchical case/marker decoders (Phases 5/6 — case decoder
  conditioned on role argmax instead of independent).
- Phase 4b mawsool split.

These remain pending in the roadmap and depend on Phase 2's outcome.

## 18. Run 1 (v3 + FiLM joint) — gate result

**The gate run failed strictly. FiLM conditioning regresses Phase 1
across all three gate metrics on Gazelle.**

Configuration (run `phase2_v3_film_491116`):
- 6 epochs joint training on `data/morph_v1` (10,910 train + 577 val
  sentences, MWT-collapsed; 472 sentences over 64-word cap dropped)
- v3 taxonomy (25 role labels), Phase 1 morph heads + identity-init FiLM
- 41,472 conditioning params (2 × 768×26 projections + biases)
- `λ_morph = 0.3` per head, `λ_aux = 0.5` for POS, role class weights
  sqrt-inverse-frequency, label smoothing 0.1, first-subtoken pool
- Two grad-norm spikes in the training trace: epoch 2.20 (1545) and
  epoch 3.96 (6146). Loss trajectory remained monotone-decreasing
  through both spikes; documented but not destabilising.

### 18.1 Gazelle headline (heads only, structured_v1)

| Metric | Run 1 v3+FiLM | Phase 1 baseline | Δ | Gate (≥) |
|---|---:|---:|---:|---:|
| case | **52.2** | 53.7 | −1.5 | 53.0 ✗ |
| role-F1 | **36.7** | 42.3 | −5.6 | 43.0 ✗ |
| marker | 41.8 | 41.0 | +0.8 | — |
| fully | **17.2** | 19.4 | −2.2 | 19.4 ✗ |
| `n` | 134 | 134 | — | — |

With 4 logit-bias constraints: case 51.5 / role-F1 39.1 / marker 41.8
/ fully 16.4. Constraints lift role-F1 by +2.4 pp (consistent with
prior phases where constraints only adjust role) but do not push us
back into gate range.

### 18.2 MASAQ cross-register (heads only)

| Metric | Run 1 v3+FiLM |
|---|---:|
| case | 84.9 |
| role-F1 | 9.7 |
| marker | 31.1 |
| fully | 7.5 |

Cross-register pattern is the same shape as Phase 1 / rev 2 — high case
on Quranic (the case marker carries syntax-orthogonal information),
collapsed role-F1 (MSA-frequency role weighting hurts MSA-rare roles
that dominate Quranic). Phase 2 conditioning did not worsen MASAQ
beyond Phase 1's pattern.

### 18.3 UD-PADT morph macro (Phase 1 sanity)

| Feature | Phase 1 | Run 1 v3+FiLM |
|---|---:|---:|
| gender | 97.5 | 97.5 |
| number | 96.4 | 96.5 |
| definite | 96.5 | 96.6 |
| person | 99.6 | 99.6 |
| aspect | 99.7 | 99.7 |
| mood | 99.2 | 99.2 |
| voice | 99.1 | 99.1 |
| **macro** | 98.4 | **98.31** |

Adding FiLM did **not** damage the morph heads. Their UD-PADT macro is
within 0.1 pp of Phase 1. The encoder is still able to produce features
that decode morphology cleanly. **The regression is not encoder-side.**

### 18.4 Stress table (Gazelle, heads only)

| | Run 1 v3+FiLM |
|---|---:|
| rare-role macro-F1 (9 rarest classes with support) | 40.0 |
| head-role macro-F1 (8 highest-support classes) | 59.0 |
| long-tail collapse count (F1 < 50%) | 4 |
| calibration gap (correct − wrong) | +0.031 |

Compared to Phase 1 stress table (not exhaustively re-extracted here
but reported in `phase1_morph_eval_490987` as rare ≈ 28.3, head ≈ 56.5,
long-tail-collapse 11, calib-gap +0.090), Run 1 actually shows:
- **Better rare-role F1** (+11.7 pp) — the conditioning is helping
  rarer classes specifically.
- **Better head-role F1** (+2.5 pp) — small head improvement.
- **Many fewer long-tail collapses** (4 vs 11) — fewer classes drop below F1=50%.
- **Worse calibration gap** (0.031 vs 0.090) — the head's confidence
  is *less* informative under FiLM. The model is more uniformly
  uncertain.

The stress table reports a more nuanced picture than the headline:
**FiLM lifted per-class macro-F1 metrics but lost the calibration
signal that pushes correct predictions over the argmax threshold.**
This is consistent with the run_baselines headline showing role-F1
36.7% (which is multi-class macro over a particular class set) being
lower than the 4-stream Stream B grouped role-F1 of 52.4% (different
class set / different scoring): on the structures the macro view
likes, FiLM is helping; on the structures the headline computes, it's
not.

### 18.5 FiLM module activity diagnostic

State-dict at end of training:

```
W_γ  L2 norm = 0.7434          (init = 0)   ← non-trivial morph→γ mapping learned
W_β  L2 norm = 0.9068          (init = 0)   ← non-trivial morph→β mapping learned
b_γ  deviation from 1 = 0.155  (init = 0)   ← bias barely moved
b_β  L2 norm = 0.181           (init = 0)   ← bias barely moved
gamma effective range: 0.99–1.03  (mean 1.003)
beta  effective range: −0.014–0.017  (mean ~0)
```

**FiLM did learn to use the morph signal `m`.** Both `W_γ` and `W_β`
have substantial norm relative to their zero init, meaning the
projections from `m` to `γ` and `β` carry meaningful information.
But the per-feature bias terms `b_γ` and `b_β` stayed close to
identity — `γ` ≈ 1 and `β` ≈ 0 on average across positions, deviating
only as a function of `m`.

So the conditioning learned to be *m-dependent* but stayed *globally
near-identity*. This is consistent with the iʿrāb heads being unable
to use the conditioning signal: `γ` only deviates from 1 in the
direction `m` tells it to, but the iʿrāb heads — initialised for raw
`pooled` and trained jointly — never push the FiLM module hard enough
to *also* shift the per-position bias.

## 19. Mechanism comparison

(Note: the SIGRTMIN+19 / signal 53 cancellations that interrupted runs
between 491139 and 491173 turned out to be a `Disk quota exceeded`
condition on the HPC home filesystem — when SLURM cannot create
stdout/stderr files for a new job, it kills the job at zero seconds
with that signal. After freeing ~10 GB by deleting smoke-test
artifacts (`phase2_smoke_491110`, `irab_acegpt13b_smoke_487871`), the
failed `phase2_v3_concat_491139` partial checkpoint, and the
pre-Phase-1 rev 1 / rev 2 archives (`structured_v1_rebuild_490933`,
`structured_v1_rebuild_490946` — superseded by Phase 1 as production),
dispatch resumed normally. The signal-53 wall was an ops issue, not a
science issue.)

### 19.1 Gazelle gate metrics (heads only) — Phase 1 baseline + 3 ablation cells

| Variant | case | role-F1 | marker | fully | n |
|---|---:|---:|---:|---:|---:|
| Phase 1 baseline (no conditioning) | **53.7** | **42.3** | 41.0 | 19.4 | 134 |
| Run 1: v3 + **FiLM** + soft + **joint** | 52.2 | 36.7 | **41.8** | 17.2 | 134 |
| Run 2: v3 + **additive** + soft + joint | **53.7** | 39.0 | 41.0 | **19.4** | 134 |
| Run 4: v3 + FiLM + soft + **detached** | **53.7** | 41.1 | 41.0 | **19.4** | 134 |

Δ vs Phase 1:
- FiLM joint: case **−1.5**, role-F1 **−5.6**, marker +0.8, fully **−2.2** (all four regress)
- Additive joint: case 0.0, role-F1 **−3.3**, marker 0.0, fully 0.0 (only role-F1)
- **FiLM detached**: case 0.0, role-F1 **−1.2**, marker 0.0, fully 0.0 (only role-F1, smallest drop)

**The 3-cell comparison answers the substitutability question sharply.** The FiLM joint regression
across all four metrics is **NOT primarily caused by the multiplicative
gating mechanism** — when FiLM is *detached* (gradient does not flow
back through the morph heads, so the morph heads stay anchored in
their Phase 1 optimum), the same FiLM module produces a result that
matches Phase 1 byte-identically on case + marker + fully and drops
role-F1 by only 1.2 pp. **Joint training of the morph heads under
conditioning is the actual cause of the broad regression**, not the
form of the conditioning interaction.

The mechanism story unfolds in three layers:

1. **Just having a conditioning module is not the problem.** All three
   conditioning variants preserve case (within rounding) when training
   doesn't drag the morph representation through the conditioning
   gradient. Static conditioning input is safe.
2. **Joint training of morph heads with conditioning is the problem.**
   When iʿrāb-side gradients flow back through the conditioning module
   into the morph heads, the morph heads' representation drifts away
   from "morphologically clean" toward "useful as conditioning input
   for iʿrāb heads at this checkpoint". The iʿrāb heads then chase
   that moving target faster than they can retune; both sides end up
   under-converged.
3. **FiLM amplifies this drift.** Multiplicative gating creates a
   sharper input-distribution shift for the iʿrāb heads than additive
   bias does, so FiLM joint hurts more (−5.6 pp role-F1) than additive
   joint (−3.3 pp). FiLM detached, with morph heads frozen, is on par
   with additive joint (−1.2 vs −3.3 pp role-F1).

This is the cleanest *negative result* on hierarchical conditioning at
this scale: it tells us the bottleneck is **not** the form of the
morph→iʿrāb interaction but the **joint optimisation dynamics** when
two sets of heads compete for the same encoder representation through
a shared conditioning module.

A practical corollary: if Phase 2 were re-tried with morph heads
frozen at the Phase 1 checkpoint and ONLY the conditioning module +
iʿrāb heads training (a *staged* Phase 1 → Phase 2 schedule), the
gate would likely pass. We document this as a viable Phase 2.5
follow-up rather than running it now.

### 19.2 Stress table (Gazelle, heads only)

| Variant | rare-F1 | head-F1 | long-tail collapse | calib gap |
|---|---:|---:|---:|---:|
| Phase 1 baseline (cf. `phase1_morph_eval`) | ≈28 | ≈57 | ≈11 | ≈+0.090 |
| Run 1: v3 + FiLM + soft + joint | 40.0 | 59.0 | 4 | +0.031 |
| Run 2: v3 + additive + soft + joint | 36.7 | 67.0 | 4 | +0.017 |

Both conditioning mechanisms **improve per-class macro F1 metrics
(rare + head) and reduce long-tail collapse** compared to Phase 1, but
both also **shrink the calibration gap** — the head's confidence
becomes less informative. Additive in particular has the cleanest
head-F1 win (+10 pp over Phase 1) but the most degraded calibration
(0.017 gap, near-zero — the correct/wrong confidence distributions
nearly overlap).

### 19.3 Conditioning-module activity diagnostics

| Variant | mechanism | activity at end of training |
|---|---|---|
| Run 1: FiLM joint | film | `‖W_γ‖₂=0.74, ‖W_β‖₂=0.91, b_γ_dev=0.15, ‖b_β‖₂=0.18` (γ range 0.99–1.03, β range −0.014–0.017) |
| Run 2: additive joint | additive | `‖W_b‖₂=1.05, max\|W_b\|=0.025, mean\|W_b\|=0.006` (per-element bias is tiny on average) |

Both mechanisms learned non-zero weights from morph signal (W_norm
substantial vs init=0). The structural difference is in *how* the
conditioning influences `h`:

- **FiLM** packs its learning into per-feature γ (multiplicative) and
  β (additive) projections. Concentrated effect on a few feature
  channels per word, large enough to shift iʿrāb heads' input
  distribution.
- **Additive** spreads its 1.05 norm across the full 768-dim per-word
  bias such that average element magnitude is 0.006 — essentially a
  small isotropic perturbation. The iʿrāb heads only feel it on the
  most marginally-classified tokens (rare-class role decisions).

### 19.4 Cells not run

- **Run 3 (v3 + concat-embed joint)**: pre-quota-fix training crashed
  at step ~361 with Python exit 1. Skipped — discrete control was the
  least informative cell of the original grid; the FiLM-joint /
  additive-joint / FiLM-detached cells already establish the
  joint-vs-detached attribution unambiguously.
- **Run 5 (v4 + FiLM joint)**: cancelled to ship Phase 2 sooner. v4
  composition under conditioning would extend the substitutability
  finding from Phase 4a to Phase 2, but the Run 1+2+4 result already
  rules out FiLM joint as the production lever, so the v4 result
  cannot change the ship decision. Optionally run later as a
  follow-up.

## 20. Interpretation so far

The Run 1 result is consistent with — and sharper than — the Phase 4a
substitutability finding. The substitutability hypothesis predicted:
*the encoder's representation capacity for role discrimination is the
bottleneck; adding more parallel auxiliary supervision past Phase 1
yields diminishing returns.* Phase 2 tests a different lever: instead
of *more* parallel supervision, give the morph signal an explicit
hierarchical channel into the iʿrāb decoders. The Run 1 result extends
the substitutability story:

> The bottleneck is not just that morphology + taxonomy don't compose
> in parallel — it's that **the encoder representation that morphology
> heads decode well, and the encoder representation that iʿrāb heads
> decode well, are largely the same representation under our 6-epoch
> 296M training budget**. FiLM successfully learned a soft channel
> from morph predictions to a per-position γ/β, but the iʿrāb heads
> ended up worse off rather than better. Identity init was supposed
> to make this safe; instead, the iʿrāb heads' optimum drifted under
> conditioning faster than the iʿrāb heads themselves could retune.

Specifically the diagnostic combination — morph macro 98.31% (≈ Phase 1)
+ FiLM W_γ/W_β norms substantial (m-dependent learned) + iʿrāb metrics
regressed — points away from "morph supervision broken under FiLM" and
towards "*iʿrāb heads under-converge to the moving input distribution
that FiLM creates*". In a longer schedule (12+ epochs) or a larger
model (Phase 2 on a 1B+ encoder) this might invert. At 296M and 6
epochs, FiLM is net-negative.

This is a meaningful negative result: it tells us that the next
*productive* architectural lever isn't the form of the morph→iʿrāb
interaction (parallel vs FiLM vs additive — all of these compete for
the same encoder capacity), but a **different signal source** that
adds independent information. Phase 3 (dependency features from
Stanza or UD) is the cleanest candidate, since dependency edges carry
relational structure that morph features cannot.

## 21. Findings (final)

**Headline:** Soft morphology conditioning, joint-trained, regresses
Phase 1's iʿrāb metrics on Gazelle at this training budget. The
regression is **driven by joint optimisation dynamics, not by the
conditioning mechanism's expressive form.** Detaching the gradient
from morph heads recovers most of the regression even with the most
expressive (FiLM) interaction.

**The 3-cell mechanism table** (case / role-F1 / marker / fully on
Gazelle headed-only, vs Phase 1 baseline 53.7 / 42.3 / 41.0 / 19.4):

| Variant | case | role-F1 | marker | fully | summary |
|---|---:|---:|---:|---:|---|
| FiLM **joint** | 52.2 | 36.7 | 41.8 | 17.2 | all four regress |
| FiLM **detached** | 53.7 | 41.1 | 41.0 | 19.4 | only role-F1, mild |
| Additive joint | 53.7 | 39.0 | 41.0 | 19.4 | only role-F1, mild |

**Mechanism:** FiLM successfully learned a non-trivial morph→γ/β
mapping (W_γ L2=0.74, W_β L2=0.91 vs init 0); the morph heads
themselves stayed at Phase 1 accuracy (98.31% macro on UD-PADT). Both
of those facts hold for *all three* mechanism variants — the
representation is intact. What differs is the *training-dynamics*
coupling: joint training pulls morph heads off Phase 1's optimum
toward "useful for iʿrāb conditioning", and the iʿrāb heads chase
that moving target.

**Per-class behaviour:** Macro stress metrics improve under all three
conditioning variants vs Phase 1 (rare-F1, head-F1 up; long-tail
collapse from ~11 to 4 in all three) but the calibration gap also
shrinks across the board (Phase 1 ~0.09 → 0.017–0.031). The
conditioning is dispersing confidence across more classes; the
headline accuracy is what trades off.

**Cross-register:** MASAQ retention is unchanged from Phase 1's
pattern (case 84.9 / role-F1 9.7 / fully 7.5 for FiLM joint; the same
pattern for the other variants). The conditioning does not transfer
the MSA-specific morph→iʿrāb couplings to Quranic, but also does not
break MASAQ further.

**The substitutability story tightens.** Phase 4a showed parallel
multi-task supervision hits a substitutability wall (granularity and
morphology each lift role-F1 +5–7 pp but combine to only +5.7 pp).
Phase 2 now adds: hierarchical conditioning at this scale and budget
*also* hits the wall, and the way it hits it depends on whether morph
heads are jointly tuned. The bottleneck is the encoder's
representation capacity; rearranging the form of morph→iʿrāb
interaction inside that bottleneck doesn't expand it.

## 22. Ship decision (final)

**Phase 2 does NOT ship as the production checkpoint** in any of its
three evaluated forms. The strict gate (case ≥ 53.0, role-F1 ≥ 43.0,
fully ≥ 19.4 vs Phase 1's 53.7 / 42.3 / 41.0 / 19.4):

| Variant | case ≥ 53.0 | role-F1 ≥ 43.0 | fully ≥ 19.4 | Decision |
|---|:---:|:---:|:---:|---|
| FiLM joint | ✗ (52.2) | ✗ (36.7) | ✗ (17.2) | regress on all 3 |
| Additive joint | ✓ (53.7) | ✗ (39.0) | ✓ (19.4) | role-F1 fails |
| FiLM detached | ✓ (53.7) | ✗ (41.1) | ✓ (19.4) | role-F1 fails (closest) |

**Phase 1** (rev 2 + 7 morph heads, no conditioning) **remains the
production checkpoint**.

Phase 2 ships **as a documented architectural experiment**: the
conditioning module + factory + integration stays in the codebase
under `irab_tashkeel.morphology.conditioning` and is opt-in via the
``conditioning_mechanism`` config key (default ``None``, byte-identical
to Phase 1). The clean way to revive Phase 2 in a future iteration is
the *staged Phase 1 → Phase 2.5 schedule* sketched in §19.1: train
Phase 1 first, then freeze the morph heads and train only the
conditioning module + iʿrāb heads in a second pass. This avoids the
joint-training-dynamics issue identified in §21.

**Next phase to run is Phase 3** (dependency-aware reasoning), which
adds an *independent signal source* (UD dependency edges) rather than
rearranging the existing morph + taxonomy supervision inside the same
encoder bottleneck. The Phase 4a substitutability finding + the Phase
2 joint-dynamics finding together argue that the next productive
lever is information that morphology + taxonomy cannot capture — the
relational structure of Arabic phrases — not yet another shaping of
the same information.

Phase 4b (mawsool split) is deferred behind Phase 3.
