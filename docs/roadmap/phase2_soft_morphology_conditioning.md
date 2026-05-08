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
