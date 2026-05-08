# Phase 6 — Hierarchical Marker Decoder

> Output-side hierarchy: condition the marker head on case + role
> softmax outputs. Same design pattern as Phase 5 (output-side
> conditioning sidesteps Phase 2 joint-dynamics). Layered on Phase 5
> if the Phase 5 gate passes, otherwise on Phase 3-A.

## 1. Motivation

Phase 5 conditions case on role. Phase 6 conditions marker on case +
role. The dependence is similarly near-deterministic in Arabic iʿrāb:

| (role, case) → marker | Example |
|---|---|
| any noun + raf → `damma_visible` (or `damma_estimated` for indeclinable) | *الكتابُ* |
| any noun + nasb → `fatha_visible` (or `fatha_estimated`) | *الكتابَ* |
| any noun + jarr → `kasra_visible` (or `kasra_estimated`) | *الكتابِ* |
| dual + nasb → `nun_dropped + fatha_alif` | *كتابَيْن* |
| sound masc plural + nasb → `nun_dropped + ya_kasra` | *المعلمين* |
| etc. |

The marker label is a near-function of (case, role) for the bulk of
labels, with a few exceptions (sound masc plural; dual; the five
nouns; mood markers for verbs).

## 2. Architecture — joint additive bias from case + role

```
case_logits   ──► softmax ──► case_softmax (5)    ─┐
role_logits   ──► softmax ──► role_softmax (25)   ─┼──► (case_role_concat) ──► role_case_to_marker_bias
                                                  │       (5+25=30 → 18 linear, zero-init)
                                                  ▼
            marker_head_base ──► marker_logits_base ─►(+)─► marker_logits_final (18)
```

Same pattern as Phase 5: small zero-initialised linear, additive on
the marker logits. Total new params: 30 × 18 = 540, negligible.

**Identity init:** `case_role_to_marker_bias.weight = 0` so step 0 is
byte-equivalent to Phase 5 (or Phase 3-A if Phase 5 didn't ship).

## 3. Layered choice

If Phase 5 ships as new production:
- Phase 6 layered on Phase 5 (case_logits already include role bias)
- Marker conditioning sees the *hierarchical* case_softmax

If Phase 5 doesn't ship:
- Phase 6 layered on Phase 3-A (case_softmax is independent)
- Phase 5 reverts to opt-in / archival

The choice is set by `enable_case_hierarchy=true/false` in the config.
Phase 6's `enable_marker_hierarchy=true` works either way (it consumes
whatever case_logits the model produces).

## 4. Joint vs detached

Default joint (gradients flow from `L_marker` through softmax back into
case_logits and role_logits). Same logic as Phase 5: case + role heads
have full iʿrāb supervision (loss weights 1.0 + 1.5), and marker
gradient (weight 1.0) reinforces rather than corrupts.

Detached available as ablation but not the default.

## 5. Decision gate

Soft two-of-three vs the **production baseline** (Phase 5 if it
shipped, Phase 3-A otherwise). Beat baseline on at least 2 of {case,
role-F1, fully} with no regression > 1.0 pp.

## 6. Reversibility

`enable_marker_hierarchy=false` (default) keeps the model graph
byte-identical. New module surface: one `nn.Linear` in
`DepAwareStructuredModel`, zero-init, gated by the new flag.

## 7. Out of scope

- Marker conditioned on POS (POS is independent in canonical schema).
- Joint case+role+marker conditioning (would over-couple).
- Schedule-based unfreeze (case head froze after Phase 5, then marker
  trained alone) — defer.

## 8. Run 1 (Phase 6) — gate result

**Phase 6 fails the gate even more clearly than Phase 5. Phase 3-A
remains production. Phase 6 ships as opt-in archival.**

Configuration (run `phase6_491270`):
- 6 epochs joint training on `data/morph_v1_dep`
- Layered on Phase 3-A (no Phase 5 stack — clean test of marker
  hierarchy in isolation)
- `case_role_to_marker_bias = nn.Linear(N_case + N_role, N_marker, bias=False)`,
  zero-init
- Joint training (marker loss flows through softmax back into case + role logits)

### 8.1 Gazelle headline (heads only, structured_v1)

| Metric | Phase 3-A baseline | Phase 6 | Δ |
|---|---:|---:|---:|
| case | 56.7 | 56.0 | **−0.7** |
| role-F1 | 41.3 | **38.8** | **−2.5** |
| marker | 44.8 | 43.3 | **−1.5** |
| fully | 20.1 | 18.7 | **−1.4** |

Zero wins among {case, role-F1, fully}. Role-F1 regresses by 2.5 pp,
fully by 1.4 pp — both exceed the 1.0 pp soft-gate threshold. **Phase 6
fails clearly.**

### 8.2 Why Phase 6 hurts more than Phase 5

Phase 5's role→case bias is small (25 → 5 = 125 params). Phase 6's
case_role→marker bias is larger (5+25 → 18 = 540 params, plus the
softmax-of-case_logits dependency creates a gradient path that
modifies case loss through marker loss). The joint training drift on
case + role logits is now amplified by the marker head's gradient
flowing back through TWO softmaxes simultaneously. The role head loses
2.5 pp role-F1 because the joint training pulls its logits toward
"useful as marker conditioning input" rather than "directly predict
role".

This is the symmetric prediction the Phase 5 writeup made (§11.3).
Stacking output-side conditioning when both heads already share the
same encoder representation does not add information — it only
redistributes the existing prediction mass, and the redistribution
costs accuracy where it touches the most heads.

### 8.3 Ship decision

**Phase 6 does NOT ship. Phase 3-A remains production.** The marker
hierarchy module + factory + flag stay in the codebase under
`enable_marker_hierarchy=False` default. Opt-in only for downstream
consumers who explicitly want case+role+marker consistency at the
documented cost.

### 8.4 The four-cell architectural case study (closed)

| Phase | Intervention | New info? | Result |
|---|---|:---:|---|
| 4a | 25 → 34 taxonomy | ✗ (same labels, more granular) | plateau |
| 2 | Morph→iʿrāb conditioning rearranged | ✗ | regress |
| 5 | Role→case output bias | ✗ | slight regress |
| 6 | Case+role→marker output bias | ✗ | larger regress |
| **3** | **UD dep edges** | **✓** | **gain** |

This is now a clean four-negative-one-positive empirical pattern. The
generalisation **at 296M / 6 epochs, rearranging the same supervision
plateaus or regresses; orthogonal information sources unlock gain**
holds across both encoder-side conditioning (Phase 2), input-side
augmentation (Phase 3 vs no-add-info baseline), and output-side
hierarchical decoders (Phase 5, Phase 6). The pattern is robust.

The next architectural levers — given this constraint — are the
remaining genuinely-new-information sources:
- Phase 9 (grammar memory expansion): adding lexicon-based features
  for rare constructions that morph + dep + taxonomy do not capture
- Phase 11 (explanation engine): not a new training signal but a new
  *output modality* (generate rationales conditioned on the
  predictions); orthogonal to the supervision plateau
- #39 (rare-construction synthetic augmentation): adds new TRAINING
  DATA (not a model architecture change) for under-covered
  constructions; bypasses the encoder-bottleneck argument by changing
  the data distribution itself

Phase 7 (CRF redo) and Phase 8 (per-constraint ablation) are
rearrangements within the existing supervision; the four-cell pattern
predicts they will plateau too. We document Phase 6 as the closing
data point on this generalisation; further phases focus on genuinely
new information.
