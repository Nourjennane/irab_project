# Phase 5 — Hierarchical Case Decoder

> Output-side hierarchy: condition the case head on the role head's
> softmax output. Sidesteps the Phase 2 joint-dynamics issue because
> output-side conditioning does not pull a head's *representation*
> off-distribution — only its *output logits* are biased by another
> head's prediction. Layered on top of Phase 3-A (current production).

## 1. Motivation — the encoder representation is already relational

Phase 3-A (rev 2 + Phase 1 morph + Stanza UD dep features as static
input augmentation) is the new production checkpoint: case **56.7**,
role-F1 41.3, marker 44.8, fully 20.1 on Gazelle. Three of four
metrics improve simultaneously vs Phase 1 baseline (53.7 / 42.3 / 41.0
/ 19.4). The encoder representation now carries the relational signal
that morph + taxonomy alone could not capture.

But the *decoders* are still independent:

```
pooled_irab  ──────────────► case head      (5 classes)
       │
       ├──────────────────► role head     (25 classes)
       │
       └──────────────────► marker head    (18 classes)
```

In Arabic iʿrāb, **case is determined in large part by role**:
- *fail* (subject) → raf (nominative)
- *mafoul_bih* (direct object) → nasb (accusative)
- *ism_majrur* (noun governed by preposition) → jarr (genitive)
- *mubtada* (subject of nominal sentence) → raf
- *naib_fail* (passive subject) → raf
- *khabar* (predicate) → raf
- etc.

The role → case dependence is near-deterministic for ~70% of role
labels in our 25-label canonical taxonomy. Yet the case head currently
predicts independently from the role head, so it can produce
inconsistent (role, case) pairs. Phase 5 makes case conditional on
role explicitly.

## 2. Architecture — soft additive role-to-case bias

```
pooled_irab  ──► role head ──► role_logits ──► softmax ──► role_softmax (25-d)
       │                                                       │
       │                                                       │  (detached or joint)
       │                                                       ▼
       └──► case_head_base ──► case_logits_base ─►(+)─►  role_to_case_bias
                                                  │       (25→5 linear)
                                                  ▼
                                             case_logits_final (5)
```

The case head's existing 768→5 linear stays. We add a small
`role_to_case_bias = nn.Linear(N_ROLE, N_CASE, bias=False)` that
projects `role_softmax (25-d) → case_bias (5-d)`. The final case logit
is the sum:

```
case_logits = case_head(pooled_irab) + role_to_case_bias(role_softmax)
```

**Identity initialisation:** `role_to_case_bias.weight = 0` so step 0
is byte-equivalent to Phase 3-A. Gradient from `L_case` teaches the
role-to-case mapping as training proceeds.

This is the same shape as Phase 2 *additive joint*, but on logits
instead of features, and on a much smaller (25-d) input than Phase 2's
80-d concat. Phase 2 additive joint preserved 3 of 4 metrics; the
analog hypothesis is that Phase 5 will also preserve case + marker +
fully (and ideally improve case via the role context).

## 3. Joint vs detached on `role_softmax`

Phase 2 finding: joint training of head A whose output conditions head
B can drift A's representation toward "useful as conditioning input"
rather than its own supervision.

But Phase 5 is *output-side*: role_softmax goes through a final
linear projection (small: 25×5 = 125 params) not into another head's
input features. The risk of representation drift is much lower:
- The role head's INPUT features (`pooled_irab`) are unchanged.
- Only the role head's *output logits* feed the case head.
- Role head still has full iʿrāb supervision (loss weight 1.5).

We default to **joint** (gradient flows from `L_case` through
`role_to_case_bias` and softmax back into role logits). Joint training
should reinforce role supervision rather than corrupt it because the
role head gets BOTH role labels AND case-derived gradient.

We will run a **detached** ablation as a control to confirm the joint
choice is correct (`role_softmax = role_softmax.detach()`).

## 4. Decision gate

Same soft two-of-three gate as Phase 3-A, vs the **Phase 3-A
baseline** (the new production: 56.7 / 41.3 / 44.8 / 20.1):

Phase 5 ships as new production *only if* it beats Phase 3-A on at
least two of {case, role-F1, fully} while not regressing any of them
by more than 1.0 pp.

The hypothesis is that hierarchical case will:
- Improve case (cleaner role→case mapping)
- Improve fully (case+role consistency)
- Not regress role-F1 (joint training reinforces, not corrupts)
- Not affect marker

If the gate fails (e.g. role-F1 drops more than 1 pp), Phase 5 ships
as opt-in and Phase 3-A stays production.

## 5. Reversibility

Phase 5 changes ship in one new module + additive changes to
`dep_aware_model.py` (new `enable_case_hierarchy` flag, default False).
With `enable_case_hierarchy=False`, the model graph is byte-identical
to Phase 3-A.

## 6. 2-cell ablation

| Variant | role_softmax detached? | what it tests |
|---|:---:|---|
| Phase 5-A — joint (primary) | ✗ | hypothesis: joint reinforces, not corrupts |
| Phase 5-B — detached (control) | ✓ | does the joint path hurt vs static role context? |

Both layered on top of Phase 3-A. `role_to_case_bias` zero-initialised
in both. Run 1 (5-A) is the gate; if it ships, run 5-B as a
diagnostic.

## 7. HPC schedule

2 retrains × ~8 min (matches Phase 1 / Phase 3-A wallclock — the
hierarchical bias adds ~125 trainable params, negligible overhead).
Plus eval (~3 min each). Total ~25 min.

## 8. Inference path

The case head needs `role_softmax` at inference, which is computed on
the same forward pass (role head output before case head). No external
input needed. Inference path is straightforward and has no
distribution-mismatch risk (unlike Phase 3, where the predictor needed
to remember to run `dep_proj`).

## 9. Risks / known gotchas

1. **Role head representation shift.** Even though output-side
   conditioning is gentler than Phase 2's input-side, the joint
   training does pass gradient through softmax back into role logits.
   The mathematical drift is bounded (softmax output is in the simplex
   so the gradient is normalised), but worth monitoring.
2. **Class-weighted role loss interaction.** The role head's
   sqrt-inv-frequency class weighting up-weights rare classes. If the
   joint case loss disagrees on rare classes (e.g. *kana_sister* + raf
   coupling is rare), the role head might be pulled toward more
   common (case-easier) classes. Diagnostic: per-class role-F1 on the
   9 rarest classes vs Phase 3-A.
3. **Identity init alone is not sufficient.** Phase 3 taught us that
   any architectural change adding a new transformation needs an
   inference-vs-training-distribution check. Phase 5 case head sees
   `role_softmax` at both training and inference — same forward pass,
   no distribution mismatch by construction. Lower risk.

## 10. Out of scope

- Hierarchical marker decoder (Phase 6 — defer behind 5-A).
- Joint case+role+marker hierarchical (would over-couple at this
  scale).
- Constraint reranking interaction with hierarchical case (test post-
  ship).

## 11. Run 1 (Phase 5-A) — gate result

**Phase 5-A fails the soft gate. Phase 3-A remains the production
checkpoint. Phase 5 ships as opt-in archival.**

Configuration (run `phase5_491263`):
- 6 epochs joint training on `data/morph_v1_dep` (Phase 3-A corpus)
- Layered on Phase 3-A: rev 2 + Phase 1 morph + dep features + role→case bias
- `role_to_case_bias = nn.Linear(25, 5, bias=False)`, zero-init
- Joint training (case loss flows through softmax back into role logits)

### 11.1 Gazelle headline (heads only, structured_v1)

| Metric | Phase 3-A baseline | Phase 5-A | Δ |
|---|---:|---:|---:|
| case | 56.7 | 56.0 | **−0.7** |
| role-F1 | 41.3 | 41.3 | 0.0 |
| marker | 44.8 | 43.3 | **−1.5** |
| fully | 20.1 | 20.1 | 0.0 |

Phase 5-A beats Phase 3-A on **zero** of {case, role-F1, fully}. The
soft two-of-three gate requires ≥ 2 wins; with zero wins, the gate
fails. Marker regresses by 1.5 pp (not in the gate set but still
documented).

### 11.2 Mechanism interpretation — the architectural lesson sharpens

Phase 3-A passed the gate because dep features were a *genuinely new
information source* (relational signal that morph + taxonomy could not
express). Phase 5's role→case bias is the opposite: it re-uses the
role head's *predictions* as a bias on the case head's logits — both
heads are already trained on the same encoder representation with
their own labels. No new information enters the model. The role→case
bias matrix can only redistribute what's already in the role logits.

The case head was already saturated by Phase 3-A's dep features (the
+3.0 pp case improvement Phase 3 delivered came from relational signal
that the dep features carried). Adding a small role→case bias on top
of that has nothing left to give.

The role-F1 staying flat (0.0) confirms the joint training did not
corrupt the role head — the Phase 2 joint-dynamics issue does not
apply to output-side conditioning, as expected. So the architectural
choice was correct; the limitation is about *information content*,
not training stability.

### 11.3 Generalisation of the architectural lesson

Phase 1 → Phase 4a → Phase 2 → Phase 3 → Phase 5 sequence delivers a
clean rule:

> At 296M / 6 epochs, **rearranging the same supervision plateaus.
> Adding genuinely new supervision unlocks gain.**

- Phase 4a: 25→34 taxonomy expansion = same supervision (per-word role
  labels, just more granular). Plateau.
- Phase 2: rearrange how morph→iʿrāb interaction happens. Plateau.
- Phase 3: dep features are new information (relational, not per-word).
  Gain.
- Phase 5: role→case bias = rearranges existing role predictions. Plateau.

This rule predicts Phase 6 (hierarchical marker, conditioned on
case+role) will also plateau — the marker head is already getting case
+ role information indirectly through the shared encoder
representation, and adding an explicit case_softmax+role_softmax→
marker bias just re-uses signal already there. We document Phase 6 as
the symmetric prediction; if it confirms, we have a clean four-cell
case study (Phase 4a, Phase 2, Phase 5, Phase 6 = all rearrangements,
all plateau; Phase 3 = new info, gain).

## 12. Ship decision (final)

**Phase 5 does NOT ship as production. Phase 3-A remains the
production checkpoint.** The role→case bias module + factory + flag
stay in the codebase under `enable_case_hierarchy=False` default;
opt-in for downstream consumers who want the case+role consistency
guarantee at the cost of −0.7 pp case + −1.5 pp marker.

Production lineage: rev 2 → Phase 1 → **Phase 3-A**. Phase 4a, Phase 2,
Phase 5 all opt-in.
