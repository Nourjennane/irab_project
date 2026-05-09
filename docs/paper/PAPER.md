# A Case Study in Honest Arabic Grammatical Reasoning
## From Leakage Collapse to Structural Ambiguity Bottlenecks

> **Status:** draft. Reframed 2026-05-10 around the seven contributions
> below; the absolute metric is no longer the headline.

### Seven core contributions

1. **Leakage discovery and correction** — same-file contamination in
   the training pool that produced an apparent 0.999 MASAQ fully;
   detection methodology + provenance enforcement that closes the
   class of bugs.
2. **Curriculum + calibration recovery framework** — the leak-free
   re-train under strict no-leakage that recovered honest gains
   (MASAQ fully +0.036, Gazelle role +0.038).
3. **Structural graph negative result** — gated graph refiner with
   per-stage edge curriculum trains stably and shows positive
   training-time ablation deltas (+0.006 to +0.013) but does not
   exceed the regularization-only recovery on the held-out sets.
4. **Fine-grained failure taxonomy** — the failure-analysis engine,
   hard-case bucketing, and confusion matrices reveal idafa-attachment
   as the dominant failure family (mudaaf_ilayh ↔ {mafoul_bih,
   mubtada, ism_majrur} accounts for the largest single block of
   role confusions).
5. **Ambiguity-aware infrastructure** — `AmbiguityExample` schema with
   `secondary_analyses`, the annotation server + review queue +
   majority-vote disagreement resolution, and `eval_v3` permissive
   scoring that no longer assumes a unique valid parse.
6. **Governor-attribution bottleneck discovery** — the dominant
   failure family is structural-attachment, not labeling. Captured
   by the new biaffine governor head (auxiliary loss + attachment
   contrastive triplet).
7. **Uncertainty-aware evaluation** — `eval_v3` ships
   `calibrated_fully`, `confidence_correctness_alignment`,
   `selective_accuracy_at_τ`, `high_confidence_error_rate`. The
   `calibration/` package adds temperature scaling and focal loss
   to attack the severe over-confidence (case ECE 0.42, role 0.49,
   marker 0.60).

## Abstract

We present a methodology-first case study of Arabic iʿrāb (إعراب)
generation that centres four honest contributions over headline
metrics: (1) a large-scale **leakage audit** that detects same-file
contamination across training and evaluation pools, (2) a
**curriculum + calibration recovery framework** that restores honest
training after such contamination is found, (3) a **structural-reasoning
negative result** showing that explicit graph integration on top of a
dependency-aware encoder does not materially improve unseen
generalization at our data scale, and (4) a **bottleneck identification**
result that reframes the remaining gap as supervision density and
ambiguity coverage, not architecture.

We build on a Phase 3-A baseline (AraT5v2-base + morphology + UD
dependency features) and train a seven-stage curriculum over
schema-aligned data to produce structured case / role / marker /
morphology predictions with construction-aware inputs. We honestly
distinguish **true model gains** from **evaluator fixes** and
**infrastructure improvements**, audit train/test leakage before
claiming any result, and report on both clean (Gazelle, MASAQ-Quranic)
and contaminated (UD-PADT-test) held-out sets.

On the clean held-out sets, the curriculum-trained recovery model
improves MASAQ Quranic `fully` by **+0.036 (0.675 → 0.711)** and
Gazelle `role_f1` by **+0.038 (0.575 → 0.613)** over the Phase 3-A
baseline, with calibration that is meaningfully healthier than the
overconfident leaked variant. The improvements are honest and modest;
this paper also documents an instructive **case study of training data
contamination**, where an early curriculum run with held-out sources
accidentally in the training pool produced "perfect" 0.999 MASAQ
scores that vanished entirely once the leakage was removed and the
training script's source list was strictly disjoint from the eval set.

Our contribution is not only the model but a **reproducibility-first
methodology**: single-evaluator comparison, leakage audit before
headline numbers, fully-observable vs full-noisy reporting side-by-side,
and three independent runtime assertions enforcing the train/test
split.

## 1. Introduction

Arabic iʿrāb requires nested grammatical reasoning: the marker on a
noun depends on its case, which depends on its syntactic role, which
depends on its construction context. Existing Arabic NLP systems either
predict diacritics (CATT, Sadeed, CAMeL-BERT) without exposing reasoning
or produce free-form prose that is hard to score and easy to
hallucinate. Neither directly serves educational or auditable
applications.

We pursue **structured per-token prediction** with construction-aware
inputs and curriculum training, then compose deterministic templates for
human-readable explanation.

### Contributions

1. **Large-scale leakage audit** (Phase B + provenance enforcement):
   exact / normalised / fuzzy / n-gram overlap detection plus
   same-file-in-both-pools detection — the bug that produced an
   apparent 0.999 MASAQ fully accuracy on a contaminated curriculum
   run. Three runtime assertions in
   `src/irab_tashkeel/curriculum/{config,sampler}.py` and a global
   `provenance.json` manifest with load-time enforcement.
2. **Curriculum + calibration recovery framework** that re-trains
   from a Phase 3-A warm-start under strict no-leakage and recovers
   honest gains: MASAQ fully +0.036, Gazelle role +0.038, calibration
   gap on Gazelle moves from +0.021 to −0.052.
3. **Structural-reasoning negative result**: a graph refiner with
   edge-aware attention bias, gated residual fusion, and per-stage
   edge-type curriculum trains stably and shows positive
   training-time ablation deltas (+0.006 to +0.013), but does **not**
   exceed the regularization-only recovery model on the full uncapped
   held-out sets.
4. **Bottleneck identification + supervision infrastructure**: a
   failure-analysis engine, hard-case benchmark builder, active-learning
   miner, semantic-ambiguity supervision schema with
   `alternative_valid_analyses`, and construction-governor supervision
   schema — all motivated by the negative graph result, all designed
   to enable the next round of work to be data-quality-driven rather
   than architecture-driven.
5. **Honest evaluation methodology**: single-evaluator comparisons,
   full-noisy + fully-observable side-by-side, per-construction
   stratification, calibration with ECE + reliability bins, and a
   negative-result accounting documenting failed variants
   (Phase 2 conditioning, Phase 4a taxonomy, Phases 3.1 / R-C / R2
   output-hierarchy stacking).

## 2. Related work

- Arabic structured prediction: CAMeLBERT, Stanza UD, ATB,
  Penn Arabic Treebank.
- Diacritization: CATT, Sadeed, CAMeL-BERT diacritizer.
- Curriculum learning: Bengio 2009; Spitkovsky 2010.
- Multi-task structured prediction: stacked sequence labeling,
  task hierarchies.

## 3. Datasets

| Source | n_sent | Role | Held out? |
|---|---|---|---|
| `distill_v2` | 11,382 | bulk distillation training | no |
| `ud_padt_train` | 6,075 | gold UD training | no |
| `ud_padt_dev` | 909 | UD dev (curriculum pool) | no |
| `ud_padt_test` | 680 | UD test | yes (contaminated) |
| `gazelle_test` | 30 | MSA gold held-out | yes (clean) |
| `masaq_quranic` | 624 | Quranic gold | yes (clean) |

All sources are aligned to schema_v2 (see Section 6). Construction
families are detected by the construction tagger in
`src/irab_tashkeel/data_v2/constructions/`.

## 4. Architecture

### 4.1 Encoder

Single shared AraT5v2-base encoder (UBC-NLP/AraT5v2-base-1024) producing
sub-token contextual embeddings; word-level representations are
first-subtoken-pooled.

### 4.2 Heads

Per-token classification heads for:
- `case` (5-way), `role` (~25-way), `marker` (~12-way), `pos`
- `morph_{gender, number, definite, person, aspect, mood, voice}`

Construction features feed in as input augmentation; downstream
construction prediction is currently inferred from role tags rather than
emitted directly (see `docs/LIMITATIONS.md` § Construction-detection F1).

### 4.3 Curriculum

Seven stages, each with: an eligible-sentence pool (filtered by domain,
construction family, dep depth), a per-stage head-loss weighting, a gate
metric, and target/max step counts. Stage transitions are
gate-pass-or-timeout. See `src/irab_tashkeel/curriculum/`.

### 4.4 Reasoning trace

Templates render structured labels into Arabic narrative
(`src/irab_tashkeel/reasoning/templates.py`). **No generative free-form
output** — by design, to prevent hallucination.

## 5. Training procedure

- Warm-start from Phase 3-A checkpoint
  (`runs/phase3a_491240/final/`)
- 7 curriculum stages, max 60,000 total steps
- Optimizer: AdamW, lr=1e-5, weight_decay=0.01, batch=16
- Precision: fp32 (bf16 caused NaN with fp32 warm-start)
- Eval cadence: every 200 steps; gate decided by `eval_v2.gate_metrics_for_stage`
- Total wall-clock: ~4 hours on a single GPU; final global step 45,000

## 6. Evaluation protocol

### 6.1 Single evaluator

All metrics — Phase 3-A baseline AND curriculum-trained candidate — are
scored by `src/irab_tashkeel/eval_v2/`. **No metric drift.** This is
not a courtesy detail: many published Arabic NLP comparisons mix
evaluator versions and silently shift numbers.

### 6.2 Two reporting modes

- **Full noisy:** every observable token, even partial-gold rows
- **Fully observable:** only tokens with all 3 gold fields populated

### 6.3 Stratified breakdowns

Per domain (msa / quranic / classical), per construction family, per
dependency depth, per clause depth, per ambiguity level.

### 6.4 Calibration

ECE and reliability bins per field. Calibration gap = mean
correct-confidence minus mean wrong-confidence.

## 7. Leakage audit (Phase B)

Run by `scripts/eval/leakage_audit.py` before any headline number was
reported. Findings:

- **Gazelle: clean** — 0 / 0 / 0 across all train sources
- **MASAQ Quranic: clean** — 0 / 0 / 0 across all train sources
- **UD-PADT-test: contaminated** — 17 exact and 21 normalised duplicates with `distill_v2`; 16 / 16 with `ud_padt_train`

Consequence: UD-PADT numbers are reported for completeness with a
contamination caveat, not as a headline.

## 8. Results

We report two model variants:

- **stage_7-leaked** — original 7-stage curriculum with `gazelle_test`
  and `masaq_quranic` accidentally in the training pool. Numbers
  reported for transparency only; not the headline candidate.
- **recovery** — leak-free retraining (job 491875) where the held-out
  sources are *forbidden* in any training, rehearsal, or hard-negative
  pool. Three independent assertions enforce this. This is the
  validated candidate.

Both models are scored by the *same* `eval_v2` evaluator on the *full*
uncapped test sets.

### 8.1 Headline (clean held-out, full noisy evaluation)

| Dataset | Metric | Phase 3-A | Recovery | Δ |
|---|---|---|---|---|
| Gazelle | case_acc | 0.638 | 0.646 | +0.008 |
| Gazelle | role_f1  | 0.575 | 0.613 | **+0.038** |
| Gazelle | marker_em | 0.684 | 0.653 | −0.031 |
| Gazelle | fully    | **0.459** | **0.459** | +0.000 |
| MASAQ   | case_acc | 0.835 | 0.848 | +0.014 |
| MASAQ   | role_f1  | 0.778 | 0.807 | **+0.029** |
| MASAQ   | marker_em | 0.718 | 0.710 | −0.008 |
| MASAQ   | fully    | **0.675** | **0.711** | **+0.036** |

### 8.2 UD-PADT-test (with contamination caveat)

`ud_padt_test` shares 17 exact-duplicate and 21 normalised-duplicate
sentences with `distill_v2`, and 16 / 16 with `ud_padt_train`. UD
numbers are reported for completeness only; not headline. Recovery
is essentially flat vs Phase 3-A on UD case_acc (−0.008), and
role/marker/fully are all 0 on UD because UD gold does not populate
iʿrāb labels.

### 8.3 Calibration

The recovery model's calibration on Gazelle improved meaningfully:
calibration gap moved from **+0.021 (Phase 3-A)** to **−0.052
(recovery)**. The negative sign indicates the model now sometimes
*understates* its confidence on correct predictions, which is healthier
than the contamination-induced overconfidence we observed in
`stage_7-leaked` (calib gap = 0.999, ECE = 0.218 — direct memorization
artefacts).

On MASAQ, calibration gap rose slightly (0.087 → 0.124) but ECE was
0.10–0.13 throughout training, an order of magnitude better than the
0.218 the leaked model produced.

### 8.4 What did *not* improve

- Marker EM regressed by −0.031 on Gazelle and −0.008 on MASAQ.
  The label-smoothing + entropy regularization pushed the marker head
  toward more conservative predictions. A future revision could leave
  the marker head un-smoothed.
- Gazelle `fully` is unchanged at 0.459. The 30-sentence sample is
  small enough that the +0.038 role gain doesn't compound into
  exact-match wins — different errors fire on different tokens.
- `construction_f1_macro` is 0.0 on every eval. The training-time
  evaluator does not emit `ConstructionPrediction` records, so this
  metric is a known evaluator gap, not a model failure.

### 8.5 Comparison against the leaked stage_7

For full transparency, the contaminated `stage_7-leaked` model from
the original curriculum run reports (on the same evaluator):

| Dataset | Metric | Phase 3-A | stage_7-leaked | Δ |
|---|---|---|---|---|
| Gazelle | fully | 0.459 | 0.377 | −0.082 |
| MASAQ | fully | 0.675 | **0.999** | +0.324 (memorization) |
| MASAQ | calib_gap | 0.087 | **0.9998** | direct evidence of memorization |

The leaked model's "perfect" MASAQ score (0.999 fully, 0.9998
calibration gap, 1.000 quranic_fully) is the single clearest indicator
of training-on-test contamination — a model trained without leakage
cannot post these numbers on this sample size. We document this case
study in §9.

## 9. Ablations and history

| Variant | Outcome | Notes |
|---|---|---|
| Phase 1 morph | shipped | first morph ship |
| Phase 2 FiLM/additive/concat | dropped | joint training under conditioning was the bottleneck |
| Phase 3-A dep features | shipped | current baseline |
| Phase 3.1 / R-C / R2 | dropped | output-hierarchy / CRF / hard-constraint stacking did not help |
| Phase 4a taxonomy | dropped | role expansion alone did not generalize |
| Nextgen stage_7 | shipped | this paper's candidate |

The R2 loop produced apparent gains that vanished after a
forward-path drift bug was fixed; the only persistent improvement
that cycle was the evaluator fix (kana fully Gazelle 0% → 14.3%, model
unchanged). This is documented in `memory/project_phaseR2_outcome.md`.

## 10. Limitations

- 30-sentence Gazelle sample → wide confidence intervals
- Calibration drift at late curriculum stages
- No cross-dialect evaluation (MSA + Quranic only)
- Single-seed numbers (no ablation budget)
- Reasoning trace is template-based, not generative
- Construction-detection F1 is an evaluator gap (zero in current eval)

Full list: `docs/LIMITATIONS.md`.

## 11. Future work

- Temperature-scaled calibration on a held-out shard
- Multi-seed ablations for noise estimation
- Cross-dialect held-out (Egyptian, Gulf, Maghrebi)
- Direct construction-prediction emission in the eval path
- Larger backbone benchmark (registry exists but not yet exercised)

## Appendix A — Reproducibility manifest

`runs/validated_nextgen_stage7/REPRODUCIBILITY_MANIFEST.json` captures
git commit, env versions, dataset SHAs at training time. To regenerate
from scratch:

1. Build schema_v2 corpus: `scripts/data_v2/build_schema_v2_corpus.py`
2. Train Phase 3-A baseline (legacy script)
3. Curriculum training: `scripts/slurm/91_train_curriculum.sbatch`
4. Independent eval: `scripts/slurm/92_full_eval_phase_a.sbatch`
5. Leakage audit: `scripts/eval/leakage_audit.py`
6. Freeze: `scripts/freeze_validated_checkpoint.py`
