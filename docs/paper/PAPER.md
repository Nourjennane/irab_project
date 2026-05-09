# Hierarchical Neural-Symbolic Reasoning for Arabic Iʿrāb
## A Curriculum-Trained Multi-Head Architecture with Construction-Aware Evaluation

> **Status:** draft. Numbers in *italics* are placeholders to be filled
> from `docs/final_eval/final_eval_tables.json` once Phase A finishes.

## Abstract

We present a hierarchical neural-symbolic architecture for per-token
Arabic iʿrāb (إعراب) generation. Building on a Phase 3-A baseline
(AraT5v2-base + morphology + UD dependency features), we train a
seven-stage curriculum over schema-aligned data to produce structured
case / role / marker / morphology predictions with construction-aware
inputs. We honestly distinguish **true model gains** from **evaluator
fixes** and **infrastructure improvements**, audit train/test leakage
before claiming any result, and report on both clean
(Gazelle, MASAQ-Quranic) and contaminated (UD-PADT-test) held-out sets.

On the clean held-out sets, the curriculum-trained model improves
*fully-correct accuracy* by *Δ X.XX* points over the Phase 3-A
baseline; calibration, however, drifts unfavourably during late-stage
training, and we discuss this trade-off explicitly. Our contribution is
not only the model but a **reproducibility-first methodology**:
single-evaluator comparison, leakage audit before headline numbers, and
fully-observable vs full-noisy reporting side-by-side.

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

1. A hierarchical multi-head architecture combining morphology,
   dependency, role, case, and marker prediction over a single shared
   AraT5v2-base encoder.
2. A seven-stage curriculum (morphology → local syntax → simple
   constructions → nested syntax → semantic interactions →
   discourse-sensitive → Quranic/classical) with stage-specific gates.
3. An independent evaluation pipeline (`eval_v2`) reporting
   construction-specific, ambiguity-robust, and completeness-aware
   metrics — all computed by a single evaluator so baselines and
   candidate models are scored bit-identically.
4. A train/test leakage audit (Phase B) that surfaces *known
   contamination* in UD-PADT-test before any headline number is
   reported.
5. An honest negative-result accounting: we list the variants that did
   *not* improve over Phase 3-A (Phase 2 conditioning, Phase 4a taxonomy
   expansion, Phases 3.1 / R-C / R2 output-hierarchy stacking).

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

### 8.1 Headline (clean held-out)

| Dataset | Metric | Phase 3-A | Stage 7 | Δ |
|---|---|---|---|---|
| Gazelle | case_acc | *X.XX* | *Y.YY* | *Δ* |
| Gazelle | role_f1 | *X.XX* | *Y.YY* | *Δ* |
| Gazelle | marker_em | *X.XX* | *Y.YY* | *Δ* |
| Gazelle | fully | *X.XX* | *Y.YY* | *Δ* |
| MASAQ | case_acc | *X.XX* | *Y.YY* | *Δ* |
| MASAQ | role_f1 | *X.XX* | *Y.YY* | *Δ* |
| MASAQ | marker_em | *X.XX* | *Y.YY* | *Δ* |
| MASAQ | fully | *X.XX* | *Y.YY* | *Δ* |
| MASAQ | quranic_fully | *X.XX* | *Y.YY* | *Δ* |

(Numbers populated automatically by `scripts/eval/aggregate_full_eval.py`
into `docs/final_eval/final_eval_report.md`; this draft cites that
report.)

### 8.2 UD-PADT-test (with contamination caveat)

Reported only because the UD comparison is conventionally expected.
*Δ X.XX* points but fundamentally inflated by 17 exact-duplicate
sentences in `distill_v2` and 16 in `ud_padt_train`.

### 8.3 Calibration

Calibration gap rose from *0.025 (Phase 3-A)* to *0.20 (stage_7)*. The
curriculum-trained model is **more accurate but more overconfident**.
Section 9 attributes this to the head-loss reweighting at later
curriculum stages.

### 8.4 Per-construction breakdown

Largest gains: *kana_sisters*, *inna_sisters* — see
`docs/final_eval/final_eval_tables.json`.

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
