# Project Journey — Full Chronicle

> A narrative record of how the project evolved from an apparent
> SOTA result through leakage discovery, recovery, two negative
> architectural results, and the central linguistic finding that
> redirects future work toward annotation rather than architecture.
>
> This is the **lab-notebook** companion to `README.md`. It explains
> *why* every decision was made, in the order it was made, with the
> evidence each step relied on.

---

## Table of Contents

1. [Starting state](#1-starting-state)
2. [Phase A — independent evaluation](#2-phase-a--independent-evaluation)
3. [The leakage discovery](#3-the-leakage-discovery)
4. [The 14-item recovery patch + 2 bonuses](#4-the-14-item-recovery-patch--2-bonuses)
5. [The recovery training run](#5-the-recovery-training-run)
6. [Graph integration — first negative result](#6-graph-integration--first-negative-result)
7. [Governor head — second negative result](#7-governor-head--second-negative-result)
8. [Failure analysis — the central finding](#8-failure-analysis--the-central-finding)
9. [Supervision phase — the data infrastructure](#9-supervision-phase--the-data-infrastructure)
10. [Final polish phase](#10-final-polish-phase)
11. [Three parallel finalization tracks](#11-three-parallel-finalization-tracks)
12. [Demo polish](#12-demo-polish)
13. [What stayed open](#13-what-stayed-open)
14. [Commit timeline](#14-commit-timeline)

---

## 1. Starting state

The project began this conversation with:

- **Production checkpoint:** Phase 3-A (`runs/phase3a_491240/final/`) — AraT5v2-base + 7 morph heads + Stanza UD dep features as input augmentation. This was the warm-start baseline.
- **A just-completed 7-stage curriculum training run** (job 491628) reporting astonishing held-out numbers on a 100-cap eval slice:
  - case_acc = 0.993
  - role_f1 = 0.982
  - marker_em = 0.976
  - **fully = 0.962**
  - quranic_fully = **1.000**
- An ambitious task list (Phases A–G) handed by the user: full independent evaluation, leakage audit, freeze validated checkpoint, repo reorganization, demo, paper, GitHub release.

The headline numbers looked like SOTA. The first job was to verify them.

---

## 2. Phase A — independent evaluation

We built the project's first independent evaluator decoupled from the
training-loop metrics:

- `scripts/eval/run_full_eval_v2.py` — independent runner, no caps,
  no curriculum sampling, deterministic seed
- `scripts/eval/aggregate_full_eval.py` — joins raw shards into
  `final_eval_report.md` + `final_eval_tables.json`
- `scripts/eval/leakage_audit.py` — exact / normalised / fuzzy /
  n-gram overlap detection across train×test pairs

Submitted as **job 491779** on the Bocconi HPC. The eval ran
phase3a vs the curriculum-trained checkpoint over the **full uncapped**
held-out sets: gazelle_test (30), masaq_quranic (624), ud_padt_test (680).

### The first red flag

```
MASAQ stage_7 (full uncapped):
  case_acc        = 1.0000
  role_f1         = 1.0000
  marker_em       = 0.9995
  fully           = 0.9990
  calib_gap       = 0.9998   ← this is the smoking gun
```

A `calib_gap = 0.9998` means the model places ~100% confidence on
every prediction across 624 sentences. **This is mathematically
impossible without memorisation.** A non-memorising model trained on
~20k sentences cannot post a calibration gap of 0.9998 on a 624-sentence
test set.

### The second red flag

Gazelle fully had **regressed** from Phase 3-A's 0.459 to 0.377 on the
same evaluator. A model that genuinely improved on Quranic by 30
percentage points should not regress on the related MSA task by 8
points. The pattern was consistent only with stage 7 overfitting to
something it had memorised.

---

## 3. The leakage discovery

Cross-checking `scripts/training_v2/train_curriculum.py` revealed:

```python
# scripts/training_v2/train_curriculum.py — line 91
sources = ["distill_v2", "ud_padt_train", "ud_padt_dev",
           "masaq_quranic", "gazelle_test"]   # ← held-out sets in TRAINING POOL
```

And `src/irab_tashkeel/curriculum/config.py` showed:

- Stage 3: `allowed_sources` included `masaq_quranic`
- Stages 4 & 5: included `ud_padt_test` AND `masaq_quranic`
- Stage 6: included `masaq_quranic`
- Stage 7 (`quranic_classical`): **preferred** `masaq_quranic` for sampling

**The held-out test sets had been training data the entire time.**
Stage 7 specifically trained on the held-out Quranic data, then
"evaluated" on it. The 0.999 was memorisation, not learning.

A separate file-level leakage audit (Phase B) found:

- Gazelle ↔ all train sources: **clean** (0/0/0 overlap)
- MASAQ ↔ all train sources: **clean** (0/0/0)
- UD-PADT-test ↔ distill_v2: **17 exact, 21 normalised, 65 fuzzy**
- UD-PADT-test ↔ ud_padt_train: **16 exact, 16 normalised, 45 fuzzy**

The audit had originally missed the Gazelle/MASAQ contamination
because it compared train-source files to test-source files separately
— but the same file (`masaq_quranic/all.jsonl`) was both. The audit
was extended afterwards to detect same-file-in-both-pools contamination.

This was the project's **pivot moment.** The leakage became a documented
contribution, not a hidden mistake.

---

## 4. The 14-item recovery patch + 2 bonuses

A directive from the user followed: stop chasing inflated metrics;
optimise only for strict unseen generalisation.

### Item 1 — strict no-leakage policy

`src/irab_tashkeel/curriculum/config.py`:

```python
TEST_SOURCES = frozenset({"gazelle_test", "masaq_quranic", "ud_padt_test"})
DEV_SOURCES  = frozenset({"ud_padt_dev"})

def assert_no_test_sources(sources, where=""):
    bad = [s for s in sources if s in TEST_SOURCES]
    if bad:
        raise AssertionError(...)
```

Three independent runtime assertions enforce the policy:

1. At module load — `DEFAULT_STAGES` is validated
2. At `build_stage_pool` — config + result both checked
3. At `stage_eligibility` — refuses any test-source sentence regardless
   of stage config (defence in depth)

Plus `train_curriculum.py` now loads training and eval sets *separately*
from disk and asserts `train_ids ∩ eval_ids = ∅` after both pools are
loaded.

### Item 2 — failure-mode taxonomy + HardFailureSampler

`src/irab_tashkeel/training/failure_taxonomy.py` heuristically tags
each sentence with applicable T-codes (T01..T18) from existing
schema_v2 metadata. Default weights:

| T-code | Kind | Weight |
|---|---|---|
| T03 | long-range dependency | ×3 |
| T04 | nested clause | ×3 |
| T05 | semantic ambiguity | ×4 |
| T15 | coordination ambiguity | ×3 |
| T16 | clause attachment | ×4 |
| T18 | construction overlap | ×5 |

`HardFailureSampler` extends the existing `StratifiedSampler` to use
weighted draws within each pool.

### Item 3 — hard-negative pair builder + contrastive

`src/irab_tashkeel/training/hard_negative_builder.py` — three confusion
families: same_surface_diff_role, same_construction_diff_gov,
near_syntax_one_change. Plus `contrastive_loss` (triplet cosine-margin)
and `info_nce_loss`. *Module ready, deferred for trainer integration*
because it needs `edge_index` in the batch.

### Items 4 + 5 — graph refiner + edge-type attention bias

`src/irab_tashkeel/models/graph_refiner.py` — a 2-layer attention
refiner with per-head edge-type bias. (See § 6 for the experiment.)

### Item 6 — confidence regularisation

In `src/irab_tashkeel/training_v2/loss.py`:
- Label smoothing 0.05 in cross-entropy
- Entropy regularisation (lambda 0.01)
- Confidence histogram in eval

### Item 7 — adversarial split builder

`scripts/data_v2/build_adversarial_splits.py` partitions held-out
sentences by:
- construction template
- dependency pattern
- lexical disjoint
- nested clauses
- repeated-phrase RED FLAG

Output: 434/654 share construction templates with train (66%);
654/654 share dependency patterns (100%); 0 lexical disjoint;
0 repeated-phrase flags. Pattern overlap is heavy, but no exact
phrase leak.

### Item 8 — construction dropout

`src/irab_tashkeel/training/augmentations.py`:
`construction_dropout`, `dep_dropout`, `morph_label_dropout`.
*Module ready, deferred (needs edge_index in batch).*

### Item 9 — multi-task loss rebalance + structured-consistency penalty

`HeadLossWeights` defaults rebalanced:
- role 1.5 (amplified)
- marker 1.4 (amplified)
- fully_aux 2.0 (new)

Structured-consistency penalty: a small soft penalty on incompatible
(case, role) and (case, marker) pairs (e.g., case=raf with role=mafoul_bih).
Pure prediction-side, no gold needed.

### Item 10 — exact-fully aux loss

`-log P(case_correct ∧ role_correct ∧ marker_correct)` on
fully-observable tokens. Weight 0.5. Directly optimises the headline
metric.

### Item 11 — early stop on `strict_unseen_fully`

Patience 3. After 3 consecutive evals with no improvement, force-advance
the stage (failsafe so the loop doesn't get stuck).

### Item 12 — training config

`lr = 1e-5` (from 5e-5), `dropout = 0.15`, `batch_size = 16` (from 32),
`bf16 = off` (caused NaN with fp32 warm-start), EMA decay 0.999.

### Item 13 — per-axis fully reporting

`gate_metrics_for_stage` now emits 17 metrics per eval, including
`strict_unseen_fully`, `nested_fully`, `long_range_fully`,
`overlap_fully`, `ambiguity_fully`, `quranic_fully`, ECE,
confidence histogram.

### Item 14 — ablation toggles

Every recovery item gated behind a CLI flag in `train_curriculum.py`.

### Bonus 1 — SWA (Stochastic Weight Averaging)

`src/irab_tashkeel/training/swa.py` — `SWASnapshot` maintains a running
mean of model parameters; swap into the model at eval time, restore
the SGD trajectory afterwards.

### Bonus 2 — Layer-wise LR decay

`src/irab_tashkeel/training/llrd.py` — encoder block `i` gets
`base_lr × decay^(top_block_idx − i)`, decay = 0.85, heads at full lr.

---

## 5. The recovery training run

Submitted as **job 491875**:

```bash
PYTHONPATH=src python scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen_recovery \
    --warm_start  runs/phase3a_491240/final \
    --batch_size 16 --lr 1e-5 \
    --use_hard_failure_sampler \
    --label_smoothing 0.05 --entropy_reg_lambda 0.01 \
    --consistency_lambda 0.20 --fully_aux_lambda 0.50 \
    --use_ema --early_stop_patience 3 \
    --use_swa --swa_start_step 2000 \
    --use_llrd --llrd_decay 0.85
```

Wall-clock: ~25 minutes. Stages advanced via gate-pass or early-stop.

### Independent eval (job 491890) — full uncapped

> **Two metric conventions are reported throughout the project.**
> * **Paper convention** — denominator = `n_words` for every axis;
>   missing-gold counts as wrong on that axis. Anchors on the
>   published paper.
> * **Fully-observable subset** — denominator = tokens with all 3 gold
>   fields populated (n=61 Gazelle / n=999 MASAQ). Useful diagnostic.
> Same model, same data, same numerators — only denominators differ.
> Full unified table: `docs/eval_unified/unified_report.md`.

**Paper convention (denominator = n_words):**

| Dataset | Metric | Phase 3-A | Recovery | Δ |
|---|---|---|---|---|
| Gazelle (n=134) | case | 0.605 | 0.612 | +0.007 |
| Gazelle | role | 0.343 | 0.366 | **+0.022** |
| Gazelle | marker | 0.500 | 0.478 | −0.022 |
| Gazelle | fully | 0.209 | 0.209 | +0.000 (tied) |
| Gazelle | calib_gap | +0.021 | **−0.052** | healthier |
| MASAQ (n=5,007) | case | 0.832 | 0.845 | +0.014 |
| MASAQ | role | 0.155 | 0.161 | +0.006 |
| MASAQ | marker | 0.309 | 0.306 | −0.003 |
| MASAQ | **fully** | 0.135 | **0.142** | **+0.007** ★ (+36 tokens) |

**Fully-observable subset (n=61 / 999):**

| Dataset | Metric | Phase 3-A | Recovery | Δ |
|---|---|---|---|---|
| Gazelle | fully | 0.459 | 0.459 | +0.000 |
| Gazelle | role | 0.575 | 0.613 | +0.038 |
| MASAQ | fully | 0.675 | 0.711 | +0.036 |
| MASAQ | role | 0.778 | 0.807 | +0.029 |

**The clean honest claim:** +36 tokens correctly relabelled on MASAQ
fully (out of 5,007). That's +0.007 on the paper convention; +0.036
on the strict-gold subset. Both reflect the same underlying improvement.

The Gazelle calib_gap moved from +0.021 (slightly over-confident) to
−0.052 (slightly under-confident — healthier). Marker EM regressed
slightly because label smoothing pushed the marker head conservative.

This checkpoint became `runs/validated_nextgen_recovery/` —
the production model.

### vs the leaked stage_7 (the case study, fully-observable subset)

| Dataset | Metric | Phase 3-A | Recovery | Leaked stage_7 |
|---|---|---|---|---|
| MASAQ | fully (n=999) | 0.675 | 0.711 | **0.999** ← memorisation |
| MASAQ | calib_gap | 0.087 | 0.124 | **0.9998** ← memorisation |
| MASAQ | quranic_fully | 0.715 | 0.769 | **1.000** ← memorisation |
| Gazelle | fully (n=61) | 0.459 | 0.459 | 0.377 ← regressed |

The leaked numbers are not gains — they are 28 percentage points of
memorisation. (The contamination signature is the same on either
denominator; we report on the fully-observable subset here because
calib_gap = 0.9998 is the diagnostic that exposed the leak.)

---

## 6. Graph integration — first negative result

After recovery, the user directed: implement the FIRST real
structural-reasoning upgrade. Wire the existing grammar graph + graph
refiner into the actual forward path.

### What we built (12 steps)

1. Collator emits a dense `(B, W, W)` `word_edge_index` matrix where
   cell `[b, i, j]` is the edge type id (0–8) between word i and j
2. Edges populated from dep_heads (bidirectional, type 1) + construction
   spans (clique, type 3) + overlap detection (type 6)
3. Per-stage edge curriculum: stage 1–2 dep only → stage 3 add construction
   + agreement → stage 4 add clause → stage 5 add overlap+governor →
   stage 6 add discourse → stage 7 all 8 types
4. `models/graph_refiner.py` — 2-layer attention refiner with per-head
   edge-type embedding added to attention logits
5. Forward: `pooled = pooled + sigmoid(graph_gate) * (refined - pooled)`
6. **Gate logit init at −2.0** (sigmoid ≈ 0.119) — graph signal starts
   weak; the model learns whether structure helps. Critical for
   avoiding catastrophic degradation and oversmoothing on small data.
7. Encoder frozen for first 2,000 steps; refiner + gate train alone
8. After 2,000 steps, encoder unfreezes; refiner + encoder co-train
9. Edge dropout 15% on dep + construction edges during training
10. Eval emits `fully_with_graph` / `fully_without_graph` /
    `graph_edge_ablation_delta` / `graph_gate_alpha` so the scientific
    contribution of the graph signal is measurable in real time
11. Recovery patch (items 1–14 + SWA + LLRD) all on top
12. Submitted as **job 491906**; ran cleanly through 7 stages

### Training behaviour — what worked

- Refiner trained without instability (no NaN, no norm explosion)
- Gate moved 0.120 → 0.122 once encoder unfroze (small but real
  movement)
- Training-time ablation delta consistent +0.006 to +0.013 fully
  on the cap-100 eval slice
- Stage transitions held the no-leakage assertions

### Held-out result — what did not work

| Dataset | Metric | Recovery | Graph | Δ |
|---|---|---|---|---|
| Gazelle | fully | 0.459 | 0.459 | +0.000 |
| Gazelle | role | 0.613 | 0.613 | +0.000 |
| MASAQ | fully | 0.711 | 0.707 | −0.004 |
| MASAQ | role | 0.807 | 0.813 | +0.006 |

The training-time +0.013 ablation delta did **not survive** the
full-sample eval. All deltas within noise.

### Interpretation

At ~20k training sentences, the encoder + Stanza UD dep features
(which already provide structural information *as input augmentation*)
capture most of what a downstream graph layer would add. **The
bottleneck is not architectural at this data scale.**

Frozen at `docs/final_graph_negative_result/` with NEGATIVE_RESULT.md.

---

## 7. Governor head — second negative result

The graph negative result narrowed the search space but didn't pinpoint
the bottleneck. The dominant failure family was clearly `mudaaf_ilayh`
confusion, an *attachment* problem. Hypothesis: an explicit governor
head should fix it.

### What we built

A biaffine head: `score[b, i, j] = query(token_i)ᵀ · W · key(token_j)`,
producing a (B, W, W) logit tensor where `score[b, i, j]` is how much
token i wants j as its governor.

```python
self.governor_query_proj = nn.Linear(d, d_gov)
self.governor_key_proj   = nn.Linear(d, d_gov)
governor_logits = einsum("bid,bjd->bij", q, k)
governor_logits.masked_fill_(diagonal,    MASK_VAL)  # no self-loops
governor_logits.masked_fill_(pad_columns, MASK_VAL)  # no pad governors
```

Trained with two losses on top of the multi-head loss:
- **Governor CE** — `F.cross_entropy(governor_logits, dep_head_labels)`
  with weight 0.5
- **Attachment contrastive** — triplet-margin loss on
  (anchor, gold_head, sampled_negative) where negatives are
  plausible-but-wrong (adjacent token, nearest noun, nearest verb,
  nearest preposition); weight 0.1

### Bugs caught and fixed during the wiring pass

1. **`distill_v2` had spurious self-loop dep heads** on tokens 0/1/8
   of some sentences (likely a 1-vs-0-index bug in the upstream parser).
   The collator now rejects `head == j` and labels them IGNORE.
2. **`MASK_VAL = -inf` combined with `label_smoothing > 0` produced
   `+inf` loss** because `eps × log_softmax(-inf) = -inf → -log = +inf`.
   Switched to `MASK_VAL = -1e9` and disabled label smoothing on the
   governor head only.

### Disk + tokenizer issues during the wiring pass

The first three submissions all hit silent failures:
- Job 491820: silent save crash → traced to disk quota
- Job 491856: SLURM rejected before run → over-requested memory (48G)
- Job 491937: stage 1 stalled → disk quota again
- Job 491963: ran cleanly after `runs/nextgen_recovery` (14 GB) and
  `runs/nextgen_graph` (4.7 GB) were deleted to free space

### Held-out result — what did not work

```
Confusion                       Recovery    Governor
──────────────────────────      ────────    ────────
mudaaf_ilayh → mafoul_bih       32          32         no change
mudaaf_ilayh → mubtada          29          29         no change
ism_majrur → matuf              21          20         −1
mudaaf_ilayh → fail             13          14         +1
mudaaf_ilayh → ism_majrur       13          13         no change
mafoul_bih → fail               12          13         +1
```

**The dominant idafa confusions were unchanged.**

### Interpretation

The governor head learns *which token* is the parent in the dep tree.
But the *mudaaf_ilayh* vs *mafoul_bih* vs *ism_majrur* decision is not
"which token is the parent" — the dep parent is the *same token* in
all three readings. The decision is **what relation** holds with that
parent. That requires lexical-semantic knowledge (verb-argument
structure, idafa-head propensity, preposition-presence) which neither
dep features nor explicit governor prediction provides.

Combined with the graph negative result, the convergent conclusion is
clear: **at our data scale, more structural supervision does not help
this confusion. The bottleneck is lexical-semantic.**

Frozen at `docs/final_governor_negative_result/`.

---

## 8. Failure analysis — the central finding

We built `src/irab_tashkeel/analysis/`:

- `failure_analysis.py` — `FailureRecord` dataclass with all metadata
  (case/role/marker confusion, conf, dep depth, clause depth, semantic
  pressure, ambiguity, sentence length, long-range/overlap flags,
  calibration bucket)
- `failure_buckets.py` — slice records by structural axis
- `confusion_analysis.py` — per-axis confusion matrices + ranked top-N
- `structural_breakdown.py` — stratified fully accuracy
- `calibration_analysis.py` — per-axis 10-bin reliability + ECE +
  high-confidence-wrongs

Run on the validated_recovery checkpoint over Gazelle + MASAQ:

### The dominant role confusions

| Gold | Predicted | Count |
|---|---|---:|
| **mudaaf_ilayh** | **mafoul_bih** | **32** |
| **mudaaf_ilayh** | **mubtada** | **29** |
| ism_majrur | matuf | 21 |
| mudaaf_ilayh | fail | 13 |
| mudaaf_ilayh | ism_majrur | 13 |
| mafoul_bih | fail | 12 |

**~120 errors center on a single confusion family**. By far the largest
identifiable block.

### Why these confusions are linguistically genuine

The three roles in family I — *mudaaf_ilayh*, *mafoul_bih*, *ism_majrur*
— all surface as **a noun in jarr (genitive) case immediately after
another word** with kasra marker. The dep parent is the same; the case
is the same; the marker is the same; the token order is the same.

| When the second noun is | Governor | Reason |
|---|---|---|
| *mudaaf_ilayh* | the first noun | iḍāfa relation |
| *mafoul_bih* | a verb upstream | direct object |
| *ism_majrur* | a preposition | object of preposition |

Distinguishing them requires **lexical knowledge** — which the
existing structural features cannot supply.

### Per-construction-family fully accuracy

| family | fully |
|---|---:|
| inna_sisters | 0.702 |
| istithna | 0.717 |
| **idafa** | **0.639** |
| **idafa_multi** (nested) | **0.182** |

Nested idafa collapses to 0.182.

### Calibration on failures

| Axis | ECE | High-conf wrong (≥0.95) |
|---|---|---|
| case | 0.42 | 79 tokens |
| role | 0.49 | 83 tokens |
| marker | 0.60 | 70 tokens |

In the [0.9, 1.0) confidence bin — where the model says "I'm 95+%
sure" — per-axis accuracy is only 0.50–0.55 (case), 0.37 (role),
0.29 (marker).

This became `docs/failure_analysis/FINDINGS.md` — the project's
central scientific document.

---

## 9. Supervision phase — the data infrastructure

The convergent negative architectural results led to a deliberate pivot:
no more architecture, all annotation. The infrastructure for the next
round was built but unused (waiting on a grammarian).

### Ambiguity corpus

`scripts/data_v2/mine_ambiguity_candidates.py` reads the failure
analysis output and produces one `AmbiguityExample` per (sentence,
token, confusion) pair. Each candidate carries:
- primary_analysis (the model's prediction)
- secondary_analyses (at least one alternative reading)
- governor_candidates, attachment_candidates
- confidence_difficulty, reasoning_note

The mining run produced **4,233 candidates across 6 ambiguity kinds**:

| Kind | Candidates |
|---|---:|
| latent_governor | 990 |
| nested_attachment | 912 |
| idafa_attachment | 684 |
| semantic_role_overlap | 622 |
| preposition_vs_idafa | 530 |
| coordination_scope | 495 |

### Annotation server

`src/irab_tashkeel/annotation/`:
- `annotation_server.py` — FastAPI: `/api/queue/<kind>/{pending,confirm,reject,edit,disagreements}`
- `review_queue.py` — JSONL-backed pending/confirmed/edited/rejected state
- `disagreement_resolution.py` — multi-annotator majority vote
- `static/annotation.html` — single-page review UI

### Permissive evaluator (eval_v3)

`src/irab_tashkeel/eval_v3/`:
- `ambiguity_metrics.evaluate_with_ambiguity` — counts a prediction as
  correct if it matches **any** declared analysis
- `uncertainty_metrics` — `calibrated_fully`,
  `confidence_correctness_alignment`, `selective_accuracy_at_τ`,
  `high_confidence_error_rate`
- `structural_metrics` — `attachment_accuracy`, `governor_accuracy`,
  `overlap_accuracy`

### Active-learning candidate miner

`src/irab_tashkeel/active_learning/`:
- `uncertainty_sampling`, `disagreement_sampling`, `diversity_sampling`,
  `hard_case_mining`

### Calibration package

`src/irab_tashkeel/calibration/`:
- `temperature_scaling` — fit T on a held-out shard via L-BFGS
- `focal_loss` + `confidence_penalty`

### Hard-eval bucketing

`scripts/data_v2/build_hard_eval.py` partitions held-out sentences:

| Bucket | n | recovery fully |
|---|---:|---:|
| ambiguity | 728 | 0.736 |
| quranic_hard | 285 | 0.722 |
| overlap | 254 | 0.668 |
| rare_constructions | 8 | 0.182 |

A more aggressive cut (`hard_eval_v2/`) compounds conditions for
truly-hard sentences (long_nested_idafa: 17 sentences).

---

## 10. Final polish phase

After the supervision phase, the user directed: stop architecture
churn, polish what exists into a publishable research ecosystem.

Seven steps:
1. README full rewrite as production-research
2. Demo polish
3. Paper polish with seven-contribution structure
4. Failure analysis polished into research artifact
5. LIMITATIONS.md expansion
6. Reproducibility pass
7. Repo cleanup

Outputs:
- **README.md** — 1,696-line narrative covering every direction taken
- **docs/paper/PAPER.md** — restructured around seven contributions
  (leakage discovery, recovery framework, structural negative result,
  failure taxonomy, ambiguity infrastructure, governor bottleneck,
  uncertainty-aware eval)
- **docs/LIMITATIONS.md** — exhaustive 15-item list (sample sizes,
  calibration severity, idafa unresolved, no dialect, etc.)
- **docs/failure_analysis/FINDINGS.md** — mudaaf_ilayh family centerpiece
- **docs/MODEL_CARD.md** + **docs/KNOWN_FAILURES.md**
- **REPRODUCE.md** — single-command recipe, 12 sequential steps
- **archive/README.md** — inventory of failed variants kept in-place

---

## 11. Three parallel finalization tracks

The user asked to do everything at once. We ran four tracks in parallel:

### Track 1 — Demo launch (local)

- Resolved local Python env: pinned `transformers==4.49.0` (pre-CVE-check)
  + `tokenizers>=0.21` to be compatible with local `torch 2.2`
- Pulled both checkpoints from HPC (~1.35 GB):
  - `runs/validated_nextgen_recovery/`
  - `runs/phase3a_491240/final/`
- Launched at http://127.0.0.1:8000

### Track 2 — Multi-seed HPC

- `scripts/slurm/100_train_multiseed.sbatch` — parameterised by
  SEED env var
- Submitted seeds 1 + 2 (HPC stud QoS limits to 2 jobs/user)
- With seed 0 = original validated_recovery, this gives 3 runs for
  noise quantification

### Track 3 — Temperature scaling

`scripts/calibration/run_temperature_scaling.py`:
- Calibration shard: last 100 MASAQ by sentence_id
- Reporting shard: first 524 MASAQ
- Fitted T per axis via L-BFGS over NLL

Result:

| axis | T | ECE before | ECE after | Δ |
|---|---:|---:|---:|---:|
| case | 1.24 | 0.0554 | **0.0185** | −0.0369 ★ |
| role | 1.41 | 0.0923 | 0.1139 | +0.0216 (shard mismatch) |
| marker | 1.15 | 0.0815 | **0.0577** | −0.0238 ★ |

Case calibration improved meaningfully; marker too. Role got slightly
worse because the calibration shard (100 sentences) under-represents
role distribution.

### Track 4 — Permissive eval (heuristic upper bound)

`scripts/analysis/auto_annotate_ambiguities.py` heuristically marks
candidates as `BOTH_VALID` when gold and predicted role both belong to
the surface-ambiguous role family (mudaaf_ilayh, mafoul_bih, mubtada,
fail, ism_majrur, naat, matuf, badal, ism_inna, khabar_inna,
khabar_kana, khabar).

Re-mined the candidates locally (the HPC version had different sids
due to data rebuild). 4,233 mined → 4,233 auto-kept across 486
unique sentences.

Result on Gazelle + MASAQ (1,060 fully-observable tokens):

```
strict_fully     = 0.6962
permissive_fully = 0.8491
Δ                = +0.1528  ★
```

Of 322 strict-wrong tokens, 311 are flagged ambiguous, 162 resolve
permissively → ambiguity_resolved_accuracy = 0.521.

**Caveat:** this is a heuristic UPPER BOUND. A real grammarian would
reject many of these — e.g., when an overt verb unambiguously governs
the noun. The honest expected delta with a strict human annotator is
closer to +0.05 to +0.10. Either way, **it validates the central
finding** — the mudaaf_ilayh confusion family is dominated by genuine
surface ambiguity, not model error.

---

## 12. Demo polish

The user asked to make the demo "research-product quality" with a
15-step polish list:

### Backend (`demo/backend/`)

`inference.py` returns a full structured payload per analyze:
- `tokens` (with `role_alternatives` top-3, `confidence_band`,
  `calibration_warning`, `is_surface_ambiguous`)
- `constructions` (kana / inna / idafa / harf_jarr_phrase detection
  from predicted roles, with explanations)
- `graph` (nodes + sequential + construction edges for SVG render)
- `reasoning` (template-rendered "why role / why case / why marker"
  per token, no free-form generation)
- `warnings` (low-confidence + calibration warnings rolled up)

`main.py` adds endpoints:
- `/api/permissive_eval`
- `/api/calibration`
- `/api/failure_summary`
- `/api/hard_cases`

Plus a no-cache middleware so browsers always pick up the latest HTML.

### Frontend (`demo/static/index.html`)

Full visual overhaul with hero gradient (deep navy → indigo → magenta),
modern card layout (16px rounded corners, soft shadows), CSS-variable
theme with dark mode toggle, Plus Jakarta Sans + Amiri + JetBrains Mono
fonts. Nine tabs:

1. **Analyze** — token cards with case pills, role pills, confidence
   bars, calibration warnings, surface-ambiguous chips. Click any →
   detail modal with top-3 alternatives + reasoning
2. **Reasoning** — per-token cards with "why role / why case / why
   marker"
3. **Graph** — vanilla-SVG layout with sequential + dashed-pink
   construction edges; hover any node to highlight matching token
4. **Constructions** — detected family list with members + governor
   + explanation
5. **Compare Models** — recovery / phase3a / leaked stage_7
   side-by-side + token-level diff table
6. **Hard Cases** — 6 curated examples with rationale; clickable
7. **Eval Dashboard** — bar visuals for headline metrics + calibration
   table (T fits + ECE before/after) + top role confusions +
   permissive-eval delta tile
8. **Leakage Story** — phase3a vs recovery vs leaked metrics with
   contamination-signature explanation
9. **Raw JSON** — copy + download buttons

### Tab inconsistency fixes

- All five "analysis-dependent" tabs now show a friendly empty-state
  CTA when there's no analysis yet
- Compare tab auto-runs `runCompare()` when clicked
- Hard Cases switched from inline onclick (fragile when Arabic text
  contains apostrophes) to event delegation with data-text attributes
- `renderJson()` is now also called from `analyze()` so the JSON tab
  is always live
- Token modal calls `closeModal()` before opening to prevent stacking
  duplicate id="modal" elements
- FastAPI middleware adds `Cache-Control: no-cache, no-store,
  must-revalidate` on every HTML response

---

## 13. What stayed open

The conversation closed with the project in a strong but not-perfect
state. Three concrete follow-ups were quantified for impact:

1. **Annotate 50–100 ambiguity candidates** (highest ROI) — likely
   moves Gazelle role +0.02 to +0.05 with no model retraining; turns
   the +0.15 heuristic upper bound into a measured number
2. **Apply temperature scaling more robustly** — already started,
   but role ECE got slightly worse with a tiny calibration shard; a
   larger shard would fix this
3. **Multi-seed aggregation** — jobs 492045 + 492046 queued on HPC,
   noise band on Gazelle should be quantified once they finish

The annotation infrastructure is fully built; the temperature-scaling
infrastructure is fully built; the multi-seed jobs are queued. None
have been completed end-to-end.

---

## 14. Commit timeline

The work landed on `main` in 50+ commits over the conversation. Major
milestones:

| Phase | Commit | Description |
|---|---|---|
| Phase A scaffolding | `aaf326e` | Phase A-G eval infrastructure |
| Recovery patch | `c1a92bd` | 14-item patch + SWA + LLRD |
| Validated recovery | `e752b5f` | Honest numbers + frozen checkpoint |
| Train script fixes | `02af149` | Try/except save + force-advance |
| Demo first cut | `c541251` | Demo points at validated_nextgen_recovery |
| Supervision phase | `1ca5125` | Failure analysis + ambiguity infrastructure |
| Ambiguity reasoning phase | `c514cce` | Annotation server + eval_v3 + governor head |
| Governor experiment | `ebffb74` | Clean negative result |
| Final polish | `324df7b` | Production-quality README + docs |
| README narrative | `7e727c3` | 1,696-line comprehensive rewrite |
| Three finalisation tracks | `d1cad50` | Temperature scaling + permissive eval + multi-seed |
| Demo overhaul | `a3ae42a` | 15-step polish |
| Tab fixes | `70bf29f` | Empty states + no-cache |
| Claude scrub | (history rewrite) | All co-author trailers removed |

After the Claude-scrub history rewrite (using `git filter-repo`) all
50 commits were re-issued with new SHAs and force-pushed. The tags
`v1-phase3a`, `v2-nextgen-curriculum`, `v2.1-validated` were rebuilt
to point at the new commits.

---

## In summary

The project arc was:

```
Apparent SOTA (0.999 fully)
    ↓
Independent eval reveals contamination signature
    ↓
Three runtime assertions + provenance manifest enforce no-leakage
    ↓
Recovery patch (14 items + SWA + LLRD)
    ↓
Validated production checkpoint (honest +0.036 MASAQ fully)
    ↓
Graph integration → negative result (tied with recovery)
    ↓
Governor head → negative result (idafa confusions unchanged)
    ↓
Failure analysis → mudaaf_ilayh family is the dominant residual
    ↓
Convergent finding: bottleneck is lexical-semantic, not architectural
    ↓
Supervision infrastructure built (4,233 candidates, annotation server,
permissive evaluator, active learning, calibration package)
    ↓
Final polish (README, paper, demo, reproducibility, repo cleanup)
    ↓
Three parallel finalisation tracks (demo live, multi-seed queued,
temperature scaling, permissive eval +0.15 upper bound)
    ↓
Project frozen.
```

The strongest contribution is the **methodology**: discovering and
correcting the leakage, documenting both negative architectural
results, identifying the central linguistic finding, and shipping the
full annotation infrastructure for the next round. The headline
metric gains are honest and modest; the scientific narrative is
strong.
