# Limitations

This document is a deliberate, exhaustive record of what the validated
nextgen-recovery model (`runs/validated_nextgen_recovery/`) does not
do well, and where the project's current evidence is genuinely thin.
We list everything we are aware of. Anything missing here is by
oversight, not by design.

## 0 · Two metric conventions — both reported, primary is the paper's

The project reports every metric under **two denominator conventions**:

- **Paper convention** (primary headline) — denominator = `n_words` for
  every axis; tokens with missing gold count as wrong on that axis.
  This is the metric the published paper uses.
- **Fully-observable subset** (secondary diagnostic) — denominator =
  tokens where all 3 gold fields (case, role, marker) are populated.
  This is the metric `eval_v2` computes by default.

The numerators (count of correct tokens on each axis) are identical
across both conventions. Only the denominators differ. We use the
paper convention for headlines because it's the metric the published
paper anchors on; reporting only the fully-observable subset would
be implicit denominator-shopping. Full unified report:
[`docs/eval_unified/unified_report.md`](eval_unified/unified_report.md).


## 1 · Held-out sample sizes are small

- **Gazelle** (the cleanest MSA gold) has only **30 sentences / 134
  words / 61 fully-observable tokens.** Differences ≤ 0.05 fully
  on Gazelle are inside the noise band. We do not draw conclusions
  from Gazelle deltas alone.
- **MASAQ Quranic** is larger (624 sentences / 5,007 words / 999
  fully-observable tokens) but is restricted to Quranic Arabic,
  not modern Arabic.
- The held-out totality is **1,334 sentences across all sources**.

## 2 · UD-PADT-test contamination

`ud_padt_test` shares **17 exact** and **21 normalised** sentence
duplicates with `distill_v2`, plus 16/16 with `ud_padt_train`. We
report UD-PADT numbers for completeness only; they cannot be used
as a clean held-out signal. Detection methodology and tables in
[`leakage_audit/leakage_report.md`](leakage_audit/leakage_report.md).

## 3 · Calibration is severe

ECE on failures (validated_recovery): case **0.42**, role **0.49**,
marker **0.60**. The model is confidently wrong far too often.
Specifically, in the [0.9, 1.0) confidence bin (where the model says
"I'm 90 %+ sure"), per-axis accuracy is only ~0.5–0.55 on case,
~0.37 on role, and ~0.29 on marker.

Mitigation infrastructure exists but is not yet applied:

- `src/irab_tashkeel/calibration/temperature_scaling.py` — post-hoc T fit
- `src/irab_tashkeel/calibration/focal_loss.py` — training-time focal loss + confidence penalty

A held-out shard would need to be carved out before applying T-scaling
to avoid recalibrating on the test set.

## 4 · No cross-dialect coverage

All training and evaluation data is MSA or classical/Quranic Arabic.
Performance on dialects (Egyptian, Levantine, Gulf, Maghrebi) is
**undefined**. Assume it is bad. Do not deploy on dialect input
without a separate evaluation step.

## 5 · Idafa-attachment confusion is unresolved

The dominant residual failure family is the
*mudaaf_ilayh* ↔ *mafoul_bih* ↔ *ism_majrur* confusion — three roles
with near-identical surface signature. The graph and governor
experiments **did not** displace this confusion; structural
supervision alone cannot resolve it.

The expected resolution path is **lexical-semantic supervision**
(verb-argument structures) plus **alternative-analysis annotations**
(genuinely ambiguous tokens marked permissively). The infrastructure
for both is in the repo; the annotation work is pending.

## 6 · Reasoning trace is template-based

[`src/irab_tashkeel/reasoning/`](../src/irab_tashkeel/reasoning) renders
explanations from structured labels using deterministic templates.
This is a deliberate design choice (it prevents hallucination), but
sentences with rare or novel constructions get a degraded fallback
(labels only, no narrative). Free-form Arabic-prose explanation is
explicitly **out of scope**.

## 7 · Construction-detection F1 is not measured

The training-time evaluator does not emit `ConstructionPrediction`
records, so `construction_f1_macro` reads 0 throughout training.
This is an evaluator gap, not a model failure — the model uses
construction features and benefits from them — but it means we
cannot directly measure how well construction membership is
predicted. Fixing the evaluator to surface construction predictions
is on the roadmap.

## 8 · Single-seed numbers throughout

We have not run multi-seed ablations. All headline numbers are
single-run results. Differences ≤ 0.005 on MASAQ fully or ≤ 0.02
on Gazelle fully are within plausible single-seed noise and should
not be over-interpreted.

## 9 · Long-sentence failures

Performance degrades on sentences past ~40 tokens. The
DepAwareStructuredModel's word-pooling is fixed at training-time
configuration; beyond ~40 tokens the encoder's positional information
begins to wash out. For production, segment longer inputs.

## 10 · Negative architectural results

Two architectural extensions were tried and **did not** improve over
validated_recovery:

- **Graph integration** ([`final_graph_negative_result/`](final_graph_negative_result)):
  gated 2-layer graph refiner over word states. Tied with recovery on
  all clean held-out metrics (within ±0.005).
- **Governor head** ([`final_governor_negative_result/`](final_governor_negative_result)):
  biaffine head + 0.1×attachment-contrastive triplet loss. Tied with
  recovery on the headline; idafa-confusion family unchanged.

Both are documented as clean reproducible negative results, not as
failures. They constrain the search space.

## 11 · Annotated semantic-ambiguity layer is empty

The `AmbiguityExample` schema and the 4,233 auto-mined candidate
ambiguities exist (`data_v2/ambiguity_corpus/`), but **no human
annotator has confirmed any of them yet**. Until that work is done:

- Permissive scoring via `eval_v3.evaluate_with_ambiguity` will return
  the same as strict scoring (no `secondary_analyses` to consult).
- The annotation server (`src/irab_tashkeel/annotation/annotation_server.py`)
  is deployable but unused.

## 12 · No discourse-level reasoning

The model operates on single sentences. Coreference, topic continuation,
rhetorical relations across sentence boundaries are not modelled.
Sentences whose iʿrāb genuinely depends on prior context (omitted
subject in Quranic verses, e.g.) are out of distribution.

## 13 · No public Arabic LLM-scale pretraining

The encoder is `UBC-NLP/AraT5v2-base-1024` — a moderate-scale Arabic
T5. We did not pretrain a larger encoder on a wider Arabic corpus.
A larger Arabic foundation model would likely move the ceiling of
this work, but it is a separate project.

## 14 · Compute and data scale are modest

Training was performed on a single GPU (Bocconi HPC stud QoS) over
~25–45 minutes per run. The full training corpus is ~18,366
non-test sentences. Both compute and data are modest by current
standards. Strong claims about general Arabic grammatical reasoning
should not be drawn from this scale.

## 15 · Eval-time fully-observable subsets are tiny

For the most-honest "fully-observable" headline numbers:

- Gazelle fully-observable: **61 tokens**
- MASAQ fully-observable: **999 tokens**

This means a single misclassified token shifts Gazelle fully by
**1.6 percentage points**. Headline deltas under 0.03 on Gazelle
should be treated as informative trend signal, not significant
result.
