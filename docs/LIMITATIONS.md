# Limitations

This document is a deliberate record of what the validated nextgen
**recovery** model (`runs/validated_nextgen_recovery/`) does not do
well. It is mandatory reading before deploying or comparing.

## Modest absolute gains over Phase 3-A

The honest improvements are:

- MASAQ Quranic: fully +0.036 (0.675 → 0.711), role +0.029
- Gazelle: role +0.038, fully +0.000 (unchanged), marker −0.031

These are *real* but small. Phase 3-A was already a strong local optimum;
the curriculum + regularization patch produced incremental gains, not
a step change.

## Calibration: better on Gazelle, slightly worse on MASAQ

`calib_gap` (mean role-confidence on correct − mean on wrong) on
**Gazelle** improved from +0.021 (Phase 3-A) to −0.052 (recovery) — the
model is now slightly *under*-confident on correct predictions, which
is healthier than the overconfidence we observed in the leaked variant
(calib gap = 0.999 — a memorization artefact).

On **MASAQ**, calib_gap rose modestly (0.087 → 0.124). ECE remained in
the 0.10–0.13 range across training, an order of magnitude better than
the leaked stage_7's 0.218.

## Held-out sample sizes

The two clean held-out sources are small:

- Gazelle: 30 sentences / ~290 tokens
- MASAQ Quranic: 624 sentences / ~5–6k tokens (Quranic only)

Confidence intervals on Gazelle metrics are wide. The MASAQ sample is
domain-restricted to Quranic Arabic; modern dialect performance is not
measured.

## UD-PADT contamination

`ud_padt_test` shares 17–21 sentences with `distill_v2` (the bulk
training source). Numbers reported on UD-PADT are for completeness, not
as headline metrics. See [`docs/leakage_audit/leakage_report.md`](leakage_audit/leakage_report.md).

## Reasoning trace is template-based

[`src/irab_tashkeel/reasoning/`](../src/irab_tashkeel/reasoning/) renders
explanations from structured labels using deterministic templates. It
**does not generate free-form prose**, by design — generative reasoning
would re-introduce hallucination risk.

Consequence: explanations exist only for constructions defined in
`data_v2/constructions/`. Sentences with rare or novel constructions get
a degraded fallback (just labels, no narrative).

## Construction-detection F1 is 0 in eval

The training-time eval did not emit `ConstructionPrediction` records, so
`construction_f1_macro` reads 0 throughout. This is an evaluator gap,
not a model failure — the model uses construction features internally
and benefits from them. Fixing the evaluator to surface construction
predictions is on the roadmap.

## No cross-dialect validation

All training and eval data are MSA or classical Arabic. Egyptian, Gulf,
Maghrebi, etc. are not represented. Performance on dialect input is
undefined.

## Single-encoder architecture

The model uses one shared AraT5v2-base encoder. We did not run a full
backbone benchmark before validating; the registry in
[`src/irab_tashkeel/backbones/registry.py`](../src/irab_tashkeel/backbones/registry.py)
exists for future comparison work but is not yet exercised.

## Compute

Training was performed on a single NVIDIA RTX 4060 (HPC node) over ~4
hours for 45 000 curriculum steps. We have no large-scale ablation budget;
single-seed numbers only.

## Eval slice cap was used during training

The training-time gate evaluator was capped at 100 sentences per eval
call (`--eval_max_sentences 100`). This cap is **why** the dramatic
gates fired so easily — quranic_fully=1.0 is on a 100-sample slice, not
the full MASAQ. The Phase A independent evaluator runs on the full sets
to recover the honest numbers.
