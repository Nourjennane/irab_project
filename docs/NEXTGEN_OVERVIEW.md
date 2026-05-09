# Next-Gen Branch Overview

**Branch:** `nextgen-grammatical-reasoning`
**Frozen baseline:** `main` (commit `31319af`) and `frozen_baselines/phase3a/`
**Date branch opened:** 2026-05-09

This document is the entry point for the next-generation phase of
the project. The frozen baseline (Phase 3-A) remains the validated
scientific contribution; the next-gen branch builds a hybrid
Arabic grammatical reasoning system on top of it without
re-litigating the frozen findings.

## Project philosophy (mandatory for this branch)

**Established by the frozen baseline:**
- orthogonal linguistic information unlocks gain
- morphology + dependency supervision are the two validated levers
- evaluator correctness adds real measurement value

**Established as plateau or regress:**
- conditioning variants
- decoder hierarchy
- retrieval wrappers
- shallow symbolic overrides
- inference-side rearrangement

**This branch focuses on:**
- scale (parameters, corpus, sequence length, epochs)
- supervision (multi-layer: morph / syntax / role / reasoning)
- syntactic depth and clause hierarchy
- semantic interpretation
- discourse-aware reasoning
- ambiguity handling and long-range dependencies

**This branch refuses:**
- FiLM variants
- hierarchy tricks
- tiny decoder tweaks
- logit hacks
- shallow retrieval voting
- small synthetic template loops
- architecture churn without new information

## Repository layout (next-gen additions)

```
frozen_baselines/phase3a/      — immutable pointers to the validated baseline
archive/                       — future archival of failed next-gen variants

data_v2/                       — new multi-layer supervision
  raw/  processed/  annotated/  reasoning/
  treebanks/  quranic/  classical/  msa_news/  educational/  discourse/

src/irab_tashkeel/
  constructions/               — Step 3: first-class construction objects
  grammar_graph/               — Step 4: unified graph engine
  long_context/                — Step 5: multi-hop reasoning
  curriculum/                  — Step 7: staged learning
  decoding/                    — Step 8: structured decoding
  reasoning/                   — Step 9: explanation supervision
  semantic/                    — Step 10: semantic reasoning
  discourse/                   — Step 11: discourse reasoning
  retrieval_v2/                — Step 12: structure-aware retrieval
  eval_v2/                     — Step 13: clause / construction / reasoning eval

docs/
  CURRENT_STATE.md             — frozen-baseline production pointer (kept as-is)
  NEXTGEN_OVERVIEW.md          — this file
  final_phase3a_summary.md     — frozen-baseline summary (one page)
  error_taxonomy.md            — Step 16: drives all next-gen development
  roadmap/
    nextgen_data_engine.md     — Step 1
    backbone_upgrade.md        — Step 6
    training_scale_plan.md     — Step 14
  research_logs/               — Step 15: pre-registered experiment logs
  ablations_v2/                — Step 13 outputs
  error_analysis_v2/           — tagged error logs
  failure_modes_v2/            — characterised failure-mode catalogue
```

## Implementation order (mandatory; from the directive)

Per Step 18 of the directive, this is the only order in which
work proceeds. Skipping ahead is not allowed.

1. ✓ Repo restructuring (this commit)
2. ✓ Frozen baseline archive (this commit)
3. Data infrastructure — Step 1
4. Construction schemas — Step 3
5. Grammar graph representation — Step 4
6. Error taxonomy population — Step 16 (initial population pass)
7. Backbone comparison matrix — Step 6
8. Curriculum framework — Step 7
9. Eval v2 framework — Step 13
10. Reasoning supervision schema — Step 9

**Only after these:** begin next-generation training.

## Engineering standards

- Modular architecture with clean separation between production /
  experimental / archival / evaluation / reasoning / retrieval /
  graph layers.
- Every experiment carries a research-log entry (Step 15) with
  hypothesis, mechanism, decision rule, and verdict. Pre-registered.
- Reproducibility: per-experiment configs, deterministic eval,
  full trace logging.
- Every experiment's failure mode is tagged in
  `docs/error_taxonomy.md` and characterised in
  `docs/failure_modes_v2/`.

## Comparison contract with the frozen baseline

Every next-gen experiment reports (at minimum):

- Gazelle + MASAQ overall: case / role / marker / fully / calib_gap
- Gazelle + MASAQ per-construction: same axes for kana / inna /
  istithna / idafa / idafa_multi / quranic_proxy
- Delta vs Phase 3-A on the corrected evaluator (numbers in
  `docs/final_phase3a_summary.md`)
- Per-error-category breakdown via `docs/error_taxonomy.md`

The decision rule for "ship as new production" is pre-registered:
no replacement of the production checkpoint without ≥+1.0 fully
on at least one surface AND no regression > 1.0 fully on the other
AND no regression on the per-construction calibration metrics.

## Pointers

- `docs/final_phase3a_summary.md` — the frozen baseline in one page
- `docs/REPORT.md` — full empirical case study (frozen)
- `docs/CURRENT_STATE.md` — production paths
- `docs/roadmap/*.md` — next-gen design docs
- `docs/error_taxonomy.md` — drives all future work
- `docs/research_logs/` — experiment registry (pre-registered, immutable)

The first concrete next-gen task is the **error-taxonomy
population pass** (Step 16): walk the frozen-baseline Gazelle +
MASAQ errors, tag each with the 10 categories, produce the
histogram that prioritises modules. That is the gating activity
before data-engine implementation begins.
