# Phase 3-A — Frozen Validated Baseline

**Status:** IMMUTABLE. This directory pins the validated scientific
baseline of the project as of 2026-05-09. It is referenced by the
next-generation branch (`nextgen-grammatical-reasoning`) and must
not be modified.

## What this represents

Phase 3-A is the final production system from the original research
cycle. It is:

- AraT5v2-base encoder (296M parameters)
- Phase 1 morphology supervision (7 morph heads)
- Phase 3 static UD dependency features (DEPREL + HEAD topology + governor POS)

Trained for 6 epochs on a 77K-row Haiku-distilled corpus. Frozen
since 2026-05-04. The full empirical case study (10 phases tested,
2 working, 8 plateau) is in `docs/REPORT.md` §5.6 and in
`docs/final_phase3a_summary.md`.

## Pointers

| Asset | Location |
|---|---|
| Trained checkpoint | `runs/phase3a_491240/final/` (~1.2 GB; on Bocconi HPC + reproducible from training data) |
| Model class | `src/irab_tashkeel/morphology/dep_aware_model.py::DepAwareStructuredModel` |
| Predictor (production inference) | `src/irab_tashkeel/inference/structured_predictor.py::StructuredPredictor` |
| Schema (canonical labels) | `src/irab_tashkeel/structured/schema.py` |
| Evaluator (with kana fix) | `src/irab_tashkeel/evaluation/structural.py` |
| Per-construction eval | `scripts/structured/eval_per_construction.py` |
| Training data | `data/morph_v1_dep/train.jsonl` |
| Training script | `scripts/slurm/73_train_phase3a.sbatch` |

## Final metrics on Phase 3-A (FIXED evaluator)

**Gazelle (n=107):** case 72.0, role 40.2, marker 62.6, fully 25.2,
calib_gap −0.101.

**MASAQ (n=5,007):** case 85.8, role 17.1, marker 33.0, fully 14.9,
calib_gap +0.048.

Per-construction tables: `docs/final_phase3a_summary.md`.

## Reproducing the final tables

```bash
PYTHONPATH=src python scripts/structured/eval_per_construction.py \
    --model runs/phase3a_491240/final --eval gazelle \
    --out_dir runs/final/p3a_gazelle

PYTHONPATH=src python scripts/structured/eval_per_construction.py \
    --model runs/phase3a_491240/final --eval masaq \
    --out_dir runs/final/p3a_masaq
```

## Policy

**Do not modify:** the model class, the schema, the evaluator, or the
training recipe in this branch (`nextgen-grammatical-reasoning`).
Future work happens in the new modules (`src/irab_tashkeel/{constructions,
grammar_graph, long_context, curriculum, decoding, reasoning,
semantic, discourse, retrieval_v2, eval_v2}/`) and operates on the
new data engine (`data_v2/`). The frozen baseline is preserved as
the validated comparison point and as the reproducible research
contribution.
