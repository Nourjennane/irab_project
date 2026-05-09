# Current State of the Repository (2026-05-09)

The experimentation phase is complete. This document is the authoritative
pointer for what is production, what is archival, and where to find each
piece.

## Production system

**Phase 3-A** — frozen since 2026-05-04. Architecture: AraT5v2-base
encoder (296M) + Phase 1 morphology supervision (7 morph heads) +
Phase 3 static UD dependency features (DEPREL + HEAD topology +
governor POS).

| Asset | Path |
|---|---|
| Trained checkpoint | `runs/phase3a_491240/final/` |
| Model class | `src/irab_tashkeel/morphology/dep_aware_model.py::DepAwareStructuredModel` |
| Predictor (production inference) | `src/irab_tashkeel/inference/structured_predictor.py::StructuredPredictor` |
| Schema (canonical labels) | `src/irab_tashkeel/structured/schema.py` |
| Evaluator (with kana fix) | `src/irab_tashkeel/evaluation/structural.py` |
| Per-construction eval harness | `scripts/structured/eval_per_construction.py` |
| Training data corpus | `data/morph_v1_dep/train.jsonl` |
| Training script | `scripts/slurm/73_train_phase3a.sbatch` |

## Final metrics on Phase 3-A (FIXED evaluator)

- **Gazelle (n=107):** case 72.0, role 40.2, marker 62.6, fully 25.2, calib_gap −0.101
- **MASAQ (n=5,007):** case 85.8, role 17.1, marker 33.0, fully 14.9, calib_gap +0.048
- **Per-construction tables:** see `docs/REPORT.md` §5.6.6 and `docs/paper/REPORT.tex`.

## Archival experiments (kept in repo for the case study)

These all ship as opt-in or unused; none are on the production path.

| Phase | Module / Path | Result |
|---|---|---|
| Phase 4a (taxonomy v4) | `src/irab_tashkeel/structured/taxonomy_v4.py` | substitutable with morph |
| Phase 2 (FiLM/additive conditioning) | `src/irab_tashkeel/morphology/conditioning.py` | joint-dynamics regression |
| Phase 5 (case hierarchy) | `enable_case_hierarchy` flag in `dep_aware_model.py` | flat / regress |
| Phase 6 (marker hierarchy) | `enable_marker_hierarchy` flag in `dep_aware_model.py` | flat / regress |
| Phase 3.1 (relational attention) | `src/irab_tashkeel/morphology/relational_reasoning.py` | flat / regress |
| Phase 39 synthetic augmentation | `scripts/augment/generate_rare_constructions.py`, `scripts/augment/build_augmented_corpus.py` | distribution mismatch on Gazelle |
| Phase R (retrieval + soft logit bias) | `src/irab_tashkeel/grammar_memory/retrieval_predictor.py` | partial; MASAQ-only |
| Phase R2 (structural reasoning) | `src/irab_tashkeel/grammar_memory/structural_predictor.py`, `structural_reasoner.py` | 0.0 everywhere with proper forward |

## Evaluator + diagnostic utilities

| Tool | Path | Purpose |
|---|---|---|
| Per-construction eval | `scripts/structured/eval_per_construction.py` | Breakdown by 7 construction families with case/role/marker/fully/calib |
| Kana failure trace dump | `scripts/structured/debug_kana_failures.py` | 15-field per-span dump for a checkpoint |
| Collateral diagnostic | `scripts/structured/diagnose_collateral.py` | Word-level prediction-flip analysis between two predictors |
| Grammar memory builder | `scripts/grammar_memory/build_memory.py` | Re-build the 7-family FAISS index if needed |

## Test suites

```
pytest tests/                          # all unit tests
pytest tests/test_structural_reasoner.py    # Phase R2 reasoner tests (11)
pytest tests/test_phase3_1_relational.py    # Phase 3.1 unit tests (7)
```

## Datasets

| Surface | n words | Path | Used as |
|---|---:|---|---|
| Gazelle iʿrāb | 134 | `src/irab_tashkeel/data/gazelle.py` | held-out MSA news eval |
| MASAQ subset | 5,007 | `data/masaq_eval.jsonl` | held-out Quranic eval |
| Distill v2 (Haiku) | ~77K | `data/morph_v1_dep/train.jsonl` | training corpus |
| UD Arabic-PADT | varies | merged into morph_v1_dep | morph supervision |
| Grammar memory | 18,839 | `data/grammar_memory/` | Phase R/R2 retrieval pool |

## Documents

- `docs/REPORT.md` — full research report (canonical narrative, latest tables)
- `docs/paper/REPORT.tex` — paper LaTeX source (mirror of REPORT.md)
- `docs/paper/REPORT.pdf` — rendered paper (rebuild with `xelatex REPORT.tex`)
- `docs/CURRENT_STATE.md` — this file
- `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/EVALUATION.md` — pre-existing module-level docs
- `docs/roadmap/` — design docs for Phase R, Phase R2, etc.

## Architecture is FROZEN

No further additions to:

- iʿrāb head families (case / role / marker / pos)
- Conditioning mechanisms (FiLM / additive / detached)
- Hierarchical decoder variants
- Output bias mechanisms
- CRF redesigns
- Relational attention variants
- Inference-side override / structural reasoning wrappers

The empirical evidence in `docs/REPORT.md` §5.6.6 is overwhelming
that this lever is exhausted at the current 296M / 6-epoch /
~7K-corpus envelope.

## Future work directions (data, scale, coverage)

See `docs/REPORT.md` §8. Summary:

1. Larger / richer corpora (Sonnet-distilled, CamelTB, Extended Quranic Treebank)
2. Richer annotation (preserve long-tail role surface forms)
3. Broader rare-construction coverage (~500 hand-annotated examples per failing construction)
4. Better dependency quality (drop alignment threshold, gold UD-PADT dep, inference-time Stanza)
5. Stronger pretrained Arabic encoders (CAMeLBERT-CA, AraBART)
6. Longer training schedules (12–18 epochs vs current 6)
7. Multi-task with diacritization (Sadeed-style)

## Reproducing the final tables

```bash
# Gazelle + MASAQ on Phase 3-A with the FIXED evaluator:
PYTHONPATH=src python scripts/structured/eval_per_construction.py \
    --model runs/phase3a_491240/final --eval gazelle \
    --out_dir runs/final/p3a_gazelle

PYTHONPATH=src python scripts/structured/eval_per_construction.py \
    --model runs/phase3a_491240/final --eval masaq \
    --out_dir runs/final/p3a_masaq
```

The R2 archival pipeline (negative result, 0.0 everywhere):

```bash
PYTHONPATH=src python scripts/structured/eval_per_construction.py \
    --model runs/phase3a_491240/final --eval gazelle \
    --use_structural_reasoning --retrieval_memory data/grammar_memory/ \
    --enabled_families kana_sisters \
    --out_dir runs/final/r2_archival_gazelle
```
