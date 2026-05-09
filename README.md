# Arabic Iʿrāb — Honest Grammatical Reasoning at Scale

> A research system for per-token Arabic grammatical analysis (إعراب)
> built around honest evaluation, leakage prevention, calibration,
> and documented negative results.

[![status](https://img.shields.io/badge/status-validated-green)]()
[![checkpoint](https://img.shields.io/badge/production-validated__nextgen__recovery-blue)]()
[![license](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)]()

---

## 1 · Overview

This repository turns a raw Arabic sentence into a per-token grammatical
analysis covering case, syntactic role, marker (the visible diacritic),
morphology, and construction membership. It targets MSA and Quranic
Arabic at sentence level.

**Production checkpoint:** [`runs/validated_nextgen_recovery/`](runs/validated_nextgen_recovery)
— curriculum-trained, leak-free, frozen with reproducibility manifest.

**Two documented negative results sit alongside it:**

- [`docs/final_graph_negative_result/`](docs/final_graph_negative_result) — gated graph refiner did not exceed the validated baseline at our data scale
- [`docs/final_governor_negative_result/`](docs/final_governor_negative_result) — biaffine governor head + attachment contrastive trained correctly but did not displace the dominant idafa-attachment confusion

These are kept on purpose. They constrain the search space for future
work and demonstrate that the project's remaining bottleneck is
**lexical-semantic supervision**, not architecture.

## 2 · Motivation

Existing Arabic NLP systems either (a) predict diacritics without
exposing reasoning, or (b) emit lengthy free-form prose that is hard
to score and easy to hallucinate. Neither directly serves educational
or auditable applications.

We instead emit **structured per-token grammatical analysis**:
machine-checkable, calibrated, renderable into pedagogically useful
prose by a deterministic template renderer.

## 3 · What is Arabic iʿrāb?

Iʿrāb assigns to each word in a sentence:

- a **case** (raf / nasb / jarr / jazm / mabni)
- a **syntactic role** (mubtada, fail, mafoul_bih, ism_kana, …)
- a **marker** — the actual diacritic that signals the case
- a **construction membership** (kana sisters, inna sisters, idafa, …)

The chain is causal: the construction governs the role, which governs
the case, which selects the marker.

## 4 · Why the problem is hard

- Reasoning is **nested**: the marker depends on the case which depends
  on the role which depends on the construction context.
- Many tokens are **legitimately ambiguous** — a noun next to another
  noun can be *mudaaf_ilayh* (idafa partner) or *mafoul_bih* (direct
  object); deciding requires verb-argument knowledge.
- Held-out gold data is **scarce**. Quality annotation requires a
  trained Arabic grammarian.
- Fine-grained roles + small sample sizes make leaderboard-driven
  evaluation deceptive — see § 7.

## 5 · Architecture evolution

| Phase | Idea | Outcome |
|---|---|---|
| 1 morph | AraT5v2-base + 7 morph heads | shipped |
| 2 cond | FiLM / additive / concat conditioning | dropped — joint dynamics broke role training |
| 3-A dep | + Stanza UD dep features as input augmentation | shipped (warm-start baseline) |
| 3.1 / R-C / R2 | output-hierarchy, CRF, hard constraints | dropped |
| 4a taxonomy | role-taxonomy expansion alone | dropped |
| nextgen leaked | 7-stage curriculum **with held-out sources accidentally in training pool** | apparent fully = 0.999 — discovered as contamination |
| **nextgen recovery** | **strict no-leakage retraining + 14-item recovery patch** | **shipped** (validated production checkpoint) |
| graph | gated graph refiner over word states | documented negative result |
| governor | biaffine governor head + attachment contrastive | documented negative result |

## 6 · Datasets

| Source | Role | n_sentences | sha256 (first 12) |
|---|---|---|---|
| `distill_v2` | training | 11,382 | 61eedb34b2c7 |
| `ud_padt_train` | training | 6,075 | 7b8583bd5c60 |
| `ud_padt_dev` | dev | 909 | 21ae6528980a |
| `ud_padt_test` | held-out (UD; partial gold) | 680 | 600deba71cc2 |
| `gazelle_test` | held-out (MSA gold) | 30 | d289d2c702f8 |
| `masaq_quranic` | held-out (Quranic gold) | 624 | 8118977c8d92 |

Provenance manifest: [`data_v2/manifests/provenance.json`](data_v2/manifests/provenance.json).
Three independent runtime assertions enforce that test sources never
enter the training pool.

## 7 · Evaluation methodology

All metrics are computed by a **single evaluator** (`src/irab_tashkeel/eval_v2/`)
applied identically to baselines and candidates. We report:

1. **Full noisy** evaluation — every observable token, even those with
   partial gold.
2. **Fully-observable** subset — tokens where all three gold labels
   (case, role, marker) are populated. Strict apples-to-apples.
3. **Calibration** — ECE + reliability bins per axis.
4. **Stratified** by domain, construction family, dependency depth,
   clause depth, sentence length, semantic pressure.
5. **Hard-case buckets** ([`docs/hard_eval/`](docs/hard_eval)) and the
   tighter v2 stress benchmark ([`data_v2/hard_eval_v2/`](data_v2/hard_eval_v2)).

Evaluation runner: [`scripts/eval/run_full_eval_v2.py`](scripts/eval/run_full_eval_v2.py).

## 8 · The leakage discovery (and recovery)

An earlier curriculum run (job 491628) reported MASAQ `fully = 0.999`.
The independent eval in [`docs/final_eval/`](docs/final_eval) traced
the result to `gazelle_test` and `masaq_quranic` being silently
present in the curriculum's training pool. The model had memorised
the test set.

Response (committed `c1a92bd`):

- **Three runtime assertions** — at module load, at pool build, at
  runtime sentence eligibility — refuse any test-source sentence in
  the training/rehearsal/hard-negative path.
- A **provenance manifest** with declared `split_role` per source
  ([`data_v2/manifests/provenance.json`](data_v2/manifests/provenance.json))
  and load-time enforcement.
- A leak-free retraining (job 491875) that produced
  `runs/validated_nextgen_recovery/`.
- The leakage audit pipeline ([`scripts/eval/leakage_audit.py`](scripts/eval/leakage_audit.py))
  is now part of the repo.

The leakage discovery is documented as a contribution, not as a
mistake. See [`docs/leakage_audit/leakage_report.md`](docs/leakage_audit/leakage_report.md).

## 9 · Validated metrics — production checkpoint

`runs/validated_nextgen_recovery` evaluated independently against
Phase 3-A on the full uncapped held-out sets:

| Dataset | Metric | Phase 3-A | Recovery | Δ |
|---|---|---|---|---|
| Gazelle (30 sent) | case | 0.638 | 0.646 | **+0.008** |
| Gazelle | role | 0.575 | 0.613 | **+0.038** |
| Gazelle | marker | 0.684 | 0.653 | −0.031 |
| Gazelle | fully | 0.459 | 0.459 | +0.000 |
| Gazelle | calib_gap | +0.021 | **−0.052** | healthier |
| MASAQ (624 sent) | case | 0.835 | 0.848 | **+0.014** |
| MASAQ | role | 0.778 | 0.807 | **+0.029** |
| MASAQ | marker | 0.718 | 0.710 | −0.008 |
| MASAQ | fully | 0.675 | 0.711 | **+0.036** |

Honest, modest, reproducible. Full report at
[`docs/final_eval_recovery/final_eval_report.md`](docs/final_eval_recovery/final_eval_report.md).

## 10 · Negative result — graph integration

A 2-layer gated graph refiner with edge-aware attention bias was
wired end-to-end into the model forward path (per-stage edge curriculum,
`-2` gate init, encoder freeze for 2 000 steps, edge dropout, full
ablation eval).

Training was stable; ablation delta was a consistent +0.006…+0.013
on cap-100 evals. On the full held-out sets the candidate did **not**
exceed `validated_recovery`:

| Dataset | metric | recovery | graph | Δ |
|---|---|---|---|---|
| Gazelle | fully | 0.459 | 0.459 | +0.000 |
| Gazelle | role  | 0.613 | 0.613 | +0.000 |
| MASAQ   | fully | 0.711 | 0.707 | −0.004 |
| MASAQ   | role  | 0.807 | 0.813 | +0.006 |

Full record: [`docs/final_graph_negative_result/`](docs/final_graph_negative_result).

## 11 · Negative result — governor head

A biaffine governor head + 0.1 × attachment-contrastive triplet loss
was wired and trained from the validated_recovery warm-start. Governor
CE went from ~3 → ~0.5 across training; attachment loss spiked
properly on nested-syntax data.

On the held-out sets, the dominant idafa confusions were unchanged:

| Confusion | recovery | governor |
|---|---|---|
| mudaaf_ilayh → mafoul_bih | 32 | 32 |
| mudaaf_ilayh → mubtada | 29 | 29 |
| mudaaf_ilayh → ism_majrur | 13 | 13 |

Full record: [`docs/final_governor_negative_result/`](docs/final_governor_negative_result).

## 12 · Failure analysis findings — the central result

Run [`scripts/analysis/run_failure_analysis.py`](scripts/analysis/run_failure_analysis.py)
on validated_recovery + the full Gazelle + MASAQ → the dominant
failure family is **idafa-attachment confusion**:

- `mudaaf_ilayh → mafoul_bih`: **32×**
- `mudaaf_ilayh → mubtada`: 29×
- `ism_majrur → matuf`: 21×
- `mudaaf_ilayh → ism_majrur`: 13×

The model cannot reliably distinguish:

| | When the second noun is a … |
|---|---|
| *mudaaf_ilayh* | partner in an idafa (set by the first noun) |
| *mafoul_bih* | direct object (set by an upstream verb) |
| *ism_majrur* | object of a preposition (set by a particle) |

These three readings have **near-identical surface forms**. The
confusion cannot be resolved structurally without verb-argument
knowledge or explicit ambiguity annotations — both of which are
explicit infrastructure in the repo (`data_v2/ambiguity_corpus/`,
`src/irab_tashkeel/data_v2/semantic/`) waiting on annotation.

Calibration is also a real problem: ECE on failures = case 0.42 /
role 0.49 / marker 0.60. The model is severely overconfident on the
hard cases. See [`docs/failure_analysis/`](docs/failure_analysis).

## 13 · Installation

```bash
# Python 3.11 + a modern PyTorch stack
pip install -e ".[dev]"
```

Dependencies pinned in `pyproject.toml`. The training stack uses
`torch`, `transformers`, `numpy`. The demo backend additionally needs
`fastapi` + `uvicorn`. The annotation server uses the same stack as
the demo.

## 14 · Training

```bash
# Build the schema_v2 corpus from raw sources (one-time)
python scripts/data_v2/build_schema_v2_corpus.py

# Build the provenance manifest (one-time, enforces split policy)
python scripts/data_v2/build_provenance_manifest.py

# Train the validated recovery checkpoint (the production model)
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

HPC (SLURM): [`scripts/slurm/93_train_curriculum_recovery.sbatch`](scripts/slurm/93_train_curriculum_recovery.sbatch).

## 15 · Inference

```python
import torch
from transformers import AutoTokenizer
from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

ckpt = "runs/validated_nextgen_recovery"
tok = AutoTokenizer.from_pretrained(ckpt)
model = DepAwareStructuredModel(
    encoder_name="UBC-NLP/AraT5v2-base-1024",
    enable_morph_heads=True, enable_dep_features=True,
)
model.load_state_dict(
    torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True),
    strict=False,
)
model.eval()
# … prepare word_starts/word_ends from tokenized words → forward → argmax
```

A complete inference helper is in [`demo/backend/inference.py`](demo/backend/inference.py).

## 16 · Demo

```bash
pip install fastapi uvicorn
PYTHONPATH=src uvicorn demo.backend.main:app --port 8000
# open http://localhost:8000
```

Six tabs: Sentence Analysis, Grammar Graph, Reasoning Trace,
Construction Breakdown, Evaluation Dashboard, Model Comparison
(and the leaked stage_7 is reachable from the Comparison tab as a
contamination case study).

## 17 · Reproducibility

Each frozen artifact ships with:

- `REPRODUCIBILITY_MANIFEST.json` — git commit, environment versions,
  dataset SHAs at training time
- `metrics.json` / `eval_tables.json` — full Phase A eval slice
- `calibration.json` — per-field reliability bins + ECE
- `training_manifest.json` — training summary + config
- `git_commit.txt`, `environment.txt`

Single-command reproduction recipe lives at the bottom of each
artifact's `README.md`.

## 18 · Repository structure

```
.
├── README.md
├── docs/
│   ├── paper/PAPER.md                    research writeup
│   ├── MODEL_CARD.md
│   ├── LIMITATIONS.md
│   ├── KNOWN_FAILURES.md
│   ├── final_eval_recovery/              validated headline metrics
│   ├── final_graph_negative_result/      graph experiment record
│   ├── final_governor_negative_result/   governor experiment record
│   ├── failure_analysis/                 idafa-confusion analysis
│   ├── hard_eval/                        per-bucket breakdown
│   └── leakage_audit/                    leakage discovery records
├── src/irab_tashkeel/
│   ├── data_v2/                          schema, loaders, provenance, semantic, normalization
│   ├── grammar_graph/                    graph engine (used as input augmentation)
│   ├── curriculum/                       7-stage scheduler + sampler + gates
│   ├── eval_v2/                          single-source-of-truth metrics
│   ├── eval_v3/                          ambiguity / uncertainty / structural
│   ├── training/                         hard-failure sampler, contrastive, SWA, LLRD, augmentations
│   ├── training_v2/                      curriculum trainer
│   ├── calibration/                      temperature scaling, focal loss
│   ├── ambiguity/                        AmbiguityExample schema
│   ├── annotation/                       review queue + server + disagreement resolution
│   ├── analysis/                         failure / confusion / structural / calibration analyses
│   ├── active_learning/                  uncertainty / disagreement / diversity / hard-case
│   ├── models/graph_refiner.py           (used in graph negative result)
│   └── morphology/dep_aware_model.py     production model class
├── scripts/
│   ├── training_v2/train_curriculum.py
│   ├── eval/{run_full_eval_v2,aggregate_full_eval,leakage_audit}.py
│   ├── analysis/{run_failure_analysis,run_hard_eval_report}.py
│   ├── data_v2/{build_*}.py
│   └── slurm/                            SLURM sbatch entry points
├── demo/                                 FastAPI + single-page UI
├── tests/
└── runs/
    ├── validated_nextgen_recovery/       PRODUCTION
    ├── final_validated/                  frozen artifact
    └── final_graph_negative_result/      frozen artifact
```

## 19 · Future work

The roadmap is now **data-driven, not architecture-driven**:

1. **Annotate the 4,233 mined ambiguity candidates**
   ([`data_v2/ambiguity_corpus/`](data_v2/ambiguity_corpus)) — particularly
   the idafa-attachment / preposition_vs_idafa / latent_governor kinds.
2. **Permissive evaluation** with `eval_v3.evaluate_with_ambiguity` once
   annotations land — the "correct" answer for genuinely ambiguous
   tokens should be a *set* of analyses, not a single label.
3. **Active-learning loop** ([`src/.../active_learning/`](src/irab_tashkeel/active_learning))
   over uncertain + disagreed-upon held-out sentences to grow gold
   data efficiently.
4. **Cross-dialect evaluation corpora** — Egyptian, Gulf, Maghrebi.
   Currently the project is MSA + Quranic only.
5. **Larger Arabic pretraining** (a separate project) once the gold
   data scale justifies it.

Architecture is **not** on this list. Both the graph and the governor
experiments showed that more structural supervision does not displace
the idafa confusion at our data scale; the bottleneck is lexical-semantic.

## 20 · Citation

```
@article{iraab_recovery_2026,
  title  = {A Case Study in Honest Arabic Grammatical Reasoning:
            From Leakage Collapse to Structural Ambiguity Bottlenecks},
  author = {Saadallah, Hatem and contributors},
  year   = {2026},
  url    = {https://github.com/Nourjennane/irab_project},
}
```

## 21 · License

See [`LICENSE`](LICENSE).
