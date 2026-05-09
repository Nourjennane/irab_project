# Arabic Iʿrāb — Hierarchical Neural-Symbolic Grammatical Reasoning

> Per-word Arabic grammatical analysis (إعراب) with morphology, dependency,
> and construction-aware reasoning. Production model: validated nextgen
> stage_7 (curriculum-trained over Phase 3-A baseline).

[![status](https://img.shields.io/badge/status-validated-green)]()
[![phase](https://img.shields.io/badge/phase-curriculum%20stage_7-blue)]()
[![eval](https://img.shields.io/badge/eval-Gazelle%20%2B%20MASAQ%20%2B%20UD--PADT-blue)]()

---

## Project overview

This repository builds an end-to-end system that takes a raw Arabic sentence
and produces, for each token, the canonical iʿrāb labels expected of a
human reviewer:

- **Case** (raf / nasb / jarr / jazm / mabni)
- **Syntactic role** (mubtada, fail, mafoul_bih, ism_kana, ...)
- **Marker** (damma, fatha, kasra, sukun, …)
- **Morphology** (gender, number, definiteness, person, aspect, mood, voice)
- **Construction membership** (kana sisters, inna sisters, idafa, …)

The current production model is `runs/validated_nextgen_stage7/` — a
curriculum-trained extension of the Phase 3-A baseline that adds
hierarchical reasoning over morphology → dependency → role → case →
marker. See [`docs/final_eval/final_eval_report.md`](docs/final_eval/final_eval_report.md)
for full evaluation tables and [`docs/leakage_audit/leakage_report.md`](docs/leakage_audit/leakage_report.md)
for the train/test contamination audit.

## Motivation

Existing Arabic NLP systems either (a) predict diacritics without
exposing reasoning, or (b) emit lengthy free-form prose that is hard to
score and easy to hallucinate. We instead emit **structured per-token
grammatical analysis** so every prediction is independently verifiable,
calibrated, and renderable into pedagogically useful prose by a
deterministic template renderer.

## Linguistic problem

Arabic iʿrāb requires nested reasoning: the marker on a noun depends on
its case, which depends on its syntactic role, which depends on its
construction context (e.g. *ism kana* takes raf, *khabar kana* takes
nasb). A flat token-classification model cannot represent this because
the construction conditions which case slots are even available. The
nextgen model addresses this with a stage-curriculum over five
linguistic levels (morphology → local syntax → simple constructions →
nested syntax → semantic interactions → discourse → Quranic/classical).

## Architecture evolution

The repo's current state is the result of an explicit case study
(see [`docs/REPORT.md`](docs/REPORT.md)):

| Phase | Architecture | Outcome |
|---|---|---|
| 1 morph | AraT5v2-base + morph heads | shipped (+morph macro F1) |
| 2 conditioning | FiLM / additive / concat / detached FiLM | did **not** ship — joint training under conditioning was the bottleneck |
| 3-A dep | + Stanza UD dep features as input | shipped (current prod baseline; case +3.0 / marker +3.8 / fully +0.7) |
| 3.1 / R-C / R2 | output-hierarchy / CRF / hard-constraint stacking | did **not** ship |
| 4a taxonomy | role taxonomy expansion | did **not** ship |
| nextgen stage_7 | 7-stage curriculum + grammar graph + reasoning supervision over Phase 3-A | **shipped (this validated model)** — see [`docs/final_eval/`](docs/final_eval/) |

The thesis underlying this evolution: **orthogonal linguistic information
plus annotation-coverage quality** drives gains at this scale; downstream
architectural rearrangement plateaus.

## Curriculum training system

Implementation: [`src/irab_tashkeel/curriculum/`](src/irab_tashkeel/curriculum/).
Seven stages, each gated on a stage-specific metric:

| Stage | Name | Pool | Gate |
|---|---|---|---|
| 1 | morphology_foundation | distill_v2 (no morph supervision) | morph_macro_f1 ≥ 0.80 |
| 2 | local_syntax | + ud_padt_train | role_f1 ≥ 0.40 |
| 3 | simple_constructions | + masaq_quranic | construction_f1_macro ≥ 0.60 |
| 4 | nested_syntax | full pool | fully ≥ 0.25 |
| 5 | semantic_interactions | full pool | fully ≥ 0.30 |
| 6 | discourse_sensitive | full pool | fully ≥ 0.30 |
| 7 | quranic_classical | masaq_quranic only | quranic_fully ≥ 0.20 |

Stages can advance via gate (metric pass) or timeout (max_steps reached);
the scheduler is at [`src/irab_tashkeel/curriculum/scheduler.py`](src/irab_tashkeel/curriculum/scheduler.py).

## Grammar graph engine

[`src/irab_tashkeel/grammar_graph/`](src/irab_tashkeel/grammar_graph/) — five node
types (token, construction, clause, agreement, dependency) with seven
edge types (heads, agrees_with, governs, depends_on, …). Built per
sentence at training and inference time.

## Reasoning supervision

[`src/irab_tashkeel/reasoning/`](src/irab_tashkeel/reasoning/) — emits
deterministic prose explanations from structured labels via templates,
not free-form generation. Used for output explanations in the demo.

## Datasets

| Source | Purpose | n_sentences | Held out? |
|---|---|---|---|
| `distill_v2` | bulk training (low-supervision distillation) | 11,382 | no |
| `ud_padt_train` | UD-PADT training split | 6,075 | no |
| `ud_padt_dev` | UD-PADT dev | 909 | dev split |
| `ud_padt_test` | UD-PADT test | 680 | yes (but contaminated — see audit) |
| `gazelle_test` | Gazelle MSA gold | 30 | yes (clean) |
| `masaq_quranic` | MASAQ Quranic | 624 | yes (clean) |

Schema: [`src/irab_tashkeel/data_v2/schema_v2.py`](src/irab_tashkeel/data_v2/schema_v2.py).

## Evaluation methodology

[`scripts/eval/run_full_eval_v2.py`](scripts/eval/run_full_eval_v2.py) runs
the model over the *full* held-out test sets — no caps, no curriculum
sampling, no train-time augmentations, deterministic seed. Metrics are
computed by [`src/irab_tashkeel/eval_v2/`](src/irab_tashkeel/eval_v2/),
not by the training-loop hooks, so Phase 3-A and stage_7 are scored by
bit-identical code.

We report:

- per-field accuracy (case_acc / role_acc / marker_em / fully)
- calibration (gap on role, ECE, reliability bins per field)
- per-construction precision/recall/F1
- ambiguity-robust accuracy (excluding tokens with multiple gold labels)
- completeness-aware accuracy (`fully_observable_only=True`)
- stratified by domain and construction family

Two complementary evaluations:
1. **full noisy evaluation** — every token, even those with partial gold
2. **fully-observable** — only tokens with all 3 gold fields populated

## Leakage prevention

Before claiming any stage_7 gains we ran [`scripts/eval/leakage_audit.py`](scripts/eval/leakage_audit.py).
Findings (see [`docs/leakage_audit/leakage_report.md`](docs/leakage_audit/leakage_report.md)):

- **Gazelle: clean** — 0 exact / 0 normalised / 0 fuzzy across all train sources
- **MASAQ: clean** — 0 / 0 / 0 across all train sources
- **UD-PADT-test: contaminated** — 17 exact / 21 normalised / 65 fuzzy with `distill_v2`; 16 / 16 / 45 with `ud_padt_train`. Reported for completeness only; not used as a headline number.

## Results

Headline: see [`docs/final_eval/final_eval_report.md`](docs/final_eval/final_eval_report.md).

The clean comparisons (Gazelle + MASAQ) tell the story; UD-PADT numbers
are reported for completeness with a contamination caveat.

## Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and
[`docs/KNOWN_FAILURES.md`](docs/KNOWN_FAILURES.md). Summary:

- Calibration drifts at later stages (calib_gap rises from ~0.025 to ~0.20
  during multi-task training); model is overconfident on the nasb/jarr
  ambiguity in idafa edge cases.
- 30 sentences (Gazelle) is a small held-out sample.
- The reasoning trace is template-based, not generative — coverage limited
  to the canonical constructions defined in `data_v2/constructions/`.
- No cross-dialect evaluation — MSA + Quranic only.

## Demo instructions

```bash
cd demo/
docker compose up      # FastAPI backend + Next.js frontend
# open http://localhost:3000
```

See [`demo/README.md`](demo/README.md) for development workflow and the six
tab spec (Sentence Analysis, Grammar Graph, Reasoning Trace, Construction
Breakdown, Evaluation Dashboard, Model Comparison).

## Training instructions

```bash
# Build the schema_v2 corpus from raw sources
python scripts/data_v2/build_schema_v2_corpus.py

# Train the curriculum (assumes runs/phase3a_491240/final/ exists as warm-start)
PYTHONPATH=src python scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen --warm_start runs/phase3a_491240/final \
    --batch_size 16 --lr 1e-5 --eval_every 200 \
    --max_total_steps 60000
```

HPC: see [`scripts/slurm/91_train_curriculum.sbatch`](scripts/slurm/91_train_curriculum.sbatch).

## Inference instructions

```python
from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel
from transformers import AutoTokenizer
import torch

ckpt = "runs/validated_nextgen_stage7"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = DepAwareStructuredModel(
    encoder_name="UBC-NLP/AraT5v2-base-1024",
    enable_morph_heads=True, enable_dep_features=True,
)
model.load_state_dict(torch.load(f"{ckpt}/pytorch_model.bin",
                                  map_location="cpu", weights_only=True),
                     strict=False)
model.eval()
# … prepare word_starts/word_ends from tokenized words → forward → argmax
```

For ONNX / TorchScript inference, see [`runs/validated_nextgen_stage7/`](runs/validated_nextgen_stage7/).

## Reproducibility

[`runs/validated_nextgen_stage7/REPRODUCIBILITY_MANIFEST.json`](runs/validated_nextgen_stage7/REPRODUCIBILITY_MANIFEST.json)
captures git commit, env versions, and dataset SHAs at training time. To
regenerate the validated model from scratch:

1. Recreate the schema_v2 corpus from raw sources (`scripts/data_v2/`)
2. Re-train Phase 3-A baseline (`scripts/slurm/`)
3. Run curriculum on top of Phase 3-A (`scripts/slurm/91_train_curriculum.sbatch`)
4. Run independent eval (`scripts/slurm/92_full_eval_phase_a.sbatch`)
5. Run leakage audit (`scripts/eval/leakage_audit.py`)
6. Freeze (`scripts/freeze_validated_checkpoint.py`)

## Citation

```
@article{iraab_nextgen_2026,
  title  = {Hierarchical Neural-Symbolic Reasoning for Arabic Iʿrāb},
  author = {Saadallah, Hatem and contributors},
  year   = {2026},
  url    = {<repo-url>},
}
```

## Roadmap

- 11-phase plan in [`docs/long_term_direction.md`](docs/long_term_direction.md)
  (morphology → dependency → role → case → marker → semantic → discourse → cross-dialect)
- Next milestone: scaling pre-training corpus + cross-dialect eval
- Long horizon: full-parse iʿrāb with explicit reasoning chains, comparable to a tutoring system

## License

See [`LICENSE`](LICENSE).
