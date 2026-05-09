# Model Card — Validated Nextgen Recovery

## Overview

- **Name:** validated_nextgen_recovery
- **Architecture:** DepAwareStructuredModel (AraT5v2-base encoder + multi-head structured prediction + UD dep-feature input augmentation)
- **Trained:** 2026-05-09 (recovery run, leak-free)
- **Training procedure:** 7-stage curriculum, 7,400 steps, lr=1e-5 with layer-wise decay 0.85, batch=16, fp32, label smoothing 0.05, structured-consistency penalty, exact-fully aux loss, hard-failure sampler, EMA, SWA. Early stop on `strict_unseen_fully` patience 3.
- **Warm-start:** Phase 3-A baseline (`runs/phase3a_491240/final/`)
- **Compute:** 1× NVIDIA GPU (Bocconi HPC stud QoS), ~27 minutes wall-clock
- **Frozen at:** `runs/validated_nextgen_recovery/`
- **Leakage policy:** `gazelle_test`, `masaq_quranic`, `ud_padt_test` are forbidden from any training, rehearsal, hard-negative, or graph-construction pool. Three independent runtime assertions enforce this. See `src/irab_tashkeel/curriculum/config.py:TEST_SOURCES`.

## Inputs and outputs

**Input:** Arabic sentence as raw text (MSA or Quranic; no diacritization required).

**Output (per token):**
- `case` ∈ {raf, nasb, jarr, jazm, mabni}
- `role` ∈ ~25 syntactic roles (mubtada, fail, mafoul_bih, ism_kana, ...)
- `marker` ∈ {damma, fatha, kasra, sukun, ya, alif, waw, ...}
- `pos` (UPOS-aligned)
- `morph` per axis: gender, number, definite, person, aspect, mood, voice
- per-field confidences (softmax max)

## Training data

| Source | n | Role |
|---|---|---|
| distill_v2 | 11,382 | bulk distillation |
| ud_padt_train | 6,075 | gold UD-aligned |
| ud_padt_dev | 909 | dev (used in curriculum pool) |
| masaq_quranic | 624 | Quranic gold |
| gazelle_test | 30 | held-out (not in train) |
| ud_padt_test | 680 | held-out (contaminated — see audit) |

## Evaluation

See [`docs/final_eval/final_eval_report.md`](final_eval/final_eval_report.md)
for the independent evaluator output.

Headline (clean held-out only):

- Gazelle: case_acc, role_f1, marker_em, fully — full table in report
- MASAQ Quranic: case_acc, role_f1, marker_em, fully — full table in report

## Calibration

Stage_7 calibration gap on the role head is ~0.20 → **the model is
overconfident**. Apply temperature scaling on a held-out shard before
using probabilities downstream. See [`docs/LIMITATIONS.md`](LIMITATIONS.md).

## Bias and risks

- All training data is MSA + Quranic. Dialect performance unmeasured.
- 30-sentence Gazelle held-out is a small sample — confidence intervals are wide.
- The reasoning trace is template-based; **the model does not invent
  explanations**. This is a design choice to prevent hallucination.

## Intended use

- Educational tools (Arabic grammar tutoring, error-flagging in student writing)
- Research baseline for Arabic structured-prediction benchmarking
- Inference-only deployments where every prediction is independently verifiable

## Out-of-scope use

- Translation, summarization, dialogue (no generation head)
- Cross-dialect grammar analysis (untested)
- Legal or safety-critical applications without human review (calibration drift)

## How to load

```python
import torch
from transformers import AutoTokenizer
from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

ckpt = "runs/validated_nextgen_stage7"
tok = AutoTokenizer.from_pretrained(ckpt)
model = DepAwareStructuredModel(
    encoder_name="UBC-NLP/AraT5v2-base-1024",
    enable_morph_heads=True, enable_dep_features=True,
)
sd = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=False); model.eval()
```

## Reproducibility

`runs/validated_nextgen_stage7/REPRODUCIBILITY_MANIFEST.json` captures
git commit hash, environment versions, and dataset SHAs at the freeze
moment. Re-run [`scripts/freeze_validated_checkpoint.py`](../scripts/freeze_validated_checkpoint.py)
on the source artifacts to recreate.

## Citation

See repo root [`README.md`](../README.md).

## License

See [`LICENSE`](../LICENSE).
