# Training-Scale Plan

**Step 14 of the next-generation branch. Design only.**

The frozen baseline trained at 296M / 6 epochs / 77K rows. The
next-gen system needs to scale up multiple axes simultaneously
without losing the discipline (per-construction stratified eval,
calibration tracking, error taxonomy) that made the frozen
baseline rigorous.

## Axes to scale

| Axis | Frozen baseline | Next-gen target |
|---|---|---|
| Parameters | 296M | 296M–1.2B (backbone-dependent) |
| Training corpus | 77K word-rows from ~5K sentences | 50K–100K+ sentences with Layer A/B/C/D supervision |
| Epochs | 6 | 12–24 (with curriculum) |
| Batch size (effective) | 32 | 256+ via gradient accumulation |
| Sequence length | 320 | 320–1024 for long-context cells |
| Multi-task heads | morph (7) + iʿrāb (4) | morph (7) + iʿrāb (4) + reasoning (1) + construction (1) + clause (1) |
| Curriculum stages | 1 (mixed) | 7 stages (Step 7) |

## Distributed training

- Single A100-80GB / 4× A100 / 8× A100 schedules for the three
  parameter buckets (small / mid / large)
- Mixed precision (bf16) by default; fp16 with grad-scaling on
  GPUs without bf16
- Gradient checkpointing for the 1B+ models when sequence length > 768
- Gradient accumulation to maintain effective batch ≥ 256 across
  scales

## Mixed-domain scheduling

- Each batch sampled domain-stratified per the Step 1 metadata
  (msa_news / quranic / classical / educational / pedagogical)
- Domain ratios swept per curriculum stage; e.g. early stages
  may emphasise educational + msa_news for clean morph supervision,
  later stages add quranic + classical complexity
- Cross-domain batch interleaving prevents domain-specific drift

## Long-context batching

- Sequence-length curriculum: 320 → 512 → 768 → 1024 across stages
- Long-context examples (multi-sentence) are gated by domain
  (Step 11 discourse data) rather than length-only
- Bucket sampling so each batch has homogeneous lengths (avoids
  pad waste)

## Curriculum checkpoints

- Stage transitions are checkpoint-driven (per-stage gates)
- Each stage's checkpoint is preserved as a comparison point
- Per-stage checkpoints feed evaluation v2 (Step 13) so we can
  attribute gains to specific stages

## Retrieval-memory scaling

- Retrieval index (Step 12) starts at the frozen baseline's
  18,839 instances and scales to 100K+ as data_v2 lands
- Index rebuild scheduled at end of each curriculum stage, since
  earlier-stage embeddings drift
- Per-stage retrieval pools (early stages may have a smaller,
  cleaner pool than late stages with full discourse)

## Checkpoint averaging

- Stochastic weight averaging across the last 3 checkpoints of a
  stage (proven recipe for Arabic UD parsers)
- Averaged checkpoints become the per-stage shipped artefacts

## Large-batch optimisation

- Cosine LR schedule with linear warmup
- LR sweeps at 1× / 2× / 4× scale over frozen-baseline 5e-5
- Adafactor for the largest models (memory) vs AdamW for ≤500M

## Extended schedules

- 12–24 epoch retrains of Phase 3-A-style configs as a sanity
  baseline (does Phase 3-A's role-F1 keep climbing past 6 epochs?)
- For new backbones, train budget = max(2× frozen-baseline epochs,
  observed plateau on per-construction val metrics)

## Progressive freezing / unfreezing

- Stage 1: encoder unfrozen, all heads training
- Stage 2: encoder unfrozen, morph head frozen (preserve Layer A)
- Stage 3+: encoder bottom layers frozen, top layers + heads
  training
- Reasoning head (Step 9) joins training at Stage 4 onwards once
  syntactic foundation is stable

## Compute budget estimate

- Backbone benchmark (Step 6): ~15 GPU-hours
- Stage 1–3 (morph + local syntax + simple constructions):
  ~30 GPU-hours
- Stage 4–7 (nested + semantic + discourse + classical):
  ~60 GPU-hours
- Reasoning supervision integration: ~20 GPU-hours
- Total estimated next-gen training budget: ~125 GPU-hours,
  spread across 6–8 weeks of HPC scheduling

## Reproducibility requirements

- Per-experiment config locked in `docs/research_logs/<exp_id>.md`
- Stage checkpoints versioned by curriculum stage, not just epoch
- Per-stage eval frozen at stage end; never re-evaluated with a
  different evaluator after the stage closes
- Training-data digest (sha256 of the per-stage train.jsonl)
  stored with each checkpoint

## Open questions

- Do we use SLURM job arrays or one-off submits per stage?
- How are per-stage gates triggered — manual checkpoint review or
  automated criterion?
- How much of the frozen baseline's training data carries over,
  vs how much is replaced by data_v2 from scratch?
