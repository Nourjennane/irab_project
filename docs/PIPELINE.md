# Pipeline Reference

End-to-end map of the i'rāb project. Three tracks; pick what you need.

```
                 ┌─────────────────────────────┐
                 │     PADT (~6.7k MSA news)   │
                 │     QAC  (~78k Quranic)     │
                 │     Yarob (459 hand-written)│
                 │     Gazelle (30 gold)       │
                 └────────────┬────────────────┘
                              │
                  ┌───────────┴────────────┐
                  ▼                        ▼
       ┌──────────────────────┐ ┌─────────────────────────┐
       │  Distill via Claude  │ │ Cache as MTLExample list│
       │  (5–10k MSA pairs)   │ │ data/cache/combined.pkl │
       └──────────┬───────────┘ └────────────┬────────────┘
                  │                          │
                  ▼                          ▼
   ┌────────────────────────┐  ┌─────────────────────────┐
   │ data/distilled_irab    │  │  Per-word decoder       │
   │ .jsonl                 │  │  (existing baseline)    │
   └───────────┬────────────┘  │  src/.../irab_decoder.py│
               │               └──────────┬──────────────┘
               │                          │
               ▼                          ▼
  ┌──────────────────────────────────────────────────┐
  │  Stack A — QLoRA on Fanar/ALLaM/Yehia (Unsloth)  │
  │  configs/llm_qlora_fanar_unsloth.yaml            │
  │  scripts/slurm/32_train_qlora_fanar_unsloth.sb…  │
  └────────────┬─────────────────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────────────────┐
  │ Inference path                                   │
  │   src/.../inference/llm_baselines.py             │
  │      (claude_zero_shot, claude_fewshot_rag)      │
  │   src/.../inference/predictor.py (decoder)       │
  │   src/.../inference/constrained.py (taxonomy snap)│
  │   src/.../inference/cli.py                       │
  └────────────┬─────────────────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────────────────┐
  │ Evaluation                                       │
  │   src/.../evaluation/structural.py               │
  │      (case-acc, role-F1, marker-EM)              │
  │   src/.../evaluation/run_baselines.py            │
  │   src/.../evaluation/prepare_gold_seed.py        │
  └──────────────────────────────────────────────────┘
```

## What runs locally (laptop, no GPU)

```bash
export ANTHROPIC_API_KEY=...

# Working i'rāb generator, no training needed:
python -m irab_tashkeel.inference.cli --backend claude_rag \
    --sentence "ذهب الطالب إلى المدرسة"

# Score baselines on Gazelle gold set:
python -m irab_tashkeel.evaluation.run_baselines \
    --baselines decoder,claude_zero,claude_rag \
    --decoder_ckpt runs/model_small/best.pt \
    --out runs/baseline_eval

# Generate ~1k MSA training pairs via Claude Haiku 4.5 (~$7):
python -m irab_tashkeel.data.distill \
    --provider anthropic --model claude-haiku-4-5 \
    --source padt --n 1000 --budget_usd 12 \
    --out data/distilled_irab.jsonl

# Seed a 200-sentence gold benchmark for manual review (~$10 with Sonnet):
python -m irab_tashkeel.evaluation.prepare_gold_seed \
    --n 200 --model claude-sonnet-4-5 --budget_usd 10 \
    --out data/gold_seed.jsonl
# Then open data/gold_seed.jsonl, correct each row's irab fields,
# set "verified": true. That becomes your trustworthy benchmark.
```

## What runs on Bocconi (HPC)

```bash
# One-time:
sbatch scripts/slurm/00_setup_env.sbatch
sbatch scripts/slurm/10_smoke_test.sbatch

# Stack A primary (Unsloth + Liger + packing, ~3-4× faster):
sbatch scripts/slurm/32_train_qlora_fanar_unsloth.sbatch

# Stack B fallback (AraT5v2 full FT):
sbatch scripts/slurm/20_train_arat5v2.sbatch

# Stack A vanilla (no Unsloth — slower, more compatible):
sbatch scripts/slurm/30_train_qlora_fanar.sbatch
```

## Decision rule: do you even need fine-tuning?

1. Run `run_baselines.py` on Gazelle.
2. If `claude_rag` ≥ 70% case-accuracy: **ship it**. Skip fine-tuning entirely. The Claude API is your production system.
3. If `claude_rag` is in 50-70%: fine-tune Stack A using Yarob + distilled (drop_templated=true). Aim for +10 points.
4. If `claude_rag` < 50%: the prompt or retrieval is broken; fix that before training.

## Cost tracker (Anthropic)

| Step | Model | Volume | Estimated cost |
|------|-------|--------|----------------|
| Smoke distillation | claude-haiku-4-5 | 10 samples | <$0.10 |
| Full distillation | claude-haiku-4-5 | 1000 samples | ~$7 |
| Scale distillation | claude-haiku-4-5 | 5000 samples | ~$35 |
| Gold seed | claude-sonnet-4-5 | 200 sentences | ~$10 |
| Gazelle eval (3 baselines × 30 sents) | claude-haiku-4-5 | 90 calls | <$0.50 |

**Always rotate your API key after any session where it appears in chat or logs.**
