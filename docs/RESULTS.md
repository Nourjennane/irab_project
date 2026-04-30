# Results — Baseline Eval on Gazelle

**Date:** 2026-04-30 (last eval). **Eval set:** Gazelle Iraab.jsonl (UBC-NLP), 30 sentences, 134 word-level i'rāb judgments.
**Metric:** structural extraction → `case` / `role` / `marker` / **`fully_correct_word`**. See `src/irab_tashkeel/evaluation/structural.py`.

`fully_correct_word` = case ∧ role ∧ marker all match — the metric Mix A is optimizing.

| Baseline | n | well-formed | POS | **case-acc** | **role-F1** | marker-EM | **fully** |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-word decoder (`runs/model_small/best.pt`) | 134 | 70.9% | 14.9% | **32.8%** | **3.8%** | 13.4% | TBD |
| Claude Haiku 4.5 zero-shot | 134 | 77.6% | 44.0% | 57.5% | 55.9% | 34.3% | TBD |
| Claude Haiku 4.5 + RAG (Yarob 459, k=5) | 134 | 79.1% | 45.5% | 66.4% | 60.7% | 38.8% | TBD |
| **Claude Haiku 4.5 + RAG (Yarob+distilled 1,060, k=5)** | 134 | **79.9%** | **45.5%** | **67.2%** | **68.8%** | **44.8%** | **27.6%** |

## Reading the numbers

- **Per-word decoder produces well-formed strings (70.9%) with mostly wrong content** (3.8% role-F1). It learned to imitate the template structure but not to parse the actual sentence. This was structurally predictable: the decoder sees one mean-pooled vector per word as cross-attention memory and cannot route information across the sentence.
- **Claude RAG ≈ doubles every metric vs. the decoder.** It beats zero-shot Claude by ~9 points case-acc and ~5 points role-F1. The retrieval pool of 459 Yarob examples carries enough style signal to nudge Claude into the traditional i'rāb register.
- **Expanding the pool with 601 Claude-distilled examples** (459 → 1060, all real-style MSA) lifted **role-F1 by +8.1 points** (60.7 → 68.8) and **marker-EM by +6 points** (38.8 → 44.8). Case-acc only nudged 0.8 — Claude already knew Arabic case from pretraining; retrieval doesn't add knowledge there.
- **Marker exact-match remains the hardest field** (44.8% even with the expanded pool). To improve, you'd need to fit the *exact gold lexical pattern* — that's a fine-tuning problem, not retrieval.
- **Aggregate `fully_correct_word` rate** (case ∧ role ∧ marker all right) is the honest end-to-end measure: **27.6%** for the current best system. Roughly 1 in 4 words gets all three fields right. Mix A's per-word routing is targeted at moving this number.

## Implication for the project plan

The original "ship a 24h-trained 9B QLoRA" framing optimized for the wrong problem. With **Claude RAG already at 66.4% case-acc with no fine-tuning**, the question is no longer "will we beat Farasa," it's "can SFT push us above 80%?" The empirical headroom for fine-tuning is now bounded — at most ~13 points of case-acc.

That makes Stack A a measurable bet, not a leap of faith.

## Next decision

Per the plan's decision rule:
- ≥70% case-acc on best baseline → ship the API system, skip fine-tuning
- 50-70% → fine-tune; aim for +10
- <50% → fix the prompt/retrieval first

We're at 66.4%. **Fine-tune Stack A** (Unsloth + Liger + packing) on Yarob + distilled-Claude pairs once distillation completes. Drop the templated QAC/PADT data — it taught the decoder the wrong thing and would do the same to an LLM.

## Reproduction

```bash
ANTHROPIC_API_KEY=... .venv/bin/python -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle \
    --baselines decoder,claude_zero,claude_rag \
    --decoder_ckpt runs/model_small/best.pt \
    --model claude-haiku-4-5 \
    --rag_k 5 \
    --out runs/baseline_eval
```

Total spend for this evaluation: ~$0.40.
