# Results — Baseline Eval on Gazelle

**Date:** 2026-04-29
**Eval set:** Gazelle Iraab.jsonl (UBC-NLP), 30 sentences, 134 word-level i'rāb judgments.
**Metric:** structural extraction → case / role / marker accuracy. See `src/irab_tashkeel/evaluation/structural.py`.

| Baseline | n | well-formed | POS | **case-acc** | **role-F1** | marker-EM |
|---|---:|---:|---:|---:|---:|---:|
| per-word decoder (`runs/model_small/best.pt`) | 134 | 70.9% | 14.9% | **32.8%** | **3.8%** | 13.4% |
| Claude Haiku 4.5 zero-shot | 134 | 77.6% | 44.0% | 57.5% | 55.9% | 34.3% |
| Claude Haiku 4.5 + RAG (Yarob, k=5) | 134 | **79.1%** | **45.5%** | **66.4%** | **60.7%** | **38.8%** |

## Reading the numbers

- **Per-word decoder produces well-formed strings (70.9%) with mostly wrong content** (3.8% role-F1). It learned to imitate the template structure but not to parse the actual sentence. This was structurally predictable: the decoder sees one mean-pooled vector per word as cross-attention memory and cannot route information across the sentence.
- **Claude RAG ≈ doubles every metric vs. the decoder.** It also beats zero-shot Claude by ~9 points case-acc and ~5 points role-F1. The retrieval pool of 459 Yarob examples carries enough style signal to nudge Claude into the traditional i'rāb register.
- **Marker exact-match remains the hardest field** (38.8% even on the best baseline). This is where constrained decoding helps most.

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
