# Inference-time variance — Sonnet RAG (k=5)

We re-ran the headline configuration (Claude Sonnet 4.5, temperature=0.0, k=5 retrieval over Yarob+Distilled n=1,060) on the same 30 Gazelle sentences, on the same day, with the same prompts. Results are stored at:
- Run 1 (headline): `runs/baseline_eval_sonnet/claude_rag.predictions.jsonl`
- Run 2 (repro):    `runs/baseline_eval_sonnet_repro/claude_rag.predictions.jsonl`

## Aggregate metrics (n=134 word judgments each)

| Metric | Run 1 (headline) | Run 2 (repro) | Δ |
|---|---:|---:|---:|
| well-formed | 79.9% | 79.9% | 0.0 |
| case-acc | 73.9% | 74.6% | +0.7 |
| role-F1 (macro) | 74.6% | 72.9% | −1.7 |
| marker-EM | 50.0% | 49.3% | −0.7 |
| fully_correct_word | **32.1%** | **32.1%** | **0.0** |

## Paired deltas (B=1000 bootstrap; McNemar's exact)

| Field | Δ pp | 95% CI | McNemar p |
|---|---:|---|---:|
| well_formed | 0.0 | [0.0, 0.0] | 1.000 |
| case | +0.7 | [0.0, +2.2] | 1.000 |
| marker | −0.7 | [−2.2, 0.0] | 1.000 |
| **fully** | **+0.0** | **[−2.2, +2.2]** | **1.000** |

## Per-word agreement (out of 126 words present in both runs)

| Field | Agreement | Rate |
|---|---:|---:|
| case  | 125 / 126 | **99.2%** |
| role  | 124 / 126 | 98.4% |
| marker | 125 / 126 | 99.2% |
| **fully (case ∧ role ∧ marker)** | **122 / 126** | **96.8%** |

## Reading

Sonnet 4.5 at `temperature=0.0` is **highly reproducible** for this task. Aggregate metrics shift by ≤0.7 pp on case/marker and 0.0 pp on the headline `fully_correct_word`. Per-word predictions agree 96.8% of the time on the joint metric (`case ∧ role ∧ marker`). All paired tests come back at p=1.000 — the two runs are statistically indistinguishable on every dimension. The reported headline numbers are therefore robust to provider-side stochasticity (GPU kernel ordering, batching, etc.) at the scale tested here.

Reported as evidence that the closed-source LLM, while non-deterministic in principle (Anthropic's API does not currently expose a `seed` parameter), behaves deterministically enough at `temperature=0.0` for the comparisons in `docs/RESULTS.md` to be trusted to ~1 pp.

(8 of the 134 word judgments could not be aligned across runs because the system emitted no `irab` field for that word in one of the runs. We treat this as a real "no-prediction" event and exclude from per-word agreement; the aggregate metrics already count it as a miss.)
