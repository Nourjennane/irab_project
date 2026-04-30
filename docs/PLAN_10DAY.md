# 10-Day Plan — Mix A + Manual Gold + LLM-as-Judge

Deadline: ~2026-05-10. HPC: confirmed. Anthropic budget: ~$30 across the run.

---

## Day 0 — today (2026-04-30, in-progress)

- [x] HPC memory note flipped from "deferred" → "active"
- [x] Combined-pool RAG eval (67.2% case, 68.8% role, 39.6% marker) — `runs/baseline_eval_v2/`
- [ ] Add `fully_correct_word` metric to `structural.py` (case ∧ role ∧ marker)
- [ ] Build marker-extraction tool (`evaluation/marker_extract.py`) — pulls a clean marker substring from each gold i'rāb string; classifies "no-marker" cases
- [ ] Update `docs/RESULTS.md` with combined-pool numbers
- [ ] Re-submit Bocconi `00_setup_env.sbatch` (queues behind rtdetr-p2; auto-runs when slot frees ~13:42 UTC today)

## Day 1 — Mon (May 1)

- [ ] Setup job completes overnight, smoke test passes
- [ ] Generate manual gold seed: `prepare_gold_seed.py --n 200 --model claude-sonnet-4-5` (~$10, ~30 min)
- [ ] Hand-off file `data/gold_seed.jsonl` for user review

## Day 2 — Tue

- [ ] **User: start hand-correcting `data/gold_seed.jsonl`** (1-2 days, ~30 sentences/h)
- [ ] Build `data/marker_pairs.jsonl`: extract (sentence, word, case, role) → marker for every word in Yarob + distilled (~7k pairs after cleanup)
- [ ] Add config + sbatch for the marker-only fine-tune

## Day 3 — Wed

- [ ] Submit `33_train_marker_arat5v2.sbatch` (AraT5v2-base on marker prediction; estimated 8-12h on `stud` MIG slice)
- [ ] User continues gold review

## Day 4 — Thu

- [ ] Marker fine-tune completes
- [ ] Build `inference/hybrid.py`: takes Claude RAG output + per-word AraT5v2 marker prediction, produces final i'rāb
- [ ] User finishes gold review (target: 200 verified rows)

## Day 5 — Fri

- [ ] Run full eval (decoder / Claude RAG / AraT5v2-marker-standalone / **Hybrid**) on:
  - Gazelle (30)
  - Manual gold (200)
- [ ] Compute `fully_correct_word` rate for each system

## Day 6 — Sat

- [ ] Add LLM-as-judge eval: 4-axis rubric (well-formed / case / role / marker), GPT-4o-mini judge, both Claude RAG and Hybrid scored against gold (~$5)
- [ ] Polish hybrid: tune the case/role/marker routing thresholds if needed

## Day 7 — Sun

- [ ] Streamlit demo using Hybrid system; deploy locally or via `streamlit run`
- [ ] Begin writeup: data engineering + 4-baseline comparison + structural metric methodology

## Day 8 — Mon (May 8)

- [ ] Continue writeup
- [ ] Generate the final result tables, Mix A architecture diagram, method-comparison plots

## Day 9 — Tue

- [ ] Final review: re-run all evals once more for the report
- [ ] Polish demo, write user-facing README

## Day 10 — Wed (May 10)

- [ ] Submit

---

## Decision points

- **End of Day 4:** if marker fine-tune lands ≥55% marker-EM, Hybrid is the headline. If 40-55%, Claude RAG remains the best system; report Hybrid as a documented attempt with mixed results.
- **End of Day 6:** if `fully_correct_word` for Hybrid is < Claude RAG, ship Claude RAG as the final system, present Hybrid as ablation.

## Honest risks

- **Marker extraction from gold might be 30-40% no-marker class.** Mitigate: train a 2-output head (marker text + has-marker flag).
- **Bocconi `stud` MIG slice may be slower than full A100.** Plan: 12-15h training instead of 6-8.
- **Anthropic API rate-limits returned earlier today** (got us stuck at 599→601 in 6 min). Mitigate: route gold-seeding via Sonnet (different rate tier), break into batches of 50 with sleep.
- **Manual gold review is the slowest step.** 200 sentences × 2-3 min/each ≈ 6-10 hours over 2 days. If user can't, fall back to scoring on Gazelle 30 only.

## API spend budget

| Step | Cost |
|---|---|
| Eval re-runs (×3) | ~$1 |
| Manual gold seed (200 × Sonnet) | ~$10 |
| LLM-as-judge eval (~600 calls × Haiku judge) | ~$3 |
| Buffer for retries / extra distillation | ~$10 |
| **Total** | **~$24** |

Hard cap any single run at $15.
