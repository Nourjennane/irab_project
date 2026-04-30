# Results — Per-word Arabic i'rāb on Gazelle

**Eval set:** Gazelle Iraab.jsonl (UBC-NLP, EMNLP 2024 shared release), 30 hand-written gold MSA sentences = **134 word-level i'rāb judgments**.
**Date of last refresh:** 2026-04-30.

**Metric definition:** structural extraction (regex/FSM) over each generated i'rāb string yields four atomic fields: `pos`, `case ∈ {rafʿ, naṣb, jarr, jazm, mabni}`, `role` (~25 traditional labels), `marker` (~200 unique phrases). The headline aggregate is `fully_correct_word = case ∧ role ∧ marker all match`. See `src/irab_tashkeel/evaluation/structural.py`.

**All numbers are reported with 95% percentile bootstrap CIs** (B=1000 resamples on the 134 word judgments). System-vs-system deltas use **paired bootstrap** + **McNemar's exact test** on matched per-word outcomes; an entry is marked ★ when both the bootstrap CI excludes 0 and McNemar p < 0.05.

---

## Headline comparison

| System | well-formed | case-acc | marker-EM | role-F1 (macro) | **fully** |
|---|---:|---:|---:|---:|---:|
| Claude Haiku 4.5 zero-shot | 77.6 [70.1, 84.3] | 57.5 [49.3, 66.4] | 40.3 [32.1, 49.3] | 55.9 [42.0, 68.6] | 18.7 [11.9, 25.4] |
| **Claude RAG (Yarob+distilled, k=5)** | **79.9 [73.1, 86.6]** | **67.2 [59.0, 75.4]** | **44.8 [35.8, 53.0]** | **68.8 [55.8, 82.2]** | **27.6 [20.1, 35.8]** |
| Hybrid (RAG case+role + AraT5v2 marker) | 77.6 [70.1, 84.3] | 68.7 [61.2, 76.9] | 41.8 [33.6, 50.7] | 59.4 [44.8, 73.8] | 26.1 [18.7, 34.3] |

(Numbers are percentages with 95% bootstrap CIs in brackets.)

The from-scratch character decoder is reported as a documented negative-result baseline in the appendix only — it scores 32.8% [25.4, 41.0] case and 3.8% [1.5, 8.2] role-F1 on the same eval, confirming that training i'rāb prose generation from scratch fails at this data scale.

---

## Statistically significant differences (paired bootstrap + McNemar p<0.05)

| Comparison | Δ case-acc | 95% CI | McNemar p | Δ marker-EM | 95% CI | McNemar p | Δ fully | 95% CI | McNemar p |
|---|---:|---|---:|---:|---|---:|---:|---|---:|
| RAG combined − Decoder | **+34.3** ★ | [+24.6, +44.0] | <0.001 | **+31.3** ★ | [+23.1, +40.3] | <0.001 | **+25.4** ★ | [+17.9, +33.6] | <0.001 |
| RAG combined − Zero-shot | **+9.7** ★ | [+3.0, +16.4] | 0.011 | +4.5 | [-0.7, +9.7] | 0.180 | **+9.0** ★ | [+3.7, +14.9] | 0.004 |
| RAG combined − RAG (Yarob-only) | +0.7 | [-2.2, +3.7] | 1.000 | +1.5 | [-0.0, +3.7] | 0.500 | +1.5 | [-1.5, +4.5] | 0.625 |
| **Hybrid − RAG combined** | **+1.5** | [-4.5, +6.7] | **0.791** | **-3.0** | [-6.7, +0.0] | **0.219** | **-1.5** | [-6.7, +3.7] | **0.791** |

**Statistical reading:**
- **Retrieval augmentation over zero-shot is a real lift** on case (+9.7 pp, p=0.011) and on the aggregate fully-correct rate (+9.0 pp, p=0.004). Marker EM and well-formedness do not change significantly.
- **Adding 601 Claude-distilled MSA pairs to the retrieval pool produced no significant change on any binary metric** at this evaluation scale. Role-F1 trended upward (+8.1 pp point estimate) but the bootstrap CIs of the two systems overlap heavily ([55.8, 82.2] vs [50.3, 74.9]). We use the combined-pool variant in subsequent comparisons for higher per-system role coverage but **cannot claim distillation lifts retrieval-augmented generation accuracy on Gazelle**.
- **The from-scratch decoder is dominated** on every metric by every LLM-based system at p<0.001.
- **Mix A (Hybrid) does not beat Claude RAG at this evaluation scale.** All four binary metrics fall within the noise floor: Δ case +1.5 pp [p=0.791], Δ marker −3.0 pp [p=0.219], Δ fully −1.5 pp [p=0.791], Δ well-formed −2.2 pp [p=0.375]. The point estimate on role-F1 dropped 9.4 pp but bootstrap CIs overlap heavily ([55.8, 82.2] vs [44.8, 73.8]). We interpret this as a **negative result on the per-word routing hypothesis**: at n=134 we cannot detect a difference in either direction. The asymmetry we hypothesized — knowledge-bound case/role vs. style-bound marker — does not translate into measurable gains when the LLM baseline is already strong on style via retrieval-augmented prose generation.

The smallest detectable difference at n=134 with α=0.05 power 0.80 is roughly **±7 percentage points** on a binary proportion; differences below that are within noise.

---

## Error analysis — per-construction breakdown of `fully_correct_word`

Each Gazelle sentence was classified into one or more construction tags using the gold i'rāb prose itself (heuristic regex on terms like فعل ماض / مضاف إليه / إن / سوى / …). Per-tag breakdown (overlapping; a sentence may have multiple tags):

| Tag | # sent | # words | Decoder fully | Zero-shot fully | RAG fully |
|---|---:|---:|---:|---:|---:|
| NOMINAL | 5 | 18 | 11.1 [0.0, 27.8] | 50.0 [27.8, 72.2] | **61.1 [38.9, 83.3]** |
| PARTICLE_MOOD | 6 | 30 | 0.0 [0.0, 0.0] | 16.7 [3.3, 30.0] | 26.7 [10.0, 43.3] |
| VERBAL | 15 | 61 | 1.6 [0.0, 6.6] | 14.8 [6.6, 24.6] | 24.6 [13.1, 36.1] |
| PREPOSITIONAL | 8 | 37 | 0.0 [0.0, 0.0] | 16.2 [5.4, 29.7] | 21.6 [8.1, 35.1] |
| IDAFA_HEAVY | 1 | 9 | 0.0 [0.0, 0.0] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] |
| **EXCEPTION (istithnāʾ)** | 2 | 9 | **0.0** | **0.0** | **0.0** |
| OTHER | 3 | 16 | 0.0 [0.0, 0.0] | 6.2 [0.0, 18.8] | 12.5 [0.0, 31.2] |

(95% bootstrap CIs in brackets.)

**Findings:**
- **Nominal sentences are ~2× easier** than verbal ones for all LLM-based systems. This is consistent with their shorter dependency chains (a copular clause has at most one case-marking interaction) and aligns with prior observations on Arabic morphological tagging difficulty.
- **Exception (istithnāʾ) constructions are a complete failure mode**: 0/9 words correctly analyzed by every system, including the strongest. The two affected sentences (`سوى تلميذين`, `عدا واحدا`) require recognizing that the noun after `سوى/عدا/إلا` takes a non-default case based on whether the exception is positive or negative — a rule Claude appears not to apply consistently from prompt context alone.
- **IDAFA_HEAVY has only one Gazelle sentence** (n=9 words), so its 22.2% rate is uninformative; the metric on this category is reported with a 95% CI of [0.0, 55.6] which is wider than its point value. **This is a sampling limitation, not a model finding.**
- **Verbal-vs-prepositional is essentially a wash** (RAG 24.6 vs 21.6) — no detectable effect of preposition presence on aggregate correctness once verbal-sentence frequency is accounted for.

---

## Reproduction

```bash
ANTHROPIC_API_KEY=sk-... python -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle --baselines decoder,claude_zero,claude_rag \
    --decoder_ckpt runs/model_small/best.pt \
    --model claude-haiku-4-5 --rag_k 5 \
    --out runs/baseline_eval_v2

python -m irab_tashkeel.evaluation.stats \
    --system decoder=runs/baseline_eval/decoder.predictions.jsonl \
    --system claude_zero=runs/baseline_eval/claude_zero.predictions.jsonl \
    --system claude_rag=runs/baseline_eval_v2/claude_rag.predictions.jsonl \
    --reference claude_rag --B 1000

python -m irab_tashkeel.evaluation.error_analysis \
    --system decoder=runs/baseline_eval/decoder.predictions.jsonl \
    --system claude_zero=runs/baseline_eval/claude_zero.predictions.jsonl \
    --system claude_rag=runs/baseline_eval_v2/claude_rag.predictions.jsonl \
    --metric fully
```

Total API spend for these evaluations: ~$0.50 across 3 systems × 30 sentences.

---

## Limitations relevant to these numbers

1. **Sample size.** All comparisons rest on n=134 word judgments from 30 sentences. The smallest detectable difference at α=0.05 is ~7 pp on binary metrics; we report only differences that survive paired bootstrap and McNemar's exact test.
2. **Construction coverage.** Gazelle's distribution skews toward short verbal sentences (50%) and prepositional phrases (27%); iḍāfa chains of length ≥3 (n=1) and exception constructions (n=2) are under-represented. Generalization claims are limited to *MSA news-style sentences within these construction frequencies*.
3. **Marker extractor floor.** The regex/FSM extractor recovers an explicit marker phrase for 89.7% of distilled training sentences; the remaining 10.3% (mahall pronouns, unparsed Claude outputs) are scored as `<NO_MARKER>`. This introduces a systematic ~10% measurement floor on marker EM.
4. **Teacher-bound retrieval pool.** 601 of 1,060 retrieval pool examples are Claude-generated (with manual lookup over Yarob 459 as the gold source). Mix A's marker training set inherits Claude's systematic phrasing biases; we cannot rule out that any AraT5v2 marker model fits Claude's style rather than gold MSA grammatical-tradition style.

---

## Appendix A — Per-word decoder (negative-result baseline)

A 5M-parameter character-level Transformer with three classification heads + a per-word seq2seq decoder for the prose form was trained from scratch on the unified MTLExample set (QAC ~78K word labels + Yarob 459 + PADT-templated 6.7K + synthetic). Final evaluation:

| Metric | Decoder | RAG combined |
|---|---:|---:|
| well-formed | 70.9 [62.7, 78.4] | 79.9 [73.1, 86.6] |
| case-acc | 32.8 [25.4, 41.0] | 67.2 [59.0, 75.4] |
| role-F1 (macro) | 3.8 [1.5, 8.2] | 68.8 [55.8, 82.2] |
| marker-EM | 13.4 [8.2, 20.1] | 44.8 [35.8, 53.0] |
| fully | 2.2 [0.0, 5.2] | 27.6 [20.1, 35.8] |

The decoder produces structurally well-formed output (70.9% parseable) but the content is essentially random with respect to actual sentence syntax — role-F1 of 3.8% indicates near-zero discriminative ability across role classes. We attribute this to two structural causes: (a) the decoder's cross-attention memory for word *n* is a single mean-pooled vector over the word's character span, which does not carry enough surrounding-word information to condition role assignment; (b) ~85% of training i'rāb labels are deterministically templated from POS+Case features, encouraging the model to memorize templates rather than learn syntactic dependencies. **The decoder is reported only to motivate the LLM-based comparison; it is not a peer system to Claude RAG or Hybrid.**
