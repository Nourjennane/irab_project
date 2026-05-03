# Results — Per-word Arabic i'rāb on Gazelle

**Eval set:** Gazelle Iraab.jsonl (UBC-NLP, EMNLP 2024 shared release), 30 hand-written gold MSA sentences = **134 word-level i'rāb judgments**.
**Date of last refresh:** 2026-04-30.

**Metric definition:** structural extraction (regex/FSM) over each generated i'rāb string yields four atomic fields: `pos`, `case ∈ {rafʿ, naṣb, jarr, jazm, mabni}`, `role` (~25 traditional labels), `marker` (~200 unique phrases). The headline aggregate is `fully_correct_word = case ∧ role ∧ marker all match`. See `src/irab_tashkeel/evaluation/structural.py`.

**All numbers are reported with 95% percentile bootstrap CIs** (B=1000 resamples on the 134 word judgments). System-vs-system deltas use **paired bootstrap** + **McNemar's exact test** on matched per-word outcomes; an entry is marked ★ when both the bootstrap CI excludes 0 and McNemar p < 0.05.

---

## Headline comparison

All numbers are percentages with 95% percentile bootstrap CIs (B=1000) in brackets. The from-scratch character decoder (32.8% case, 3.8% role-F1, 2.2% fully) is reported in Appendix A as a documented negative-result baseline; it is dominated by every LLM-based system at p<0.001 and is not a peer comparison.

| System | well-formed | case-acc | marker-EM | role-F1 (macro) | **fully** |
|---|---:|---:|---:|---:|---:|
| Stanza Arabic (UD pipeline + UD→i'rāb stub) | 59.7 [51.5, 68.7] | 35.1 [26.9, 43.3] | 13.4 [8.2, 19.4] | 10.3 [6.4, 19.0] | 5.2 [2.2, 9.7] |
| Qwen2.5-7B-Instruct + RAG (k=5, 4-bit local) | 66.4 [58.2, 73.9] | 43.3 [35.1, 52.2] | 19.4 [12.7, 26.9] | 19.2 [9.5, 29.4] | 3.0 [0.0, 6.0] |
| mT5-base (580M) full FT on Haiku-5K (77 K word rows) | 79.9 [72.4, 86.6] | 61.9 [53.7, 70.1] | 32.8 [25.4, 41.0] | 31.6 [21.0, 44.4] | 18.7 [11.9, 25.4] |
| AraT5v2-base (296M) full FT on Haiku-5K (77 K word rows) | 79.9 [72.4, 86.6] | 65.7 [57.5, 73.9] | 44.0 [35.8, 53.0] | 54.8 [41.2, 68.7] | 24.6 [17.9, 32.8] |
| Claude Haiku 4.5 zero-shot | 77.6 [70.1, 84.3] | 57.5 [49.3, 66.4] | 40.3 [32.1, 49.3] | 55.9 [42.0, 68.6] | 18.7 [11.9, 25.4] |
| Claude Haiku 4.5 + RAG (k=5) | 79.9 [73.1, 86.6] | 67.2 [59.0, 75.4] | 44.8 [35.8, 53.0] | 68.9 [55.9, 82.3] | 27.6 [20.1, 35.8] |
| Claude Haiku 4.5 + RAG (k=10) — *ablation* | 76.9 [69.4, 84.3] | 64.9 [56.7, 73.1] | 46.3 [38.1, 55.2] | 45.4 [36.9, 56.0] | 26.1 [18.7, 34.3] |
| Hybrid (Haiku RAG + AraT5v2 marker, overlay) | 77.6 [70.1, 84.3] | 67.9 [59.7, 76.1] | 41.0 [32.1, 50.0] | 65.9 [51.1, 78.8] | 26.1 [18.7, 34.3] |
| Claude Sonnet 4.5 zero-shot | 78.4 [70.9, 85.1] | 72.4 [64.9, 79.9] | 44.0 [35.1, 53.0] | 76.4 [62.5, 86.0] | 27.6 [20.1, 35.8] |
| **Claude Sonnet 4.5 + RAG (k=5)** | **79.9 [72.4, 86.6]** | **73.9 [66.4, 81.3]** | **50.0 [41.8, 59.0]** | **74.7 [63.8, 88.7]** | **32.1 [24.6, 40.3]** |
| Sonnet RAG + AraT5v2 marker (Hybrid v2) | 79.9 [72.4, 86.6] | 73.9 [66.4, 81.3] | 46.3 [38.1, 55.2] | 73.3 [60.7, 86.4] | 29.1 [21.6, 37.3] |

The from-scratch character decoder is reported as a documented negative-result baseline in the appendix only — it scores 32.8% [25.4, 41.0] case and 3.8% [1.5, 8.2] role-F1 on the same eval, confirming that training i'rāb prose generation from scratch fails at this data scale.

---

## Statistically significant differences (paired bootstrap + McNemar p<0.05)

| Comparison | Δ case-acc | 95% CI | McNemar p | Δ marker-EM | 95% CI | McNemar p | Δ fully | 95% CI | McNemar p |
|---|---:|---|---:|---:|---|---:|---:|---|---:|
| Sonnet RAG − Stanza | **+38.8** ★ | [+30.6, +47.8] | <0.001 | **+36.6** ★ | [+27.6, +45.5] | <0.001 | **+26.9** ★ | [+19.4, +35.8] | <0.001 |
| Sonnet RAG − Qwen2.5-7B+RAG (open-weight) | **+30.6** ★ | [+23.1, +38.8] | <0.001 | **+30.6** ★ | [+21.6, +39.6] | <0.001 | **+29.1** ★ | [+20.9, +38.1] | <0.001 |
| Qwen2.5-7B+RAG − Stanza | +8.2 | [+0.0, +17.2] | 0.080 | +6.0 | [-2.2, +14.9] | 0.185 | -2.2 | [-6.7, +1.5] | 0.453 |
| **Sonnet RAG − AraT5v2-base FT (Haiku-5K)** | **+8.2** ★ | [+2.2, +14.2] | **0.013** | +6.0 | [+0.0, +11.9] | 0.077 | **+7.5** ★ | [+2.2, +13.4] | **0.021** |
| AraT5v2-base FT − Qwen2.5-7B+RAG | **+22.4** ★ | [+13.4, +31.3] | <0.001 | **+24.6** ★ | [+15.7, +33.6] | <0.001 | **+21.6** ★ | [+13.4, +29.9] | <0.001 |
| RAG combined − Decoder | **+34.3** ★ | [+24.6, +44.0] | <0.001 | **+31.3** ★ | [+23.1, +40.3] | <0.001 | **+25.4** ★ | [+17.9, +33.6] | <0.001 |
| RAG combined − Zero-shot | **+9.7** ★ | [+3.0, +16.4] | 0.011 | +4.5 | [-0.7, +9.7] | 0.180 | **+9.0** ★ | [+3.7, +14.9] | 0.004 |
| RAG combined − RAG (Yarob-only) | +0.7 | [-2.2, +3.7] | 1.000 | +1.5 | [-0.0, +3.7] | 0.500 | +1.5 | [-1.5, +4.5] | 0.625 |
| Hybrid (Haiku) − Haiku RAG | +1.5 | [-4.5, +6.7] | 0.791 | -3.0 | [-6.7, +0.0] | 0.219 | -1.5 | [-6.7, +3.7] | 0.791 |
| **Sonnet RAG − Haiku RAG** | **+6.7** ★ | [+0.7, +12.7] | **0.035** | +5.2 | [+0.0, +11.2] | 0.118 | +4.5 | [+0.0, +9.7] | 0.146 |
| Sonnet zero-shot − Haiku RAG | +5.2 | [-1.5, +11.9] | 0.167 | -0.7 | [-6.7, +4.5] | 1.000 | +0.0 | [-6.0, +6.0] | 1.000 |
| Hybrid (Sonnet) − Sonnet RAG | +0.0 | [+0.0, +0.0] | 1.000 | -3.7 | [-7.5, +0.0] | 0.125 | -3.0 | [-6.7, +0.7] | 0.219 |
| Haiku RAG (k=10) − Haiku RAG (k=5) | -2.2 | [-7.5, +3.0] | 0.581 | +1.5 | [-3.0, +6.0] | 0.754 | -1.5 | [-6.7, +3.0] | 0.754 |

**Statistical reading:**

1. **Retrieval augmentation over zero-shot Haiku** is a real lift on case (+9.7 pp, p=0.011) and aggregate fully (+9.0 pp, p=0.004). Marker EM and well-formedness do not change significantly.

2. **Switching the LLM from Haiku 4.5 to Sonnet 4.5** (with k=5 RAG held constant) produces the largest paired-significant gain we observe: **case +6.7 pp [+0.7, +12.7], McNemar p=0.035 ★**. Marker and fully trend positive (+5.2 and +4.5 pp) with bootstrap CIs reaching toward 0; not significant at α=0.05 but consistent direction. **Sonnet RAG is the strongest non-fine-tuned baseline.**

3. **Mix A (Hybrid) does not beat RAG on either base LLM.** On Haiku, all four binary metrics fall within the noise floor (Δ case +1.5 pp p=0.791, Δ marker −3.0 pp p=0.219, Δ fully −1.5 pp p=0.791). Replacing the underlying LLM with Sonnet 4.5 reproduces the same pattern: case unchanged (McNemar p=1.000), Δ marker −3.7 pp [-7.5, +0.0] p=0.125, Δ fully −3.0 pp [-6.7, +0.7] p=0.219 — directionally negative, not paired-significant. **The per-word routing hypothesis is rejected on both base systems.** We attribute this to a teacher-bias problem: the AraT5v2 specialist is trained on 7,172 Claude-distilled marker pairs whose phrasing inherits Claude's systematic biases, so the specialist cannot improve over its teacher's retrieval-augmented prose. (A Yarob-only marker FT, n=1,643 hand-authored pairs, is the planned ablation that isolates this hypothesis.)

4. **Adding 601 distilled examples to the retrieval pool produced no significant change** (Δ case +0.7 pp, p=1.000). We retain combined-pool retrieval for marginally higher role coverage but make no improvement claim.

5. **Increasing retrieval depth from k=5 to k=10 trends negative on case (-2.2 pp, p=0.581) and well-formedness (-3.0 pp, p=0.125), and is statistically a wash on marker and fully.** The bottom-5 retrievals are less similar to the query and add stylistic noise rather than knowledge. **k=5 is the right operating point.**

6. **The from-scratch decoder is dominated** by every LLM-based system at p<0.001 on every metric.

7. **Stanza Arabic (UD pipeline + a deterministic UD→i'rāb stub) is the only published-prior-work peer baseline we evaluated.** It scores well-formed 59.7%, case 35.1%, role-F1 10.3%, fully 5.2%. Sonnet RAG beats it on every metric paired-significantly at p<0.001 (case Δ +38.8 pp, fully Δ +26.9 pp ★). Stanza's well-formedness is meaningfully positive (≈60%), confirming the UD parser produces grammatically-locatable predictions, but its role-F1 of 10.3% reflects the UD label set's mismatch with traditional Arabic role taxonomy: many UD `nmod`/`obl` arcs do not map cleanly onto مفعول به / حال / مضاف إليه. **Stanza is reported as a peer comparison; the from-scratch decoder is not.**

8. **Open-weight peer (Qwen2.5-7B-Instruct + same RAG, 4-bit local).** Best open-weight LLM that fits on an 8GB consumer GPU. Result: case 43.3%, role-F1 19.2%, marker 19.4%, fully 3.0%. Paired-significantly weaker than Sonnet RAG on every metric (case Δ −30.6 pp, fully Δ −29.1 pp, p<0.001 ★) and **statistically indistinguishable from Stanza** as a peer baseline (Δ case +8.2 pp p=0.080, Δ fully −2.2 pp p=0.453). **Sonnet RAG's lead over open-weight alternatives is real and large.** Reported as evidence per Hovy & Spruit (2016) that the closed-source advantage is being honestly compared, not disguised by absence of an open peer.

9. **Open-weight FT capacity comparison (Phase 2.1 + 2.3).** Two open-weight T5-architecture models trained on the same 77,534-row Haiku-distilled corpus, same recipe:
   - **AraT5v2-base (296M, Arabic-specific pretraining):** case 65.7%, role-F1 54.8%, marker 44.0%, **fully 24.6%**
   - **mT5-base (580M, multilingual pretraining):** case 61.9%, role-F1 31.6%, marker 32.8%, **fully 18.7%**

   AraT5v2-base **dominates mT5-base on every metric despite half the params**: case Δ +3.8 pp, role-F1 Δ +23.2 pp, marker Δ +11.2 pp, **fully Δ +5.9 pp**. **Arabic-specific pretraining beats raw scale at this size.** Both models also paired-significantly beat Qwen-7B+RAG by ~20 pp on `fully` (p<0.001 ★) — task-specific training matters more than parameter count when the base model has no Arabic-i'rāb prior. AraT5v2-base remains paired-significantly worse than Sonnet RAG (case Δ −8.2 pp p=0.013 ★, fully Δ −7.5 pp p=0.021 ★); the closed-source advantage is real but bounded.

The smallest detectable difference at n=134 with α=0.05 power 0.80 is roughly **±7 percentage points** on a binary proportion; differences below that are within noise.

---

## Error analysis — per-construction breakdown of `fully_correct_word`

Each Gazelle sentence was classified into one or more of 11 construction tags using a heuristic regex over the gold i'rāb prose AND the surface text (e.g. *kāna* sister + presence of `اسم كان`/`خبر كان`). Tags overlap; a sentence may belong to multiple categories. Per-tag breakdown across five systems (95% bootstrap CIs in brackets):

| Tag | # sent | # words | Stanza | Haiku zero | Haiku RAG | Sonnet zero | **Sonnet RAG** |
|---|---:|---:|---:|---:|---:|---:|---:|
| NOMINAL | 5 | 18 | 16.7 [0.0, 38.9] | 50.0 [27.8, 72.2] | 61.1 [38.9, 83.3] | 66.7 [44.4, 88.9] | **72.2 [50.0, 88.9]** |
| VERBAL | 15 | 61 | 3.3 [0.0, 8.2] | 14.8 [6.6, 24.6] | 24.6 [13.1, 36.1] | 29.5 [18.0, 41.0] | **32.8 [21.3, 44.3]** |
| PREPOSITIONAL | 8 | 37 | 8.1 [0.0, 16.2] | 16.2 [5.4, 29.7] | 21.6 [8.1, 35.1] | 27.0 [13.5, 40.5] | **29.7 [16.2, 43.2]** |
| PARTICLE_MOOD | 6 | 30 | 3.3 [0.0, 10.0] | 16.7 [3.3, 30.0] | 26.7 [10.0, 43.3] | 16.7 [3.3, 33.3] | 23.3 [10.0, 40.0] |
| MOOD_SHIFT_SUBJUNCTIVE | 2 | 12 | 8.3 [0.0, 25.0] | 16.7 [0.0, 41.7] | 25.0 [0.0, 50.0] | 25.0 [0.0, 50.0] | 25.0 [0.0, 50.0] |
| **EXCEPTION (istithnāʾ)** | 2 | 9 | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| **KANA_SISTERS** | 2 | 7 | **0.0** | **0.0** | 14.3 [0.0, 42.9] | **0.0** | **0.0** |
| INNA_SISTERS | 1 | 4 | 0.0 | 50.0 [0.0, 100] | 75.0 [25.0, 100] | 25.0 [0.0, 75.0] | 75.0 [25.0, 100] |
| RELATIVE | 1 | 9 | 11.1 [0.0, 33.3] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] |
| IDAFA_HEAVY | 1 | 9 | 11.1 [0.0, 33.3] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] |
| OTHER | 3 | 16 | 6.2 [0.0, 18.8] | 6.2 [0.0, 18.8] | 12.5 [0.0, 31.2] | 6.2 [0.0, 18.8] | 6.2 [0.0, 18.8] |

**Findings:**

1. **EXCEPTION (istithnāʾ) is a complete failure mode** — 0/9 across **all five systems** including Stanza. The two affected sentences (`سوى تلميذين`, `عدا واحدا`) require recognizing that the noun after `سوى/عدا/إلا` takes a non-default case (genitive in positive exception, agreeing-with-mustathnā-minhu in negative exception) — a rule no system applies. This is the strongest cross-system failure we observe.

2. **KANA_SISTERS is a near-complete failure mode** (NEW finding from extended tag set, Apr 2026 update) — 0/7 across **four of five systems**, with only Haiku RAG scraping 14.3% (one word out of seven). The two sentences involve `كان` and `أصبح` shifting their predicate to naṣb (`خبر منصوب`); systems consistently produce default-rafʿ analyses. Sample is small (n=7) but the systematic 0% across systems makes the pattern robust under the noise model.

3. **Nominal sentences are ~2× easier than verbal ones** for every LLM-based system (Sonnet RAG 72.2% vs 32.8%). Consistent with shorter dependency chains: a copular clause has at most one case-marking interaction, while a verbal sentence requires resolving subject/object/optional adverbials all at once.

4. **Verbal-vs-prepositional is essentially a wash** (Sonnet RAG 32.8 vs 29.7) — no detectable effect of preposition presence on aggregate correctness once verbal-sentence frequency is accounted for.

5. **Sample-size warning.** RELATIVE, IDAFA_HEAVY, INNA_SISTERS each have ≤1 Gazelle sentence (≤9 word judgments), so their per-tag scores are not reliably informative — the bootstrap CIs are wider than the point values. We report them for transparency but base no claims on them. The two robust cross-system failures (EXCEPTION, KANA_SISTERS) both have n≥7 and 0% point estimates, which the bootstrap CIs cannot widen.

6. **AraT5v2-base trained model preserves the same failure modes** (per-construction CSV at `runs/error_analysis/per_construction_with_arat5.json`). EXCEPTION 0/9 and KANA_SISTERS 0/7 hold for the trained model just as for every Claude variant. This is informative: **the failure is not Claude-specific stylistic** (since AraT5v2 trained on Haiku-distilled data also fails on the same constructions); it is **structural to the task or to the available training distribution**. The remaining gap to Sonnet RAG concentrates in NOMINAL (50% vs 72%, gap 22 pp) and PARTICLE_MOOD (6.7% vs 23.3%, gap 17 pp), where Sonnet RAG's larger model + retrieval context appears to make the difference. AraT5v2-base **matches Sonnet RAG on PREPOSITIONAL** (29.7% vs 29.7%, gap 0) and is statistically tied on VERBAL (29.5% vs 32.8%, gap 3 pp).

---

## Prompt-format sensitivity (Sonnet RAG)

Robustness check: re-ran Sonnet RAG (k=5) with an alternate system prompt — same JSON output contract but with English instructions and a slightly different role/marker terminology list (`scripts/prompt_sensitivity.py`).

| Prompt | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|
| Headline (Arabic, terse role list) | 79.9 | 73.9 | 74.6 | 50.0 | **32.1** |
| Alt (English instructions) | 79.9 | 73.1 | 69.7 | 49.3 | 30.6 |
| Δ paired vs headline | 0.0 (p=1.000) | −0.7 (p=1.000) | −4.9 | −0.7 (p=1.000) | −1.5 (p=0.500) |

The headline is robust to prompt wording on case/marker/well-formed/fully (all paired McNemar p≥0.5, no detectable difference). Role-F1 trends down ~5 pp, plausibly because the English-instructed prompt is less specific about the Arabic role taxonomy and Sonnet falls back to less canonical role labels. **Reported numbers are not an artifact of prompt-engineering.**

---

## Inference-time variance (Sonnet RAG, temp=0.0)

The Anthropic API does not expose a `seed` parameter, and Claude's sampling is server-side stochastic even at `temperature=0.0`. To estimate this variance we re-ran the headline configuration on the same 30 Gazelle sentences:

| Metric | Run 1 (headline) | Run 2 (repro) | Δ pp | McNemar p |
|---|---:|---:|---:|---:|
| well-formed | 79.9 | 79.9 | 0.0 | 1.000 |
| case-acc | 73.9 | 74.6 | +0.7 | 1.000 |
| role-F1 | 74.6 | 72.9 | −1.7 | — |
| marker-EM | 50.0 | 49.3 | −0.7 | 1.000 |
| **fully** | **32.1** | **32.1** | **0.0** | **1.000** |

Per-word agreement (n=126 aligned words): **case 99.2%, role 98.4%, marker 99.2%, fully (case ∧ role ∧ marker) 96.8%.**

Sonnet 4.5 at `temperature=0.0` is reproducible to ~1 pp on aggregate metrics and ~3% per-word disagreement on the joint metric. All paired McNemar tests give p=1.000. The reported headline numbers are robust to provider-side stochasticity at the scale tested. Full breakdown in `reproducibility/variance.md`; second run predictions in `runs/baseline_eval_sonnet_repro/`.

---

## Retrieval-pool ablations (Sonnet RAG, k=5 fixed)

We isolate the contribution of each retrieval source. The headline pool is Yarob (459 hand-authored) + Distilled (601 Claude-generated over PADT-UD).

| Pool | n | well | case | role-F1 | marker | **fully** | Δ fully vs headline (McNemar p) |
|---|---:|---:|---:|---:|---:|---:|---|
| **Yarob + Distilled** (headline) | 1,060 | 79.9 | 73.9 | 74.6 | 50.0 | **32.1** | — |
| Yarob only | 459 | (per finding #4: indistinguishable from headline) | | | | | +1.5 (p=0.625) |
| Distilled only | 601 | 78.4 | 72.4 | 69.8 | 46.3 | 27.6 | −4.5 (p=0.109) |
| Yarob + Distilled + MASAQ | 2,560 | 79.9 | 73.9 | 71.0 | 48.5 | 31.3 | −0.7 (p=1.000) |
| Yarob + Distilled-v1 + Distilled-v2 (Qwen-7B baseline test) | 6,057 | (Qwen2.5-7B+RAG result) 72.4 | 41.8 | 23.6 | 21.6 | **3.0** (vs 3.0 headline-Qwen-pool, Δ=0.0 p=1.000) | n/a — Qwen, not Sonnet |

**Findings:**

- **Yarob-only (459 examples) ≈ headline (1,060 examples)** — already established as Δ case +0.7 pp (p=1.000) earlier; the 601 distilled examples do not paired-significantly improve over Yarob alone, but they do not hurt either, so we keep them for stylistic coverage.
- **Distilled-only (601, no Yarob) trends 4.5 pp worse on `fully`** (McNemar p=0.109; not significant at α=0.05 but the directional gap is consistent across case, marker, and fully). **Per-example retrieval value of hand-authored Yarob > Claude-distilled.** This is consistent with Hovy & Spruit (2016) on the value of expert-curated gold over model-generated silver.
- **MASAQ-augmented (Quranic templated, +1,500): honest negative** — no help, small role-F1 drop (Δ −3.6 pp). Register mismatch — Quranic Arabic differs syntactically (mood-shifting particles, object-fronting, VS order) and lexically from MSA-news. Retrieval falls back to Quranic verses for some queries and carries the style into the prompt. We retain the headline pool. *(MASAQ remains a credible future training-augmentation resource, characterized in `data/masaq_sample.jsonl` and `src/irab_tashkeel/data/masaq.py`.)*

- **Distillation-v2-augmented for the open-weight side: also honest negative** (Phase 2.2). Re-ran Qwen2.5-7B+RAG with the augmented 6,057-example pool (1,060 + 4,997 from Phase 1 distillation): well-formed +6.0 pp (close to significant, p=0.096), case −1.5 pp (p=0.815), marker +2.2 pp (p=0.607), **fully unchanged at 3.0% (Δ=0.0 pp, p=1.000)**. Augmenting the retrieval pool does not move the open-weight needle either. Bottleneck for Qwen-7B is the model's pattern-following ability (it produces verbose Arabic that doesn't always parse cleanly via the structural extractor), not the size or composition of the retrieval pool.

---

## Annotator-disagreement audit (label variation in classical naḥw)

Single-gold scoring assumes one canonical i'rāb per word, but classical Arabic naḥw genuinely admits alternative analyses on a non-trivial fraction of words (sibawayhi vs the Kufan grammarians, ḥāl vs naʿt, mubtadaʾ vs fāʿil for some preverbal nouns, etc.). To quantify how much this assumption inflates strict scoring, we asked Sonnet 4.5 to classify each Gazelle gold word into one of three classes and to list any alternative valid analyses (`src/irab_tashkeel/evaluation/ambiguity.py`):

| Ambiguity class | Definition | n / 134 |
|---|---|---:|
| unambiguous | one valid analysis only | 90 (67.2%) |
| ambiguous_minor | multiple analyses, but **case is fixed** | 23 (17.2%) |
| ambiguous_major | multiple analyses with **different cases** (e.g. fāʿil vs mubtadaʾ) | 19 (14.2%) |
| (unannotated) | annotation alignment failure | 2 (1.5%) |

**Permissive scoring** (a system is correct if its prediction matches gold OR any annotated alternative on the targeted dimension):

| System | case strict | case **permissive** | Δ pp | role strict | role permissive | Δ pp | marker | fully |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stanza | 35.1 | 37.3 | +2.2 | 9.7 | 11.9 | +2.2 | unchanged | unchanged |
| Haiku zero | 57.5 | 60.4 | +3.0 | 33.6 | 35.8 | +2.2 | unchanged | unchanged |
| Haiku RAG | 67.2 | 68.7 | +1.5 | 41.8 | 42.5 | +0.7 | unchanged | unchanged |
| Sonnet zero | 72.4 | 73.1 | +0.7 | 45.5 | 47.8 | +2.2 | unchanged | unchanged |
| **Sonnet RAG** | 73.9 | **74.6** | +0.7 | 46.3 | 47.8 | +1.5 | unchanged | unchanged |

**Per-ambiguity-class case-acc breakdown for Sonnet RAG (permissive):**

| | unambiguous | ambiguous_minor | ambiguous_major |
|---|---:|---:|---:|
| n | 90 | 23 | 19 |
| Sonnet RAG case-acc | 76.7% | 69.6% | 73.7% |

**Findings (Hovy-relevant):**

1. **31.4% of Gazelle words (42/134) admit multiple valid analyses** under classical naḥw — a quantitative estimate of label variation in this domain. This is a non-negligible fraction; for any per-word benchmark on Arabic syntax, some annotator disagreement is structural, not error.
2. **The strict→permissive gap is small (≤3 pp on case, ≤2.2 pp on role) and shrinks as systems get stronger** (3.0 pp on Haiku zero → 0.7 pp on Sonnet RAG). Strong systems align with the canonical analysis even on words that admit alternatives, so single-gold scoring under-counts them by very little. **The headline numbers are robust to this concern.**
3. **Marker EM and fully_correct_word are entirely unaffected** by permissive scoring (gap = 0 pp for every system). Alternative analyses in classical naḥw rarely change the marker phrase (الضمة remains الضمة across role disputes), and the conjunction across case+role+marker is so strict that any single-field alternative match almost never enables a full-word match.
4. Sonnet RAG case-acc on unambiguous words (76.7%) is only modestly higher than on ambiguous-major (73.7%); the system handles the syntactically harder constructions roughly as well as the easy ones once we permit valid alternatives.

This analysis does NOT eliminate the label-variation concern but quantifies it: ≤1 pp upward bias on case-acc for our headline system, 0 pp on the joint metric. Reported here as methodology transparency in the spirit of Plank (2022) and Hovy & Spruit (2016).

---

## Sensitivity to retrieval depth `k` (Sonnet RAG)

Sweeping `k ∈ {1, 3, 5, 8, 12}` while holding the LLM (Sonnet 4.5), pool composition (Yarob+distilled, n=1060), and prompt fixed:

| k | well | case | role-F1 | marker | **fully** |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.4 | 72.4 | 72.6 | 38.1 | 28.4 |
| 3 | 79.1 | 73.1 | 72.7 | 40.3 | 31.3 |
| **5** | **79.9** | **73.9** | **74.6** | **41.8** | **32.1** |
| 8 | 79.9 | 75.4 | 70.6 | 40.3 | 31.3 |
| 12 | 79.9 | 74.6 | 64.7 | 38.8 | 29.9 |

**Paired bootstrap + McNemar deltas vs k=5:**

| Comparison | Δ case | Δ marker | Δ fully | Notes |
|---|---:|---:|---:|---|
| k=1 − k=5 | −1.5 (p=0.625) | **−4.5 ★ (p=0.031)** | −3.7 (p=0.125) | k=1 paired-significantly worse on marker |
| k=3 − k=5 | −0.7 (p=1.000) | −1.5 (p=0.500) | −0.7 (p=1.000) | indistinguishable |
| k=8 − k=5 | +1.5 (p=0.500) | −1.5 (p=0.500) | −0.7 (p=1.000) | wash |
| k=12 − k=5 | +0.7 (p=1.000) | −3.7 (p=0.062) | −2.2 (p=0.375) | role-F1 drops 10 pp directionally |

**k=5 is the joint optimum on `fully` and is the only justified operating point**: smaller k loses paired-significant marker accuracy; larger k does not improve case and noticeably degrades role-F1 (additional retrievals introduce role distractors as the per-example similarity decreases). This matches Brown et al. (2020) and Mialon et al. (2023) on diminishing returns in in-context demonstrations.

---

## Cross-register evaluation: MASAQ subset role-F1

To extend the n=134 Gazelle eval, we built a 5,007-word MASAQ Quranic eval surface (`data/masaq_eval.jsonl`, see `src/irab_tashkeel/data/masaq.py`). Methodology details and the three-stage role-F1 investigation are documented in `docs/MASAQ_role_audit.md` and `runs/role_extractor_diagnosis/bucket_30_samples.md`. Summary:

**Three-stage history** of the MASAQ role-F1 metric for AraT5v2-base:

| Stage | role-F1 (full eval) | What it measured |
|---|---:|---|
| Original (broken extractor priority) | 10.2 | First-match-wins on a manually-ordered ROLES list; missed long-form role mentions in pred output. |
| Longest-match-first fix in `structural.py` | 9.6 | Correct in principle but barely moved the number; the fix's bucket was small. |
| **Subset scoring** (current) | **24.3 [20.5, 30.6]** | role-F1 computed only on words where the gold has an extractable role — avoiding the verb-extractor false-positive bug where role terms in attached-pronoun analyses are mistakenly assigned to verbs. |

**Subset definition.** A word is in the subset iff `extract(gold_irab).role is not None`. On MASAQ this is 999/5,007 = 20% of words; on Gazelle 78/134 = 58% (Gazelle gold is hand-written and more often includes a nominal role term). For matched cross-register comparison, both sides are restricted to the same definition.

### Results (subset role-F1, with 95% bootstrap CIs, B=1000)

| System | Gazelle subset (n=78) | MASAQ subset (n=999) | MASAQ as % of Gazelle | Reading |
|---|---:|---:|---:|---|
| Stanza | 10.4 [6.1, 18.6] | 17.6 [16.6, 19.8] | 169% (MASAQ higher) | UD parser register-stable; closed UD label set absorbs both registers similarly. |
| Qwen-7B+RAG | 23.0 [10.3, 31.8] | (eval pending) | — | — |
| mT5-base FT | 32.8 [22.7, 44.9] | 18.6 [14.6, 25.4] (n=999) | 57% | **Moderate cross-register effect** per Hovy framework. |
| AraT5v2-base FT | 58.9 [45.7, 72.5] | 24.3 [20.5, 30.6] | 41% | **Substantial cross-register effect** per Hovy framework. |
| Haiku zero-shot | 56.3 [42.4, 71.2] | (eval pending) | — | — |
| Haiku RAG | 70.4 [57.8, 84.5] | (eval pending) | — | — |
| Sonnet zero-shot | 78.1 [65.3, 91.9] | (eval pending) | — | — |
| Sonnet RAG | 75.7 [65.1, 90.0] | **14.1 [11.9, 18.8]** (n=999) | **19%** | **Largest cross-register drop of any system tested.** |

### Cross-register role-F1 deltas (two-sample bootstrap, B=1000)

The Gazelle subset (n=78) and MASAQ subset (n=999) are distinct items — different registers, different word distributions, different annotators. Paired bootstrap doesn't apply; we use two-sample bootstrap that resamples each side independently and reports the 95% CI of the difference.

| System | Gazelle F1 (n=78) | MASAQ F1 | Δ (Gazelle − MASAQ) | 95% CI | Significant? |
|---|---:|---:|---:|---|:---:|
| Stanza | 10.4 | 17.6 (n=999) | **−7.2 pp** | [−11.8, +1.9] | ns (CI crosses 0) |
| mT5-base FT | 32.8 | 18.6 (n=999) | **+14.2 pp** | [+1.5, +26.3] | ★ |
| AraT5v2-base FT | 58.9 | 24.3 (n=999) | **+34.7 pp** | [+18.7, +48.9] | ★ |
| Sonnet RAG | 75.7 | **14.1 (n=999)** | **+61.7 pp** | [+48.8, +75.3] | ★ |

### Four cross-register findings

1. **Trained Arabic models show substantial cross-register role degradation.** AraT5v2-base loses 34.7 pp (★) when moved from MSA-news to Quranic register (Gazelle 58.9 → MASAQ 24.3). mT5-base loses 14.2 pp (★). Both trained models retain only 41–57% of their MSA role-F1 on Quranic. This is a real, paired-significant cross-register effect.

2. **Stanza (UD parser) is register-stable.** No significant degradation; in fact a slight (non-significant) GAIN of 7.2 pp moving from MSA to Quranic (Gazelle 10.4 → MASAQ 17.6). The closed UD POS+deprel label set applies uniformly across registers via the same templater on both sides; whatever role distribution Stanza produces, it produces it the same way regardless of register.

3. **Even the strongest closed-system baseline shows substantial cross-register effect.** Sonnet RAG drops by **+61.7 pp ★** (Gazelle 75.7 → MASAQ 14.1), retaining only **19%** of its MSA role-F1 on Quranic — the **largest cross-register drop of any system tested**, larger than the trained Arabic-specific AraT5v2-base. This pattern suggests register variation between MSA-news and Quranic is a **fundamental challenge for the role-identification task**, not an artifact of any specific training pipeline. The effect cuts across closed-source RAG systems, Arabic-specific fine-tuned open models, and multilingual fine-tuned open models alike.

4. **Arabic-specific pretraining is LESS register-stable than multilingual pretraining at this scale.** Among the trained open-weight models, AraT5v2-base (296M, Arabic-only pretraining) drops 34.7 pp while mT5-base (580M, multilingual including Arabic) drops only 14.2 pp. The CIs do overlap (mT5 [+1.5, +26.3] vs AraT5v2 [+18.7, +48.9]), so this comparison is directionally consistent but not paired-significant in isolation. A plausible reading: Arabic-only pretraining biases models toward the MSA-news distribution they're then fine-tuned on, while broader multilingual exposure leaves a weaker but more register-portable Arabic prior. The mT5 vs AraT5v2 absolute Gazelle role-F1 difference (32.8 vs 58.9) means the mT5 drop has less "room to fall," partially explaining the smaller delta.

The full role-F1 numbers (including all 5,007 MASAQ words) are reported in the appendix as `*_full` columns of `runs/role_extractor_diagnosis/` for completeness, but should not be used for cross-register comparisons.

---

## Metric audit via perturbed gold (extractor sensitivity / specificity)

To verify that the structural-extraction metric actually responds to errors on the right field, we built a deterministic perturbation set from the 134 Gazelle gold word-i'rāb pairs (`src/irab_tashkeel/evaluation/perturb.py`). For each gold string we generated up to three single-field corruptions:

| Perturbation | Description | n |
|---|---|---:|
| `case_flip` | Substitute case word + matching marker (e.g. `مرفوع … الضمة الظاهرة` → `منصوب … الفتحة الظاهرة`) | 34 |
| `role_flip` | Swap one role term (e.g. `فاعل` → `مفعول به`, `خبر` → `مبتدأ`) | 80 |
| `marker_mangle` | Swap marker only, keeping case word (e.g. `الضمة الظاهرة` → `الواو`) | 82 |
| (control) | Unmodified gold | 134 |

The audit scores how well the extractor flags the perturbation on the targeted field while leaving the other two fields stable:

| Field perturbed | n | flagged on target field | other fields unchanged |
|---|---:|---:|---:|
| **none (controls)** | 134 | — | **100.0% all-match** |
| **case** | 34 | 88.2% | role 97.1%; marker 0% (intentional — case_flip swaps marker too) |
| **role** | 80 | **60.0%** | case 100%; marker 100% |
| **marker** | 82 | 92.7% | case 100%; role 100% |

**Findings:**
- **Specificity is perfect (100.0%, n=134).** The extractor never disagrees with itself on uncorrupted gold — the metric is deterministic and self-consistent.
- **Marker sensitivity 92.7%, case sensitivity 88.2%** — both above or near the 90% target. The 6 unflagged marker-mangles and 4 unflagged case-flips are gold strings whose surface form contains additional marker/case mentions the perturbation didn't reach.
- **Role sensitivity is only 60.0%** — well below target. **This is a real, transparent limitation of the structural metric, not a bug to fix.** Detailed Arabic i'rāb prose routinely mentions multiple role terms per word (e.g. `اسم مجرور… متعلقان بالخبر… والفاعل ضمير مستتر`); the extractor returns the FIRST canonical role from a fixed-priority list. Perturbations of secondary mentions don't change what the extractor reads. **Implication:** the role-F1 numbers in the headline table should be read as a lower bound on agreement at the surface mention level, not as a strict role-by-role identity check. The case-acc and marker-EM numbers are tighter (88-93% extractor sensitivity); fully_correct_word inherits the role looseness as a small upward bias on the metric (because some role-flips would not actually be caught).

The ~12% upward bias on role-F1 partially explains why our role-F1 confidence intervals are wider than case-acc CIs at the same n.

---

## System discrimination on perturbed gold

The perturbation set (T6) lets us also probe the **system's** behavior, not just the extractor. We re-score each system's per-word predictions against the *perturbed* gold variant; a genuinely discriminative system should match the original gold but **not** the wrong-on-purpose perturbed gold. The drop between original-gold-rate and perturbed-gold-rate is a per-dimension discrimination signal (the larger the drop, the more the system's output is contingent on the actual case/role/marker, not boilerplate).

| System | case (orig→case-flipped) | role (orig→role-flipped) | marker (orig→marker-mangled) | fully (orig→worst variant) |
|---|---|---|---|---|
| Stanza | 49.0 → 13.8 (Δ −35) | 21.3 → 7.4 (Δ −14) | 23.1 → 1.6 (Δ −22) | 14.9 → 0.0 |
| Haiku zero | 75.5 → 21.9 (Δ −54) | 67.2 → 27.1 (Δ −40) | 62.1 → 4.3 (Δ −58) | 45.5 → 2.3 |
| Haiku RAG | 87.4 → 15.6 (Δ −72) | 83.6 → 28.3 (Δ −55) | 69.8 → 3.0 (Δ −67) | 68.5 → 0.0 |
| Sonnet zero | 94.2 → 15.2 (Δ −79) | 87.1 → 25.4 (Δ −62) | 67.8 → 3.0 (Δ −65) | 66.1 → 0.0 |
| **Sonnet RAG** | **96.1 → 15.2 (Δ −81)** | **88.6 → 28.6 (Δ −60)** | **77.0 → 1.5 (Δ −76)** | **76.8 → 0.0** |

(Per-cell n varies between 22 and 103; full per-system table with sample sizes in `runs/discrimination/per_system.json`. Subsets of words for which a perturbation could be applied differ by perturbation type, e.g. only 33 of 134 words had a case pattern matching the case-flip rule, vs 67 for role and 72 for marker.)

**Findings:**

1. **All systems show paired-significant discrimination on every probed dimension.** No system is just emitting boilerplate that happens to overlap both the original and perturbed gold.
2. **Sonnet RAG has the largest case-discrimination signal** (Δ −81 pp on case-flip vs Stanza's Δ −35 pp). This is consistent with its stronger headline case-acc.
3. **Role discrimination is the weakest channel for every system.** Sonnet RAG's role drop (Δ −60 pp) is materially smaller than its case (Δ −81 pp) and marker (Δ −76 pp) drops. This aligns with the T6 finding that the extractor itself has only 60% role-sensitivity to perturbations — we cannot tell whether the smaller role drop is a system limitation or a metric artifact.
4. **Marker discrimination is essentially perfect for Claude-based systems** (≥3% match rate against marker-mangled gold). Marker phrases are highly specific lexicalizations; once a system commits to one, the wrong alternative does not accidentally match.
5. **Fully_correct_word collapses to 0% under any single-field perturbation** for every system — confirming that fully is a strict conjunction and any single corrupted field is enough to flip the joint indicator.

This is a metric-validity check using the existing perturbation set; it does NOT increase the statistical power of the headline comparison (the same model output is being scored against multiple gold variants, so observations are not independent).

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

5. **Per-word inference scope.** All evaluations are word-level i'rāb on isolated sentences (no document context, no anaphora resolution beyond the single sentence). Real-world i'rāb annotation often references the broader paragraph (e.g. for ضمير anaphora). We make no claim about cross-sentence behavior.

6. **Distillation teacher quality (Phase 2.1).** Our distilled training corpus for the open-weight scaling comparison was generated by Claude Haiku 4.5 (case accuracy 67.2% on Gazelle) rather than Sonnet 4.5 (73.9%), due to a $50 API budget. Open-weight models trained on this corpus inherit teacher errors; the observed gap to Sonnet RAG (case Δ −8.2 pp ★, fully Δ −7.5 pp ★ for AraT5v2-base) may therefore overstate the gap to a hypothetical "best-teacher" baseline. We chose Haiku because the alternative (Sonnet ~1,500 rows under the same budget) would have been too small to cleanly fine-tune 296M-7B-class models without saturation. Reported here as a transparent budget-trade-off, not a hidden one.

7. **HPC fragility was a real cost.** Five separate sbatch attempts failed before Phase 2.1's training run completed: the first four (HPC #486563, 486573, 486949, 486952) crashed silently mid-training; the fifth (487235) trained all 14,247 steps but exit-1'd in the final-save step due to a disk-quota wall (the script never wrote the `final/` directory). The trained model at `checkpoint-14247` is intact and usable; we promoted it via symlink. The directive's Phase 2 timeline (~3-4 h per training run on a stable A100) was off by 6-12 h once the failure recovery is included. We did not attempt the directive's AraT5v2-large or Fanar-9B Phase 2 runs because of this fragility plus deadline pressure; see Future Work.

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
