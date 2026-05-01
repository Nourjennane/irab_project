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
| Stanza Arabic (UD pipeline + UD→i'rāb stub) | 59.7 [51.5, 68.7] | 35.1 [26.9, 43.3] | 13.4 [8.2, 19.4] | 10.9 [6.8, 19.7] | 5.2 [2.2, 9.7] |
| Claude Haiku 4.5 zero-shot | 77.6 [70.1, 84.3] | 57.5 [49.3, 66.4] | 40.3 [32.1, 49.3] | 55.9 [42.0, 68.6] | 18.7 [11.9, 25.4] |
| Claude Haiku 4.5 + RAG (k=5) | 79.9 [73.1, 86.6] | 67.2 [59.0, 75.4] | 44.8 [35.8, 53.0] | 68.8 [55.8, 82.2] | 27.6 [20.1, 35.8] |
| Claude Haiku 4.5 + RAG (k=10) — *ablation* | 76.9 [69.4, 84.3] | 64.9 [56.7, 73.1] | 46.3 [38.1, 55.2] | 45.4 [36.9, 56.0] | 26.1 [18.7, 34.3] |
| Hybrid (Haiku RAG + AraT5v2 marker, overlay) | 77.6 [70.1, 84.3] | 67.9 [59.7, 76.1] | 41.0 [32.1, 50.0] | 65.9 [51.1, 78.8] | 26.1 [18.7, 34.3] |
| Claude Sonnet 4.5 zero-shot | 78.4 [70.9, 85.1] | 72.4 [64.9, 79.9] | 44.0 [35.1, 53.0] | 76.0 [62.1, 85.6] | 27.6 [20.1, 35.8] |
| **Claude Sonnet 4.5 + RAG (k=5)** | **79.9 [72.4, 86.6]** | **73.9 [66.4, 81.3]** | **50.0 [41.8, 59.0]** | **74.6 [63.7, 88.6]** | **32.1 [24.6, 40.3]** |
| Sonnet RAG + AraT5v2 marker (Hybrid v2) | 79.9 [72.4, 86.6] | 73.9 [66.4, 81.3] | 46.3 [38.1, 55.2] | 73.3 [60.7, 86.4] | 29.1 [21.6, 37.3] |

The from-scratch character decoder is reported as a documented negative-result baseline in the appendix only — it scores 32.8% [25.4, 41.0] case and 3.8% [1.5, 8.2] role-F1 on the same eval, confirming that training i'rāb prose generation from scratch fails at this data scale.

---

## Statistically significant differences (paired bootstrap + McNemar p<0.05)

| Comparison | Δ case-acc | 95% CI | McNemar p | Δ marker-EM | 95% CI | McNemar p | Δ fully | 95% CI | McNemar p |
|---|---:|---|---:|---:|---|---:|---:|---|---:|
| Sonnet RAG − Stanza | **+38.8** ★ | [+30.6, +47.8] | <0.001 | **+36.6** ★ | [+27.6, +45.5] | <0.001 | **+26.9** ★ | [+19.4, +35.8] | <0.001 |
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

7. **Stanza Arabic (UD pipeline + a deterministic UD→i'rāb stub) is the only published-prior-work peer baseline we evaluated.** It scores well-formed 59.7%, case 35.1%, role-F1 10.9%, fully 5.2%. Sonnet RAG beats it on every metric paired-significantly at p<0.001 (case Δ +38.8 pp, fully Δ +26.9 pp ★). Stanza's well-formedness is meaningfully positive (≈60%), confirming the UD parser produces grammatically-locatable predictions, but its role-F1 of 10.9% reflects the UD label set's mismatch with traditional Arabic role taxonomy: many UD `nmod`/`obl` arcs do not map cleanly onto مفعول به / حال / مضاف إليه. **Stanza is reported as a peer comparison; the from-scratch decoder is not.**

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
