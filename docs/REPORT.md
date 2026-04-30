# Per-Word Arabic I'rāb Generation: A Comparison Study with a Routing-Based Hybrid

*Working draft — final ~3 pages. Word budget per section in [brackets]; current draft in *italics* where complete, [TODO] where pending data.*

**Status legend:** ✅ drafted · 🟡 partial · ⏳ depends on Mix A result · ❌ not started

---

## Abstract — [TODO ~150 words] ⏳

*Will be written last. Single paragraph. Frames the contribution as comparison study + routing-based hybrid + structural metrics, with one headline number from the strongest system on the manually-curated benchmark (or Gazelle if that benchmark is dropped).*

---

## 1. Introduction — [TODO ~250 words] 🟡

*Open with the i'rāb problem statement: given an undiacritized Arabic sentence, produce per-word traditional grammatical analysis as Arabic prose, including diacritization. Cite Gazelle (Hijjawi et al. 2024) for prior framing of i'rāb-as-LLM-task. State that traditional Arabic grammatical analysis is structured prediction over a closed taxonomy (4 cases × ~25 roles × ~20 markers) but expressed as natural-language prose — making it a hybrid surface-form/structured-output problem. Note that prior approaches either (a) train task-specific classifiers on per-tag annotations, or (b) prompt frontier LLMs zero-shot. Neither directly addresses the asymmetry we observe: the "knowledge" needed for case and role assignment is captured by pretrained Arabic LLMs, but the "style" of writing the marker phrase ("الضمة الظاهرة على آخره" vs "الضمة" vs "الضمة المقدرة على الألف") is fitting to a corpus-specific surface convention.*

**Contributions.** Two:

1. **A per-word routing hybrid (Mix A) for Arabic i'rāb generation.** We decompose the structured prediction problem into a knowledge-bound part (case, role) handled by retrieval-augmented Claude Haiku 4.5 and a style-bound part (marker phrasing) handled by a fine-tuned 296M-parameter AraT5v2 specialist. To our knowledge this specialist+generalist routing applied to Arabic morphological generation — with a frontier LLM as the syntactic-knowledge component and a small specialist as the surface-form-fitting component — has not been previously evaluated. [⏳ Hybrid headline pending FT.]

2. **A statistically-grounded evaluation methodology for open-generation Arabic morphological analysis.** We avoid string-similarity metrics (chrF/BLEU give partial credit to wrong-case outputs); instead we extract structured fields ({pos, case, role, marker}) by regex and report (i) per-system metrics with 95% percentile bootstrap CIs and (ii) system-vs-system deltas with paired bootstrap and McNemar's exact test on matched word judgments. We additionally provide a per-construction error analysis (verbal/nominal/iḍāfa/prepositional/particle-mood/exception) that reveals **istithnāʾ (exception) as a complete failure mode** across all evaluated systems including the strongest.

---

## 2. Related Work — [drafted ~280 words] ✅

**Arabic morphosyntactic analysis tools.** Traditional Arabic grammatical analysis tooling is dominated by classifier-pipeline systems: CamelParser2.0 (Elshabrawy et al., 2023) wraps a CAMeL Tools BERT disambiguator with a SuPar biaffine parser to produce CATiB-format dependency trees with rich morphological features; MADAMIRA (Pasha et al., 2014) and Farasa (Abdelali et al., 2016) target similar surface representations. These systems output structured POS + dependency annotations, **not the prose-form i'rāb prose we target**. Their natural use as a baseline requires a templating layer from CATiB (or UD) features into the traditional i'rāb register; we evaluate this externally where feasible (§6).

**i'rāb-specific datasets and benchmarks.** The richest expert-annotated resource is **MASAQ** (Sawalha et al., 2025), a 131K-morphological / 123K-syntactic-entry annotation of the entire Quran with a 72-tag i'rāb scheme, released open-license. We do not use MASAQ for training in the present work because its Classical Arabic vocabulary distribution does not match our MSA test set (Gazelle), but it is the natural target for a future extension; we discuss this in §9. **Gazelle** (Hijjawi et al., 2024) provides the manually-curated MSA i'rāb evaluation set we adopt; their original methodology focuses on closed multiple-choice probing of LLM grammatical knowledge, while we evaluate open-text generation with structural-extraction metrics.

**Arabic LLMs and small-model morphology.** Our specialist's base, AraT5v2-base-1024 (Nagoudi et al., 2023), is a continued-pretrained Arabic T5 with strong sub-word coverage. The methodological precedent for our setting is **Sadeed** (Aldallal et al., 2025), which demonstrates that a small Arabic-specialist model fine-tuned on carefully filtered targets achieves state-of-the-art on MSA diacritization — we transfer the recipe to marker prediction. Anh et al. (2024) report on LLM-based Arabic morphosyntactic tagging and find that frontier LLMs match or exceed task-specific classifiers given enough demonstrations, motivating our retrieval-augmented baseline.

**System combination and routing.** Decomposing structured prediction into specialist sub-models is well-studied in machine translation system combination (Bangalore et al., 2001; Rosti et al., 2007) and in NER ensembling. To our knowledge, **specialist–generalist routing applied to Arabic morphological generation — frontier LLM for syntactic knowledge plus small fine-tuned specialist for surface-form fitting — has not been previously evaluated**. Our negative result on this routing decomposition (§6.2) is a contribution to the empirical literature on when such decomposition does and does not help.

A 2025 comparative study of Arabic syntactic parsers (Frontiers in AI, August 2025) places these tools on a common benchmark; Arabic-DeepSeek-R1 (2026) recently demonstrated reasoning-trace distillation as a path to LLM-based Arabic-grammar improvements — both relevant pointers for future iterations of this line of work.

---

## 3. Data — [drafted ~350 words] ✅

We assemble three categories of i'rāb data, distinguished by their level of human authorship:

**Manual gold (human-authored prose).** *Yarob* (linuxscout/yarob, GPL) provides 459 sentences of MSA i'rāb in classical Arabic-grammar prose style. *Gazelle* (Hijjawi et al. 2024) provides 30 hand-written sentences specifically designed as an i'rāb evaluation set. We use Gazelle exclusively as held-out evaluation; Yarob is used as the retrieval pool for our RAG baseline.

**LLM-distilled (silver).** We use Claude Haiku 4.5 as a teacher to generate per-word i'rāb for 601 modern Arabic news sentences sampled from UD_Arabic-PADT (length 5–25 words, length-balanced). The teacher is prompted with a fixed Arabic system specification and a JSON schema covering `{word, diacritized, irab, pos, case, role, marker}`. Total cost: $4.30. Outputs are validated by a JSON-schema check; rows that fail to parse are dropped. Note that this pool inherits Claude's systematic phrasing biases — addressed in §7.

**Templated (rule-derived from morphological tags).** From QAC (~78K Quranic words with full morphological annotation) and UD_Arabic-PADT (~6.7K MSA news sentences), we derive synthetic i'rāb prose by deterministic templates over POS + Case + dependency-role triples. **This data is used only as training material for the from-scratch decoder baseline; it is excluded from RAG retrieval and from the marker-FT training set** because the templates are themselves a deterministic rule that the model would memorize rather than learn from.

For the Mix A specialist's training, we extract the marker phrase from Yarob + distilled i'rāb prose using a regex/FSM extractor. The extractor recovers an explicit marker for 89.7% of sentences; the remaining 10.3% (mahall pronouns, indeclinable particles, edge cases) are labeled `<NO_MARKER>` so the model learns to abstain. Final marker training set: **8,815 (sentence, word, case, role) → marker_phrase** pairs.

**Construction tag distribution.** For error analysis we tag each Gazelle sentence by its dominant construction (regex on the gold prose itself: presence of فعل ماض, مضاف إليه, إن, سوى, …). The 30-sentence eval is dominated by short verbal sentences (15) and prepositional phrases (8); iḍāfa chains of length ≥3 (1) and exception constructions (2) are under-represented — a fact we return to in §7.

---

## 4. Methodology — [drafted ~400 words] ✅

### 4.1 Three systems compared

**System A: Claude Haiku 4.5 zero-shot.** A single API call per sentence, with a fixed Arabic system prompt instructing Claude to output a JSON array `[{word, diacritized, irab, pos, case, role, marker}, ...]`. Temperature 0. No in-context demonstrations.

**System B: Claude Haiku 4.5 + RAG (k=5).** For each input sentence, we retrieve the top-5 most similar sentences from a 1,060-example pool (459 Yarob + 601 distilled) using token-level Jaccard similarity over diacritic-stripped Arabic surface forms. The retrieved (sentence, prose-i'rāb-block) pairs are prepended to the prompt as in-context demonstrations. This is the strongest no-training baseline.

**System C: Hybrid (Mix A).** RAG produces a per-word JSON record. For each word we discard the marker field and call a fine-tuned AraT5v2-base specialist with the structured input `[case={c}] [role={r}] {word} | {sentence}` → `marker_phrase`. The specialist's output replaces RAG's marker, and the surface i'rāb prose is rebuilt from the canonical template (`{role} {case_word} وعلامة {case_marker} {marker_phrase}`).

The specialist is fine-tuned for 5 epochs on the 8,815-pair marker training set, with `paged_adamw_8bit`, gradient checkpointing, label smoothing 0.1, batch size 16 effective (4×4 grad-accum). [⏳ final-loss number from job 486438 inserted on completion.]

### 4.2 Structural metric definition

Each generated i'rāb string is parsed by a regex/FSM extractor into four fields: `pos`, `case ∈ {rafʿ, naṣb, jarr, jazm, mabni}`, `role` (~25 traditional labels), `marker` (~200 unique phrases). We report:

- **case_acc**: per-word match on case
- **role-F1 (macro)**: F1 averaged over the role label set; macro penalizes systems that miss rare roles
- **marker-EM**: exact-string match on the marker phrase
- **well-formed**: structural extractor parses without error
- **fully**: case ∧ role ∧ marker all match — the headline aggregate

We deliberately **avoid chrF/BLEU** for this task: those metrics give partial credit for "فاعل **منصوب** بالفتحة" (case dead wrong, but ~70% character overlap with the gold), which would mask the morphological errors we care about.

### 4.3 Statistical inference

With n=134 word judgments on Gazelle, the smallest detectable difference at α=0.05 power 0.80 is roughly ±7 percentage points on a binary metric. We report:

- **95% percentile bootstrap CIs (B=1000)** for every per-system metric
- **Paired bootstrap CIs** for system-vs-system deltas (resample matched word judgments)
- **McNemar's exact test** for paired binary outcomes (case, marker, fully)

A delta is reported as significant only if its 95% paired CI excludes 0 **and** McNemar p < 0.05.

---

## 5. Experimental Setup — [drafted ~250 words] ✅

**Hardware.** Marker fine-tuning ran on Bocconi `stud` partition (NVIDIA A100 80GB, allocated as a full unsplit slice). All evaluation and inference used a local laptop (RTX 4060 8GB), via the Anthropic API for the LLM-based systems. The from-scratch decoder was trained earlier in the project on the same A100 over a longer schedule (multi-day, not the focus of this paper).

**Hyperparameters.** RAG retrieval k=5 (selected as a balance between context length and demonstration diversity; k=3 and k=10 were not ablated due to evaluation budget). Marker specialist: AraT5v2-base-1024 (296M params), fine-tuned with `learning_rate=1e-4`, 5 epochs, validation loss tracked every epoch. Total training cost: 1× A100 hour × 5 epochs ≈ 30 minutes wall time.

**Inference cost.** Per-sentence Claude API call: ~$0.005 (Haiku 4.5, ~1500 input + 500 output tokens with k=5 retrieval). The full Gazelle eval (30 sentences × 3 systems = 90 calls) costs ~$0.50. Hybrid inference adds one local AraT5v2 forward pass per word (~3 ms on the 4060), negligible.

**Reproducibility.** Code: `https://github.com/HatemSaadallah/irab_project`. The retrieval pool is materialized at `data/distilled_irab.jsonl` (601 rows from Claude Haiku 4.5). The marker training set is at `data/marker_pairs.jsonl` (8,815 rows derived from Yarob + distilled). API calls used `temperature=0` for determinism, but Claude Haiku 4.5 is non-stationary across model snapshots; future replications should pin the model snapshot.

---

## 6. Results — [drafted ~450 words] ✅

### 6.1 Headline comparison

Three LLM-based systems on Gazelle (n=30 sentences = 134 word judgments). All numbers are percentages with 95% percentile bootstrap CIs (B=1000) in brackets. The from-scratch decoder is reported as a documented negative-result baseline in Appendix A.

| System | well-formed | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|
| Claude Haiku 4.5 zero-shot | 77.6 [70.1, 84.3] | 57.5 [49.3, 66.4] | 55.9 [42.0, 68.6] | 40.3 [32.1, 49.3] | 18.7 [11.9, 25.4] |
| **Claude RAG (k=5, 1,060-pool)** | **79.9** [73.1, 86.6] | 67.2 [59.0, 75.4] | **68.8** [55.8, 82.2] | **44.8** [35.8, 53.0] | **27.6** [20.1, 35.8] |
| Hybrid (Mix A: RAG case+role + AraT5v2 marker) | 77.6 [70.1, 84.3] | **68.7** [61.2, 76.9] | 59.4 [44.8, 73.8] | 41.8 [33.6, 50.7] | 26.1 [18.7, 34.3] |

### 6.2 Significance testing

Paired bootstrap deltas + McNemar's exact test on matched binary outcomes:

| Comparison | Δ case | p-McNemar | Δ marker | p-McNemar | Δ fully | p-McNemar |
|---|---:|---:|---:|---:|---:|---:|
| RAG combined − Zero-shot | **+9.7** ★ | 0.011 | +4.5 | 0.180 | **+9.0** ★ | 0.004 |
| RAG combined − RAG (Yarob only) | +0.7 | 1.000 | +1.5 | 0.500 | +1.5 | 0.625 |
| **Hybrid − RAG combined** | **+1.5** | 0.791 | **−3.0** | 0.219 | **−1.5** | 0.791 |

The smallest detectable binary-metric difference at α=0.05 is approximately ±7 pp at this sample size; differences below that are within noise.

**Three findings, in order of certainty:**

1. **Retrieval-augmented Claude is significantly stronger than zero-shot Claude** on case (+9.7 pp, p=0.011) and on aggregate fully-correct (+9.0 pp, p=0.004). The 5-shot retrieval pool acts not as a knowledge addition (Claude already knows MSA grammar) but as a style anchor that reduces output variability on closed-vocabulary fields.

2. **Mix A's per-word routing did not produce a statistically detectable improvement over RAG.** Δ case +1.5 pp (p=0.791), Δ marker −3.0 pp (p=0.219), Δ fully −1.5 pp (p=0.791). The point estimate on role-F1 dropped from 68.8 to 59.4, but the bootstrap CIs overlap substantially ([55.8, 82.2] vs [44.8, 73.8]). At n=134 we can neither confirm nor reject Mix A as an improvement; we report it as a **negative result on the routing hypothesis at this evaluation scale**.

3. **Adding distilled pairs to the retrieval pool produced no significant change.** Δ case +0.7 pp (p=1.000). We retain combined-pool retrieval for marginally higher role coverage but make no improvement claim.

### 6.3 Per-construction error analysis

Each Gazelle sentence was tagged by construction type (regex on the gold prose: VERBAL / NOMINAL / IDAFA_HEAVY / PREPOSITIONAL / PARTICLE_MOOD / EXCEPTION). Per-tag aggregated `fully_correct_word` rate (95% bootstrap CIs):

| Tag | n words | Decoder | Zero-shot | RAG combined |
|---|---:|---:|---:|---:|
| NOMINAL | 18 | 11.1 [0.0, 27.8] | 50.0 [27.8, 72.2] | **61.1** [38.9, 83.3] |
| PARTICLE_MOOD | 30 | 0.0 | 16.7 [3.3, 30.0] | 26.7 [10.0, 43.3] |
| VERBAL | 61 | 1.6 [0.0, 6.6] | 14.8 [6.6, 24.6] | 24.6 [13.1, 36.1] |
| PREPOSITIONAL | 37 | 0.0 | 16.2 [5.4, 29.7] | 21.6 [8.1, 35.1] |
| IDAFA_HEAVY | 9 (n=1 sent) | 0.0 | 22.2 [0.0, 55.6] | 22.2 [0.0, 55.6] |
| **EXCEPTION** | 9 | 0.0 | **0.0** | **0.0** |

**Two findings.** First, nominal/copular sentences are roughly 2× easier than verbal sentences across all LLM systems (RAG 61.1% vs 24.6%), consistent with shorter dependency chains and fewer interacting case markers. Second, **istithnāʾ (exception) constructions are a complete failure mode**: all systems including the strongest score 0/9 on the two affected sentences (containing سوى and عدا). These constructions require recognizing that the post-marker noun takes a non-default case based on whether the exception is positive or negative — a rule that does not transfer reliably from prompt context alone, even with 5-shot retrieval.

---

## 7. Discussion — [TODO ~300 words] ⏳

**The negative result on Mix A is the central finding.** Our routing hypothesis was that case and role are knowledge-bound (recoverable from frontier-LLM pretraining + retrieval) while marker phrasing is style-bound (recoverable only by fitting a corpus-specific surface convention). If true, decomposing prediction into RAG-for-case-role and AraT5v2-for-marker should produce a measurable lift over RAG alone. **We found no such lift on Gazelle** (Δ case = +1.5 pp, p=0.791; Δ marker = −3.0 pp, p=0.219; Δ fully = −1.5 pp, p=0.791). We interpret this in three ways.

First, **Claude RAG is already strong on style at this evaluation scale**. The 5-shot retrieval block effectively anchors Claude's output to the demonstration set's marker phrasing, leaving a specialist trained on the same demonstration distribution little additional signal to exploit. Second, **the marker training set is style-skewed by its own teacher**: 7,172 of 8,815 marker pairs are derived from Claude-distilled outputs, so the AraT5v2 specialist learns to predict *what Claude would have said* rather than *what Yarob's gold style requires*. The intersection is exactly where RAG already performs well, leaving no headroom for the specialist to win. Third, **the role-F1 point-estimate drop (−9.4 pp) deserves caution**: the bootstrap CIs overlap heavily, so we cannot conclude Hybrid hurts role-F1. The direction is consistent with our hybrid prose rebuild dropping some of Claude's longer role labels (e.g., "اسم مجرور بحرف الجر إلى" → "اسم مجرور"). A future iteration should preserve the LLM's role string verbatim and overlay only the marker.

**The exception-construction failure is the second finding.** All systems including the strongest score 0/9 on the two istithnāʾ sentences (containing سوى and عدا). This is not an LLM capability gap — Claude can recite the rule when asked directly. It is a *prompt-context gap*: the standard prompt + 5-shot retrieval does not surface the exception rule reliably. A targeted prompt with explicit istithnāʾ rules, or a small specialist trained on exception examples specifically, would be the next intervention.

**Self-consistency observation (engineering note).** During Streamlit demo development we noticed Claude's structured `case` field can disagree with its own surface diacritization (e.g., `case=jarr` but `diacritized` ends in damma) on individual outputs. A small postprocessor in the demo uses the structured case to override the surface vowel for declinable words. We do not quantify this rate in the paper because the eval predictions JSONLs were generated before the `diacritized` field was added to the API schema; it is mentioned only to document demo behavior and is not a contribution.

**Why distillation didn't significantly help RAG retrieval.** The +0.7 pp case-acc gain from adding 601 distilled examples to the retrieval pool is statistically indistinguishable from zero on n=134 (p=1.000). Two explanations: (a) at k=5, Claude's pretrained Arabic-grammar capacity already saturates on most test sentences, leaving little headroom for retrieval to improve case; (b) Yarob alone already covers the dominant constructions in Gazelle's test distribution. Worth re-testing with a larger benchmark — see §8.

---

## 8. Limitations — [drafted ~200 words] ✅

1. **Sample size.** All comparisons rest on n=134 word judgments from 30 sentences. The smallest detectable difference at α=0.05 is roughly ±7 percentage points on binary metrics; we report only differences that survive paired bootstrap and McNemar's exact test.

2. **Construction coverage.** Gazelle's distribution skews toward short verbal sentences (50% of sentences) and prepositional phrases (27%); iḍāfa chains of length ≥3 are represented by a single sentence (n=9 words), and exception (istithnāʾ) constructions by two sentences (n=9 words). Generalization claims about Mix A are limited to *MSA news-style sentences within these construction frequencies*; we cannot speak to performance on syntactically more complex MSA without a larger benchmark.

3. **Marker extractor measurement floor.** The regex/FSM extractor recovers an explicit marker for 89.7% of distilled training sentences; the remaining 10.3% are scored as `<NO_MARKER>`. This introduces a systematic ~10% measurement floor on the marker-EM metric independent of model quality.

4. **Teacher-bound retrieval pool and training set.** 601 of 1,060 retrieval pool examples — and 7,172 of 8,815 marker training pairs — are Claude-generated. The Mix A specialist may fit Claude's systematic phrasing biases rather than gold MSA grammatical-tradition style; a manually-validated benchmark larger than Gazelle is needed to disentangle these.

---

## 9. Future Work — [drafted ~150 words] ✅

Three concrete directions, each addressing a specific limitation in §8:

1. **Expand the test set with a manually-validated 200-sentence MSA benchmark.** This is the highest-leverage next step. Hand-correcting 200 PADT sentences (Claude-Sonnet seeded for efficiency) would cut the smallest-detectable-difference floor from 7 pp to ~3 pp and enable construction-stratified analysis with adequate per-cell sample sizes. Estimated effort: 7 hours of human time + ~$10 of Sonnet seeding.

2. **Re-design the marker metric beyond exact-match.** The 10% measurement floor in §8 limits resolution. A token-level n-gram F1 over the marker phrase, combined with a small set of canonical-form mappings (e.g., "الضمة الظاهرة" ≡ "الضمة الظاهرة على آخره"), would produce a softer, more informative metric.

3. **Targeted annotation of mabni and mahall cases.** The current Mix A training distribution is heavily skewed toward declinable nouns (4,967 of 8,815). Targeted annotation of ~500 mabni words and istithnāʾ constructions would address the 0/9 exception failure mode observed in §6.3.

---

## Notes for the writer

- **Word counts** above are budgets. Total target: ~3 pages = roughly 2,500-3,000 words. Drafted sections (3, 4, 5, 8, 9) total ~1,400. Remaining (Abstract, 1, 2, 6, 7) need ~1,200 once the hybrid number lands.
- **One contribution defended thoroughly is better than four claimed loosely.** The current intro lists three; consider folding the "self-consistency repair" into the Discussion section as an observation, not a contribution.
- **Every claim in §6 must reference a specific table cell with a CI.** No naked "+8 points" anywhere.
- **The decoder is in Appendix A.** Not the main table.
- **Limitations are sharp and specific** (per the directive). Future work bullets each connect to a §8 limitation.
