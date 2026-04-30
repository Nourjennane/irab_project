# Per-Word Arabic I'rāb Generation: A Comparison Study with a Routing-Based Hybrid

*Working draft — final ~3 pages. Word budget per section in [brackets]; current draft in *italics* where complete, [TODO] where pending data.*

**Status legend:** ✅ drafted · 🟡 partial · ⏳ depends on Mix A result · ❌ not started

---

## Abstract — [TODO ~150 words] ⏳

*Will be written last. Single paragraph. Frames the contribution as comparison study + routing-based hybrid + structural metrics, with one headline number from the strongest system on the manually-curated benchmark (or Gazelle if that benchmark is dropped).*

---

## 1. Introduction — [TODO ~250 words] 🟡

*Open with the i'rāb problem statement: given an undiacritized Arabic sentence, produce per-word traditional grammatical analysis as Arabic prose, including diacritization. Cite Gazelle (Hijjawi et al. 2024) for prior framing of i'rāb-as-LLM-task. State that traditional Arabic grammatical analysis is structured prediction over a closed taxonomy (4 cases × ~25 roles × ~20 markers) but expressed as natural-language prose — making it a hybrid surface-form/structured-output problem. Note that prior approaches either (a) train task-specific classifiers on per-tag annotations, or (b) prompt frontier LLMs zero-shot. Neither directly addresses the asymmetry we observe: the "knowledge" needed for case and role assignment is captured by pretrained Arabic LLMs, but the "style" of writing the marker phrase ("الضمة الظاهرة على آخره" vs "الضمة" vs "الضمة المقدرة على الألف") is fitting to a corpus-specific surface convention.*

**Contributions:**
1. *A retrieval-augmented Claude-Haiku-4.5 baseline that achieves **67.2% case accuracy** [59.0, 75.4] on the Gazelle benchmark with **no fine-tuning** — establishing that frontier-LLM + 5-shot retrieval is a strong floor for this task.*
2. *A **per-word routing hybrid** (Mix A): RAG produces case + role labels (knowledge-bound), a 296M-parameter AraT5v2 specialist produces the marker phrase (style-bound), and the two are recombined per-word. We hypothesize this decomposition exploits the asymmetry above. [⏳ headline number after FT].*
3. *Structured-extraction evaluation methodology with **paired bootstrap CIs and McNemar's exact test** for system comparisons; per-construction error analysis revealing istithnāʾ (exception) constructions as a complete failure mode across systems.*

---

## 2. Related Work — [TODO ~250 words] ❌

*Three threads to cover concisely:*
1. *Arabic grammatical analysis tools: CamelParser2.0 (Obeid et al. 2022), Madamira (Pasha et al. 2014), Farasa (Abdelali et al. 2016). Note: these produce structured POS+dependency outputs, not the prose-form i'rāb we target.*
2. *Arabic LLMs: AraT5/AraT5v2 (Nagoudi et al. 2022, 2023), AraGPT/Jais (Sengupta et al. 2023), the Sadeed framework for diacritization (Misraj-AI 2024). **AraT5v2** is our specialist's base.*
3. *I'rāb-as-LLM evaluation: Gazelle (Hijjawi et al. 2024) introduced the eval set we use; their methodology focuses on closed multiple-choice-style probing. Our methodology is open-generation evaluation with structural-extraction metrics.*

**System combination/routing baseline:** *Search for prior art on decomposed routing for structured prediction (likely candidates: NER ensemble routing, MT specialist+generalist combinations). Argue why our framing — frontier LLM for syntactic knowledge + small specialist for marker phrasing — is, to our knowledge, novel applied to Arabic morphological analysis.*

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

## 6. Results — [TODO ~400 words] ⏳

*Three subsections:*

### 6.1 Headline comparison
*Table from RESULTS.md with 95% CIs. Three systems × six metrics. Hybrid row pending FT completion.*

### 6.2 Significance testing
*Paired bootstrap + McNemar table from RESULTS.md. Honest finding: distilled-pool RAG vs Yarob-only RAG is **not** significantly different on Gazelle; we use combined for higher role coverage but make no significance claim. RAG over zero-shot is significant on case and fully (+9-10 pp).*

### 6.3 Per-construction error analysis
*Table from RESULTS.md. Two findings to highlight in prose: (1) Nominal sentences are ~2× easier than verbal across all systems, consistent with shorter dependency-chain length. (2) Exception (istithnāʾ) constructions are 0/9 across all systems — a complete failure mode worth a paragraph.*

---

## 7. Discussion — [TODO ~300 words] ⏳

*Three threads after the hybrid number lands:*

1. **Whether the routing hypothesis pays off.** Did Hybrid significantly improve marker-EM and aggregate fully-correct over RAG alone? If yes (paired McNemar p<0.05 on marker), the routing decomposition is empirically validated. If no (no detectable improvement), report honestly and discuss why: marker phrasing may not be sufficiently style-bound (i.e., Claude's existing prose generation already fits gold style), or the AraT5v2 specialist did not converge on the diverse marker vocabulary in 8.8K pairs.

2. **Self-consistency repair as engineering observation.** While developing the demo we noticed that Claude's structured `case` field disagrees with its own `diacritized` final vowel on roughly [TODO%] of distilled outputs (e.g., `case=jarr` but `diacritized` ends in damma). A 30-line postprocessor that uses the structured case to override the surface vowel resolves these. We report this as an engineering observation rather than a research contribution; it has no effect on the metric numbers above.

3. **Why distillation didn't significantly help RAG retrieval.** The +0.7 pp case-acc gain from adding 601 distilled examples to the retrieval pool is statistically indistinguishable from zero on n=134. Two explanations: (a) at k=5, Claude's pretrained Arabic-grammar capacity already saturates on most test sentences, leaving little headroom for retrieval to improve case; (b) the distilled examples increase pool *style diversity* but at our small eval sample size, Yarob alone already covers the dominant patterns. Worth re-testing with a larger benchmark — see §8.

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
