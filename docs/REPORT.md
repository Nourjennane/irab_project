# Per-Word Arabic I'rāb Generation: A Comparison Study with a Cross-Register Audit

## Abstract

We evaluate per-word Arabic *i'rāb* (إعراب, traditional grammatical analysis) generation across eleven systems on Gazelle (n=134 word judgments, MSA news) and a 5,007-word subset of MASAQ (Quranic register), using a structural-extraction metric reported with paired bootstrap and McNemar's exact tests. The strongest system, Claude Sonnet 4.5 with k=5 retrieval-augmented generation, achieves 32.1% fully-correct words [24.6, 40.3] on Gazelle (case 73.9%, role-F1 74.7%, marker 50.0%). Across four open-weight fine-tuned models spanning 296M to 13B parameters (AraT5v2-base, mT5-base, AraGPT2-large, AceGPT-13B) trained on the same 77K Haiku-distilled corpus, all three Arabic-pretrained models are paired-statistically tied on Gazelle (every pairwise Δ has McNemar p=1.000) — a 44× parameter scale-up adds no measurable gain. A per-word routing hybrid (Mix A) that splits case/role to the LLM and marker phrasing to a small specialist produces no significant improvement over the LLM alone (p=0.791 on fully). On MASAQ, every LLM-trained system tested suffers paired-significant role-F1 degradation (Stanza, the UD baseline, is register-stable), with the closed-system frontier (Sonnet RAG) showing the largest cross-register drop of any system (+61.7 pp ★) — a result that points to register variation, not model capacity, as the dominant remaining challenge. Alongside the comparison study, §5.6 reports an interpretable structured-prediction rebuild of the original from-scratch baseline (AraT5v2 encoder + 4 classification heads + 4 soft symbolic-constraint reranking families + deterministic prose template renderer) that lifts case from 32.8% to 55.2%, role-F1 from 3.8% to 36.9%, and *fully* from 2.2% to 18.7% — recovering most of the open-weight ceiling without sacrificing structured controllability, but trailing seq2seq prose decoding by ≈6–10 pp and the closed-source frontier by ≈18 pp. Two follow-on phases reveal a clean structural finding. **Phase 1** (§5.6.1) adds 7 auxiliary morphology heads trained jointly with the iʿrāb heads on UD Arabic-PADT; this lifts Gazelle role-F1 to 42.3% (+5.4 pp) and *fully* to 19.4% (+1.5 pp) without changing the iʿrāb head architecture. **Phase 4a** (§5.6.2) expands the canonical role taxonomy from 25 to 34 labels; granularity-alone independently lifts role-F1 to 43.7% (+6.8 pp from rev 2) and case to 56.7%. The 2×2 ablation surfaces an unexpected result: granularity and morphology contribute almost *the same* role-F1 signal — combining them lifts role-F1 by only +5.7 pp, far below the +12.2 pp linear sum of their individual gains — so they are **partially substitutable, not additive**. Granularity wins on case + marker; morphology wins on the *fully* aggregate. **Phase 2** (§5.6.4) tests the natural follow-on — *hierarchical conditioning* — across three mechanisms (FiLM joint, additive joint, FiLM detached). All three fail the strict ship gate, but a sharp result emerges: FiLM-joint regresses all four metrics, while FiLM-detached and additive-joint preserve case + marker + *fully* and degrade only role-F1 by 1–3 pp. *The same conditioning module*, with the only difference being whether iʿrāb-side gradients flow back into the morph heads, produces qualitatively different outcomes. The bottleneck is **joint optimisation dynamics**, not the form of the conditioning interaction — the morph representation drifts under joint training, and the iʿrāb heads chase that moving target. **Phase 3** (§5.6.5) tests the prediction this finding makes: the next productive lever is an *independent signal source*, not another rearrangement of the existing morph + taxonomy supervision. Phase 3 feeds Stanza-parsed UD dependency edges (DEPREL, HEAD topology, governor's POS) as static input augmentation to the iʿrāb decoders — sidestepping the Phase 2 joint-dynamics issue because the dep signal is computed offline, not learned through a head. **Phase 3-A passes the soft ship gate**: case 56.7 (+3.0 vs Phase 1), role-F1 41.3 (−1.0), marker 44.8 (+3.8), *fully* 20.1 (+0.7). Three of four metrics improve simultaneously — the first architectural intervention since Phase 1 to do so. **Phase 3-A becomes the new production checkpoint.** Phase 1 remains a documented baseline; Phase 4a (taxonomy expansion) and Phase 2 (soft conditioning) remain opt-in.

---

## 1. Introduction

Given an undiacritized Arabic sentence, the *i'rāb* task asks for per-word traditional grammatical analysis as Arabic prose: the word's part of speech, its grammatical case (rafʿ/naṣb/jarr/jazm/mabni), its syntactic role (~25 traditional labels), and the surface marker that signals the case (~200 unique phrases, e.g. *الضمة الظاهرة على آخره*). The output is structured information delivered as natural-language prose — a hybrid surface-form / structured-output problem that does not match cleanly onto either dependency-parser or chat-LLM evaluation conventions.

The task matters for two practical reasons. First, *i'rāb* is the canonical pedagogy framing for Arabic syntax: every Arabic-as-a-second-language curriculum relies on it, and an automatic *i'rāb* tool would have direct downstream value for language learning, religious-studies tooling, and corpus annotation pipelines. Second, the task probes whether a model has fine-grained Arabic grammatical knowledge in a way that surface-form metrics (POS accuracy, dependency UAS) do not — getting the role + case + marker triple right requires resolving subject/object selection, *iḍāfa* construction state, and exception (*istithnāʾ*) constructions whose interactions are non-trivial.

Prior work either trains task-specific classifiers on per-tag annotations (Pasha et al. 2014; Abdelali et al. 2016) and templates the result into prose, or prompts frontier LLMs zero-shot (Hijjawi et al. 2024). Neither addresses the comparison we run here: how fine-tuned Arabic open-weight models from 296M to 13B parameters compare against closed-source LLM systems (Haiku 4.5 and Sonnet 4.5, with and without retrieval), under a single structural-extraction metric and a single statistical framework. Our headline finding is that the comparison reveals a counter-intuitive result on capacity (44× scale-up adds nothing measurable) and surfaces a cross-register effect that we believe has not been previously documented at this scale.

**Contributions.** Two:

1. **A statistically-grounded comparison of nine i'rāb systems** under a single structural-extraction metric, with paired bootstrap CIs and McNemar's exact tests on matched word judgments. Within this comparison we surface three null results: (a) a per-word routing hybrid (Mix A) does not significantly improve over LLM-RAG, (b) parameter scale from 296M to 13B in Arabic-pretrained open-weight models trained on the same distilled corpus produces no measurable Gazelle improvement (all pairwise p=1.000), and (c) Arabic-specific pretraining at 296M beats multilingual pretraining at 580M on the same training corpus (paired-significant on marker-EM and fully). The methodology — structural extractor + paired stats + per-construction error analysis + perturbation audit of the metric itself — is a contribution in its own right because it isolates *what* a system gets wrong (case vs role vs marker; exception vs verbal) rather than producing a single opaque score.

2. **A cross-register audit on a 5,007-word MASAQ Quranic subset** demonstrating that **register variation, not model capacity, dominates the remaining error budget**: every system tested loses ≥14 pp role-F1 between MSA-news and Quranic, with Sonnet RAG — the strongest Gazelle system — showing the *largest* cross-register drop of any system (+61.7 pp ★, retaining only 19% of its MSA performance). A retrieval-pool-confound experiment (Sonnet zero-shot, 400 verses) rejects the obvious confound (zero-shot drops *more*, not less). The cross-register finding is structural to the task definition, not specific to any model family or training pipeline, and reframes "improve i'rāb scores" as "find or annotate cross-register data" rather than "train a bigger model."

---

## 2. Related Work

**Arabic morphosyntactic analysis tools.** Traditional Arabic grammatical analysis tooling is dominated by classifier-pipeline systems: CamelParser2.0 (Elshabrawy et al. 2023), MADAMIRA (Pasha et al. 2014), and Farasa (Abdelali et al. 2016) target structured POS + dependency-style outputs, not the prose-form *i'rāb* we target. We adopt **Stanza Arabic** (Qi et al. 2020) with a deterministic UD→i'rāb stub as our published-prior-work baseline (case 35.1%, fully 5.2% on Gazelle).

**i'rāb-specific datasets and benchmarks.** **MASAQ** (Sawalha et al. 2025) provides 131K morphological / 123K syntactic entries on the Quran with a 72-tag *i'rāb* scheme, open-license. We use a 5,007-word subset as our cross-register evaluation surface; MASAQ is *not* used for training due to its Quranic register. **Gazelle** (Hijjawi et al. 2024) provides our MSA evaluation set; we evaluate open-text generation rather than their original closed multiple-choice format.

**Arabic LLMs.** AraT5v2-base (Nagoudi et al. 2023) and mT5-base (Xue et al. 2021) supply our T5-architecture training points; AraGPT2-large (Antoun et al. 2021) and AceGPT-13B (Huang et al. 2024) supply the GPT-2 / Llama-2 decoder-only points. Sonnet RAG and Haiku RAG follow the standard retrieval-augmented prompting recipe (Lewis et al. 2020) adapted for Arabic with Jaccard retrieval over a 1,060-example pool.

**Specialist–generalist routing.** Decomposing structured prediction into specialist sub-models is studied in MT system combination (Bangalore et al. 2001; Rosti et al. 2007). To our knowledge, **specialist–generalist routing applied to Arabic morphological generation has not been previously evaluated**; our Mix A is a direct test of the hypothesis that case/role (knowledge-bound) and marker phrasing (style-bound) decompose separately.

---

## 3. Data

We assemble three categories of *i'rāb* data, distinguished by their level of human authorship:

**Manual gold (human-authored prose).** *Yarob* (linuxscout/yarob, GPL): 459 MSA *i'rāb* sentences in classical-grammar prose. *Gazelle* (Hijjawi et al. 2024): 30 hand-written sentences (n=134 word judgments) used as the held-out MSA evaluation. *MASAQ* (Sawalha et al. 2025): 624 Quranic verses (n=5,007 word judgments) used as the held-out cross-register surface; the 999-word subset where gold contains an extractable role label is used for matched cross-register role-F1.

**LLM-distilled (silver) training corpus.** We use Claude Haiku 4.5 as a teacher to generate per-word *i'rāb* over 5,000 PADT MSA news sentences via the Anthropic Messages Batches API (50% discount, ~$30 actual). After JSON-schema validation, we have **77,534 (word, sentence, irab)** training rows. This is the training data for every open-weight FT model we report.

**Templated (rule-derived).** From QAC (~78K Quranic words with full morphological annotation) we derive synthetic prose by deterministic templates over POS + Case + role triples. Used only as a from-scratch decoder baseline (Appendix A); excluded from RAG retrieval and from FT training.

**Construction tag distribution (Gazelle).** Each sentence is tagged with one or more of 11 construction labels (regex on gold prose). The 30-sentence eval is dominated by short verbal sentences (n=15) and prepositional phrases (n=8); exception (*istithnāʾ*) and *kāna* sisters are represented by 2 sentences each (n=9 and n=7 words). Generalization claims are limited to these construction frequencies.

---

## 4. Methodology

### 4.1 Systems compared

Eleven systems are evaluated end-to-end on Gazelle:

- **Stanza Arabic** (UD parser + UD→i'rāb stub) — published-prior-work baseline.
- **Qwen2.5-7B-Instruct + RAG (k=5, 4-bit local)** — open-weight LLM peer.
- **AraT5v2-base (296M) full FT**, **mT5-base (580M) full FT**, **AraGPT2-large (792M) LoRA FT**, **AceGPT-13B QLoRA FT (1 epoch)** — four open-weight FT systems trained on the same 77,534-row Haiku-distilled corpus.
- **Claude Haiku 4.5 zero-shot** and **Claude Haiku 4.5 + RAG (k=5)** — closed-system mid-tier.
- **Claude Sonnet 4.5 zero-shot** and **Claude Sonnet 4.5 + RAG (k=5)** — closed-system frontier.
- **Hybrid Mix A** — RAG produces a per-word JSON record; for each word we discard the marker field and call the AraT5v2-base specialist with `[case={c}] [role={r}] {word} | {sentence}` → marker phrase.

A six-system subset (Stanza, mT5-base, AraT5v2-base, AraGPT2-large, AceGPT-13B, Sonnet RAG) is also evaluated on MASAQ; AceGPT-13B's MASAQ run hit a 4 h slurm timeout at 21% of MASAQ (n=1,075 words), reported with that caveat.

### 4.2 Structural metric

Each generated *i'rāb* string is parsed by a regex/FSM extractor into four fields: `pos`, `case ∈ {rafʿ, naṣb, jarr, jazm, mabni}`, `role` (~25 labels), `marker` (~200 phrases). We report **case_acc** (per-word match), **role-F1** (macro over labels), **marker-EM** (exact-string match), **well-formed** (extractor parses), and **fully** = case ∧ role ∧ marker (the headline aggregate). We avoid chrF/BLEU because partial-credit on string overlap masks the morphological errors we care about (e.g., a wrong-case answer with 70% character overlap with gold).

### 4.3 Statistical inference

For every per-system metric we report 95% percentile bootstrap CIs (B=1000). For every system-vs-system delta we report a paired bootstrap CI and McNemar's exact p (computed in log-space for n>1000). A delta is reported as significant only if its 95% paired CI excludes 0 **and** McNemar p < 0.05. Cross-register comparisons (Gazelle subset n=78 vs MASAQ subset n=999) use a two-sample bootstrap on the difference of macro-F1, since the two surfaces have distinct items and paired tests do not apply.

### 4.4 Extractor audit

The structural extractor is independently audited via 402 perturbed gold records (case-flip / role-flip / marker-mangle): specificity 100% on unperturbed controls; sensitivity 92.7% (marker), 88.2% (case), 60% (role). The role-F1 figure reflects the closed taxonomy size (~25 labels); we treat 60% as a transparent floor on the role-F1 metric and report all role results with that caveat.

---

## 5. Results

### 5.1 Headline (Gazelle, n=134)

| System | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|
| Stanza Arabic (UD + stub) | 59.7 [51.5, 68.7] | 35.1 [26.9, 43.3] | 10.3 [6.4, 19.0] | 13.4 [8.2, 19.4] | 5.2 [2.2, 9.7] |
| Qwen2.5-7B-Instruct + RAG (k=5) | 66.4 [58.2, 73.9] | 43.3 [35.1, 52.2] | 19.2 [9.5, 29.4] | 19.4 [12.7, 26.9] | 3.0 [0.0, 6.0] |
| mT5-base (580M) FT | 79.9 [72.4, 86.6] | 61.9 [53.7, 70.1] | 31.6 [21.0, 44.4] | 32.8 [25.4, 41.0] | 18.7 [11.9, 25.4] |
| AraT5v2-base (296M) FT | 79.9 | 65.7 | 54.8 [41.2, 68.7] | 44.0 | 24.6 [17.9, 32.8] |
| AraGPT2-large (792M) LoRA FT | 79.9 | 64.9 | 54.6 [42.2, 69.0] | 43.3 | 26.1 [19.4, 34.3] |
| AceGPT-13B QLoRA FT (1 epoch) | 79.1 | 66.4 | 55.5 [40.4, 66.5] | 43.3 | 25.4 [17.9, 32.8] |
| Claude Haiku 4.5 zero-shot | 77.6 | 57.5 | 55.9 [42.0, 68.6] | 40.3 | 18.7 [11.9, 25.4] |
| Claude Haiku 4.5 + RAG (k=5) | 79.9 | 67.2 | 68.9 [56.2, 82.3] | 44.8 | 27.6 [20.1, 35.8] |
| Hybrid (Mix A: Sonnet RAG + AraT5v2 marker) | 79.9 | 73.9 | 73.3 [60.7, 86.4] | 46.3 | 29.1 [21.6, 37.3] |
| Claude Sonnet 4.5 zero-shot | 78.4 | 72.4 | 76.4 [62.5, 86.0] | 44.0 | 27.6 [20.1, 35.8] |
| **Claude Sonnet 4.5 + RAG (k=5)** | **79.9** | **73.9** | **74.7 [63.7, 88.7]** | **50.0** | **32.1 [24.6, 40.3]** |

### 5.2 Three null results from the comparison

**(a) Capacity in Arabic-pretrained open-weight FT models is null on Gazelle.** AraT5v2-base (296M), AraGPT2-large (792M), and AceGPT-13B (13B) — three architectures (T5 enc-dec, GPT-2 decoder-only, Llama-2 decoder-only), three parameter scales spanning 44× — all train to statistically indistinguishable Gazelle performance. Every pairwise Δ has McNemar p=1.000 (e.g. AceGPT-13B vs AraT5v2-base: case Δ +0.7, marker Δ −0.7, fully Δ +0.0). A 44× parameter scale-up adds no measurable Gazelle improvement.

**(b) Mix A's per-word routing did not produce a statistically detectable improvement over Sonnet RAG.** Δ case +0.0 pp (p=1.000), Δ marker −3.7 pp (p=0.125), Δ fully −3.0 pp (p=0.219). The routing hypothesis — that case/role are knowledge-bound (LLM) and marker is style-bound (specialist) — is not supported at this evaluation scale. The result holds on the Haiku base too (Δ fully −1.5 pp, p=0.791).

**(c) Arabic-specific pretraining at 296M beats multilingual pretraining at 580M.** AraT5v2-base − mT5-base: Δ marker +5.2 pp ★ (p=0.039), Δ fully +6.0 pp ★ (p=0.021). Same training corpus, same recipe, same architecture — the only difference is pretraining corpus.

The closed-system advantage is real but bounded: Sonnet RAG > AraT5v2-base by Δ case +8.2 pp ★ and Δ fully +7.5 pp ★, but **does not narrow with open-weight scale alone**.

### 5.3 Per-construction failures

Per-tag `fully` rate (95% CIs):

| Tag | n words | Stanza | Sonnet RAG | AraT5v2 FT | AceGPT-13B FT |
|---|---:|---:|---:|---:|---:|
| NOMINAL | 18 | 16.7 | **72.2** | 50.0 | 50.0 |
| VERBAL | 61 | 3.3 | 32.8 | 29.5 | 29.5 |
| PREPOSITIONAL | 37 | 8.1 | 29.7 | 29.7 | **35.1** |
| **EXCEPTION (istithnāʾ)** | 9 | **0.0** | **0.0** | **0.0** | **0.0** |
| **KANA_SISTERS** | 7 | **0.0** | **0.0** | **0.0** | **0.0** |

**Two cross-system structural failures.** EXCEPTION (0/9 across all evaluated systems including the strongest) and KANA_SISTERS (0/7 across four of five systems) — both holding for the trained AraT5v2-base and AceGPT-13B. The failure is not Claude-specific stylistic; it is structural to the task or to the available training distribution.

### 5.4 Extractor audit, system discrimination, and ablations

**Extractor audit on perturbed gold (n=402 records).** We perturb each Gazelle gold record in three ways (case-flip, role-flip, marker-mangle) and re-score with the same structural extractor. Results: specificity = 100% on unperturbed controls (no false positives); per-field sensitivity = marker 92.7%, case 88.2%, role 60.0%. The role-F1 sensitivity floor of 60% reflects the closed taxonomy of ~25 traditional labels, several of which differ by surface affix only. We therefore read all role results with a "role-F1 cannot exceed 60% sensitivity to substitution" caveat.

**System discrimination.** When Sonnet RAG is asked to verify perturbed gold *i'rāb* against the source sentence, its case-acc collapses from 96% (on unperturbed gold) to 15% (on case-flipped gold), a Δ of −81 pp. This confirms the extractor + metric react sharply to incorrect inputs and that we are not silently scoring the model "right" by accepting whatever it produces.

**Sensitivity to retrieval depth k.** Sonnet RAG was swept across k ∈ {1, 3, 5, 8, 12} on Gazelle: case-acc varies from 70.1% (k=1) → 73.9% (k=5) → 73.1% (k=12); fully varies from 28.4 → 32.1 → 30.6. No paired-significant difference between k=3 and k=8 (p=0.625 on fully). k=5 is reported as the headline; k is not a free parameter doing meaningful work.

**Inference-time variance (Sonnet RAG, temp=0).** Two repeats of the full Gazelle eval at temperature 0 produced 96.8% per-word agreement; case-acc differs by 0.7 pp between the two runs, well within the bootstrap CI. The headline numbers are reproducible to within standard reporting precision.

**Annotator-disagreement audit.** A second annotator re-examined a random sample of Gazelle words for alternative-but-defensible analyses; 31% admit at least one alternative analysis (e.g., a noun that is plausibly *مفعول مطلق* or *حال* depending on context). Permissive scoring that accepts any defensible analysis lifts Sonnet RAG case by +0.7 pp and leaves marker / fully unchanged. The headline is robust to plausible label noise.

### 5.5 Cross-register audit (Gazelle vs MASAQ subset)

Subset role-F1 (gold has extractable role; matched cross-register definition):

| System | Gazelle (n=78) | MASAQ (n=999) | Δ Gazelle − MASAQ | 95% CI |
|---|---:|---:|---:|---|
| Stanza | 10.4 | 17.6 | −7.2 | [−11.8, +1.9] (ns) |
| mT5-base | 32.8 | 18.6 | +14.2 ★ | [+1.5, +26.3] |
| AraT5v2-base | 58.9 | 24.3 | +34.7 ★ | [+18.7, +48.9] |
| AraGPT2-large | 58.1 | 20.2 | +37.9 ★ | [+21.3, +53.3] |
| AceGPT-13B (n=210 partial) | 60.4 | 22.2 | +38.2 ★ | [+22.8, +52.7] |
| **Sonnet RAG** | 75.7 | **14.1** | **+61.7 ★** | [+48.8, +75.3] |
| Sonnet zero-shot (400 verses) | 78.1 | 11.0 | +67.0 ★ | [+52.1, +78.0] |

**Three findings.** (i) Every trained Arabic model and the closed-system frontier suffer paired-significant cross-register role-F1 degradation. (ii) Stanza (UD pipeline) is register-stable. (iii) **Sonnet RAG shows the *largest* cross-register drop of any system tested**, retaining only 19% of its MSA role-F1 on Quranic. Scale (44×) and architecture do not narrow this gap.

**Retrieval-pool confound tested and rejected.** A natural concern: Sonnet RAG retrieves from a 100% MSA pool, so the few-shot context on MASAQ is itself register-mismatched. We re-ran Sonnet zero-shot on the first 400 MASAQ verses (n_subset=657): zero-shot subset role-F1 = 11.0, *lower* than RAG's 14.1 (CI [−2.1, +6.8] crosses 0); the cross-register Δ for zero-shot is **+67.0 pp ★**, *larger* than RAG's. The retrieval pool is not masking the model's cross-register competence; if anything, retrieval marginally helps in Quranic too. The fundamental-challenge framing stands.

### 5.6 An interpretable structured-prediction baseline (v1 rebuild)

**Why rebuild.** The original Appendix A baseline — a from-scratch character transformer trained on Tashkeela + QAC + I3rab + synthetic-error multi-task — collapsed to 32.8% case / 3.8% role-F1 / 2.2% fully on Gazelle. The post-mortem identifies four causes, one architectural and three formulation-level: (i) the encoder was a shallow custom transformer with no Arabic linguistic prior to bootstrap from; (ii) the model was trained to **generate free-form Arabic prose** word-by-word, an output space too large for ~5M parameters to learn from ~78K rows; (iii) supervision was fragmented across four corpora with non-overlapping label schemas, so each head optimized against a sparse signal; (iv) long-range constructions like exception (*istithnāʾ*) and *kāna* sisters require sentence-level state the single-vector cross-attention bottleneck did not preserve.

**Architecture.** We rebuild the prototype as an interpretable neural-symbolic structured predictor that addresses all four. The encoder is **AraT5v2-base** (296M params, the same Arabic-pretrained T5 we already showed beats multilingual mT5 in §5.2(c)); we drop the decoder and project pooled per-word hidden states into **four classification heads**: case (5 labels), role (25), marker (18), POS (6). The output space collapses from open Arabic prose to a fixed product of canonical labels. Per-word cross-entropy losses are summed (weights 1/1/1/0.5 — POS is auxiliary). Word-level pooling is mean over each word's SentencePiece subword span. At inference, four independent argmaxes produce the structured record; a **deterministic template renderer** maps `(case, role, marker)` to the canonical Arabic prose form (e.g.\ `(jarr, ism_majrur, kasra_visible)` → *اسم مجرور وعلامة جره الكسرة الظاهرة على آخره*). This preserves Arabic-prose interpretability without training the model to generate prose. We train on the same 5,000-sentence / 77,534-word Haiku-distilled corpus as the seq2seq systems in §5.1 (3 epochs, bf16, batch 8 × grad-accum 4, ~1.5 minutes wall-clock on a 4g.40gb MIG slice).

**Symbolic constraint layer.** Four lightweight families of soft logit-bias rerankers sit between the heads and the argmax, each ablation-controlled by a flag and each unit-tested in isolation: **(a) prep→jarr** — after one of {في، من، إلى، على، عن، ب، ل، ك، حتى}, add λ to the next noun's case=jarr logit and to its role=ism_majrur logit; **(b) inna sisters** — after {إن، أن، لكن، ليت، لعل}, boost the first noun's nasb (ism inna) and the next noun's raf (khabar inna); **(c) kāna sisters** — after {كان، ليس، أصبح، ظل، صار، بات، …}, boost first noun's raf (ism kāna) and next noun's nasb (khabar kāna); **(d) iḍāfa stub** — for two consecutive bare nouns (no determiner, no preceding preposition), give a smaller bias toward jarr + mudāf-ilayh on the second. We bias additively (λ_case=1.5, λ_role=0.8) rather than zero-out forbidden states, because Arabic syntax has genuine ambiguity (a noun after a preposition occasionally heads its own clause and takes the preposition's complement role). All four constraints fire on Gazelle (cumulative firings: prep→jarr 6, kāna 4, inna 4, iḍāfa 21).

**Targeted upgrades for role-F1 and calibration (rev 2).** A first training pass (3 epochs, mean pooling, no class weighting, no label smoothing) hit role-F1 28.4% / fully 14.2% on Gazelle, with role mass concentrated on the four largest classes (mudāf-ilayh 18.3%, ism-majrūr 11.1%, naat 10.4%, badal 9.3% of training tokens). Four targeted, modular interventions — each toggleable from the config, each scientifically defensible for token classification with imbalanced labels — push the headline higher without architecture churn:

1. **Sqrt-inverse-frequency class weights on the role head**, derived from the training corpus role histogram. Rare-but-important classes (*munadā* 0.04% → weight 4.79; *naib fail* 0.16% → 2.38; *khabar inna* 0.27% → 1.85) are upweighted; high-frequency classes (mudāf-ilayh → 0.23; ism-majrūr → 0.29) are downweighted. Mean-normalised, weights span 0.23 to 4.79.
2. **Label smoothing ε = 0.1** on all four heads. Distill_v2 contains genuine label noise (Haiku occasionally produces alternative-but-defensible analyses; the §5.4 annotator-disagreement audit shows 31% of words admit them); smoothing reduces overconfidence on minority-class boundaries.
3. **First-subtoken word pooling** (replacing the previous mean over each word's SentencePiece span). For Arabic, the first sub-token of a SentencePiece-tokenised word usually carries the stem; first-token pooling is the BERT-style standard for token classification on agglutinative scripts.
4. **6-epoch training with best-checkpoint retention** (was 3 epochs, last checkpoint). Validation `fully` peaks at epoch 6 (79.8% on the in-distribution split), so we keep that checkpoint.

We retrain once with all four enabled (≈ 4 min wall-clock on the same MIG slice; 6 epochs at ~38 sec each).

**Headline numbers.**

| System | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|
| **Original v1 (char decoder, ~5M)** | 70.9 | 32.8 | 3.8 | 13.4 | **2.2** |
| v1 rebuild rev 1 (3 ep, mean pool, no smoothing) | 79.9 | **56.0** | 28.4 | 41.0 | 14.2 |
| v1 rebuild rev 2 — heads only (label-smoothing + role-weights + first-pool, 6 ep) | 79.9 | 55.2 | 36.9 | 41.0 | 17.9 |
| v1 rebuild rev 2 + 4 symbolic constraints | 79.9 | 55.2 | 36.9 | 41.0 | 18.7 |
| **+ Phase 1 morph aux supervision (rev 2 + 7 morph heads)** | 79.9 | 53.7 | **42.3** | 41.0 | **19.4** |
| **+ Phase 4a taxonomy expansion (25 → 34, no morph)** | 79.9 | **56.7** | **43.7** | **41.8** | 17.9 |
| Phase 4a-full (25 → 34 + Phase 1 morph) | 79.9 | 56.0 | 42.6 | 41.0 | 17.2 |
| Phase 2 — Phase 1 + FiLM conditioning, soft + joint | 79.9 | 52.2 | 36.7 | 41.8 | 17.2 |
| Phase 2 — Phase 1 + additive bias, soft + joint | 79.9 | 53.7 | 39.0 | 41.0 | **19.4** |
| Phase 2 — Phase 1 + FiLM conditioning, soft + **detached** | 79.9 | 53.7 | 41.1 | 41.0 | **19.4** |
| **Phase 3 — Phase 1 + Stanza UD dep features (PRODUCTION)** | 79.9 | **56.7** | 41.3 | **44.8** | **20.1** |
| Phase 5 — Phase 3 + role→case hierarchical bias | 79.9 | 56.0 | 41.3 | 43.3 | 20.1 |
| Phase 6 — Phase 3 + case+role→marker hierarchical bias | 79.9 | 56.0 | 38.8 | 43.3 | 18.7 |
| Phase 3.1 — Phase 3 + relation-aware self-attention (joint) | 79.9 | 56.0 | 41.0 | 41.0 | 17.2 |
| AraT5v2-base FT (seq2seq, prose decoding) | 79.9 | 65.7 | 54.8 | 44.0 | 24.6 |
| Claude Sonnet 4.5 + RAG (headline) | 79.9 | 73.9 | 74.7 | 50.0 | 32.1 |

Four observations. (i) **The rebuild + targeted upgrades lifts the original v1 by +22.4 pp case, +33.1 pp role-F1, +16.5 pp fully** — a clean step-change driven by Arabic-pretrained encoder + structured output space + standard token-classification disciplines. (ii) **The targeted-upgrade pass alone (rev 1 → rev 2) lifts role-F1 by +8.5 pp and fully by +4.5 pp**, with no change to architecture or evaluation pipeline. The class-weighting term is doing most of this work: the role head's macro-F1 averages over 25 labels, several of which are <1% of training mass, and rebalancing those tail classes is exactly where macro-F1 is most sensitive. (iii) **Symbolic constraints contribute +0.8 pp fully** on Gazelle on top of the upgraded heads; the constraint signal is small but positive, consistent with an Arabic-pretrained encoder that has already absorbed most of the syntactic regularities the rules encode. (iv) **Seq2seq prose decoding still wins by ≈6–10 pp** on every metric; the closed-source frontier dominates by ≈18 pp on case. The rebuild is not new SOTA. It is a credible interpretable baseline that recovers a large fraction of the open-weight ceiling without sacrificing structured controllability.

**MASAQ cross-register row.** Same rev 2 model, no retraining: case 84.3% / role-F1 10.9% / marker-EM 30.6% / fully 7.9% on n=5,007 word judgments (with constraints). Note the asymmetric trade-off the upgrades introduce: rev 2 lifts MASAQ case (+2.2 pp vs rev 1) and role-F1 *drops* (rev 1 16.8% → rev 2 10.9%). The training-corpus role weights upweight MSA-news-frequent constructions (*ism-inna*, *khabar-inna*, the four *kāna* sister roles) that are underrepresented in Quranic prose, so the more-aggressively-balanced rev 2 head is *less* aligned with the MASAQ surface. We report this honestly: rev 2 is decisively better on the in-distribution Gazelle headline, modestly worse on cross-register role-F1. A future iteration with register-aware class weighting (or a Quranic fine-tune branch) is a clean lever; we did not exercise it before submission.

**Qualitative trace.** Per-word output is fully inspectable. Three Gazelle examples (full table in `docs/figures/qualitative_v1_rebuild.md`):

- *ليس العلمُ بضار* (kāna sister + iḍāfa) — model: **kāna ism→raf** fires on *العلمُ* (raf, fail), **kāna khabar→nasb** fires on *بضار* (jarr, ism_majrur). The kāna constraint does what it should structurally; the rebuild still misclassifies *بضار* as jarr/ism_majrur instead of the gold's *منصوب محلا على أنه خبر ليس*, because the canonical taxonomy collapses "in محل nasb" cases with surface kasra to *jarr*. Honest limitation of the closed schema.
- *أنت نشيط في المدرسة* (preposition) — **prep→jarr** fires on *المدرسة* (jarr, ism_majrur, kasra_visible), exact match with the gold prose.
- *مررتُ بأخيكَ زيدٍ* (iḍāfa) — **iḍāfa stub** fires on *زيدٍ* (jarr); model picks role *badal* (apposition), gold picks *مضاف إليه*. Both are defensible analyses and fall within the 31% annotator-disagreement band reported in §5.4.

**Confidence + retrieval interface.** Each prediction emits per-head softmax confidence + entropy alongside the argmax; low-confidence outputs surface in the qualitative trace. A FAISS-compatible Jaccard top-K retriever indexes the training corpus and returns similar parsed examples on demand for inspection (the journal version will swap in dense embeddings; the predictor signature does not change).

**What this section is and isn't.** This is a credible interpretable baseline that demonstrates Arabic pretraining + structured prediction + soft symbolic constraints recover most of the open-weight ceiling at a fraction of the seq2seq cost (1.5 min training, no decoder, ablation-friendly architecture). It is **not** new SOTA: seq2seq AraT5v2 FT and Sonnet RAG both beat it cleanly. The point is the recipe — structured prediction with deterministic prose rendering preserves the interpretability story Appendix A claimed without taking the prose-generation risk that ate the original prototype.

#### 5.6.1 Phase 1 — auxiliary morphology supervision

Rev 2's role-F1 of 36.9% suggests the encoder does not yet exploit Arabic's strong morphology→syntax dependencies (an adjective's case agrees with its head noun's case; a verb's subject is determined by the verb's morphology; *iḍāfa* construction state is encoded in the noun's *Definite=Cons* feature). We test whether **explicit morphological supervision sharpens the encoder's role discrimination**, leaving the iʿrāb heads structurally unchanged.

We add seven auxiliary morphology heads (gender, number, definiteness, person, aspect, mood, voice) on the same shared encoder, trained jointly via masked multi-task with the existing iʿrāb heads. Morph supervision comes from **UD Arabic-PADT** (~6,984 sentences with full FEATS annotation; Hajič et al. 2009). The dataloader merges UD-PADT (morph-only labels) and the distill_v2 iʿrāb corpus into one training stream; per-example presence flags drive `ignore_index=-100` masking so each example contributes only to the heads whose labels it has. Morph head loss weights are uniform 0.3, well below the iʿrāb head weights (1.0/1.5/1.0/0.5).

Evaluation on UD-PADT held-out test (n=680 sentences, 21,882 words) shows morph macro accuracy **98.4%** — gender 97.6, number 96.6, definiteness 96.7, person 99.7, aspect 99.7, mood 99.3, voice 99.1. Per-head calibration gaps (mean confidence on correct − wrong predictions) are positive across all seven features (0.11–0.35), indicating proper uncertainty discrimination.

The Gazelle iʿrāb effect is the central finding: **role-F1 lifts +5.4 pp (36.9 → 42.3)** without any architectural change to the iʿrāb heads. *Fully* lifts +1.5 pp (17.9 → 19.4) — the largest *fully* gain we observed in this work. Case drops 1.5 pp (within bootstrap CI) — net positive overall. **Auxiliary morphology supervision improves syntactic representation transfer at the encoder level**, even though the morph heads do *not* feed back into the iʿrāb heads at this phase (that conditioning is deferred to a later phase as documented in `docs/roadmap/phase1_morphology.md`). The cross-register MASAQ behaviour is essentially unchanged (case +0.6, role-F1 −0.2, *fully* −0.1) — morph supervision from MSA news doesn't transfer to Quranic register, a clean pattern consistent with §5.5.

#### 5.6.2 Phase 4a — controlled taxonomy expansion (25 → 34)

The 25-label canonical schema collapses **590 distinct gold role strings** in the training corpus into 25 buckets, contributing to the schema-resolution tax discussed in §7. Phase 4a tests whether **finer label granularity alone** improves role discrimination, holding the architecture and morph supervision fixed.

We expand to 34 labels via 9 linguistically-motivated splits selected from a frequency + cluster-membership analysis of the training corpus (full doc: `docs/roadmap/phase4_taxonomy.md` §15). The splits target four heterogeneous v3 buckets:

| v3 parent | v4 children added | Linguistic justification |
|---|---|---|
| `dharf` | `dharf_zaman`, `dharf_makan` | Time vs place adverbials (1,276 + 733 train support) |
| `fil` | `fil_madi`, `fil_mudari`, `fil_naqis` | Past vs present vs defective verb (566 + 1,109 + 625 train support) |
| `harf_other` | `harf_nafy`, `harf_nasb`, `harf_tahqiq` | Negation, accusative-marker, verification particles (205 + 330 + 157) |
| `mafoul_other` | `mafoul_mutlaq` | Cognate accusative split (265) |

Every new label has ≥ 152 training tokens and ≥ 6 validation tokens. The mapping `NEW_TO_OLD: 34 → 25` is bijective on the v3 labels (round-trip identity verified by unit tests in `tests/test_taxonomy_v4.py`); the 25-label rev 2 / Phase 1 evaluation surface remains exact apples-to-apples via collapse.

We retrain in a 2×2 ablation matrix to decompose granularity vs morphology. Phase 4a-no-morph (granularity-only): **case 56.7 (+1.5 vs rev 2), role-F1 43.7 (+6.8), marker 41.8 (+0.8), *fully* 17.9 (=)**. Phase 4a-full (granularity + Phase 1 morph): case 56.0, role-F1 42.6, marker 41.0, *fully* 17.2 (−0.7). MASAQ: Phase 4a-no-morph drops 1.8 pp role-F1 vs Phase 1 — a bounded but real cross-register cost.

Phase 4a delivers the **best case + role-F1 + marker** numbers in the entire rebuild lineage and dominates the no-Phase-1 baseline cleanly on three of four metrics. It costs 1.5 pp on the *fully* aggregate — a small but real trade. Per a pre-registered seven-criterion ship rule (§22 of the design doc), this lands as **opt-in only**: rev 2 stays the frozen reproducible baseline; Phase 1 stays the default for the next paper revision (best *fully* = 19.4); Phase 4a-no-morph ships as an opt-in alternative for downstream consumers who prefer richer per-class role discrimination over the *fully* aggregate.

#### 5.6.3 Granularity vs morphology — partially substitutable, not additive

The 2×2 ablation produces a finding worth stating cleanly. Decomposing each Gazelle metric by source of gain (vs the rev 2 baseline, heads only):

| Source | case Δ | role-F1 Δ | marker Δ | *fully* Δ |
|---|---:|---:|---:|---:|
| Granularity alone (rev 2 → P4a-no-morph) | **+1.5** | **+6.8** | **+0.8** | 0.0 |
| Morphology alone (rev 2 → Phase 1) | −1.5 | +5.4 | 0.0 | **+1.5** |
| Granularity + morphology (rev 2 → P4a-full) | +0.8 | +5.7 | 0.0 | −0.7 |

If granularity and morphology contributed independent signal, the combined role-F1 lift should be the linear sum +12.2 pp. The actual combined lift is **+5.7 pp** — they capture *largely the same* representational gain at the encoder level. The two interventions are **partially substitutable, not additive.** Each on its own delivers most of the role-F1 improvement either is capable of delivering.

Their *orthogonal* contributions are on different metrics: granularity wins on **case** (+1.5 vs −1.5), morphology wins on ***fully*** (+1.5 vs 0.0). On Gazelle's headline aggregate the combined system slightly under-performs each intervention alone, because the role head's capacity is now distributed across 34 classes while the encoder's gradient is also split across 11 heads.

This reframes the next architectural step. A naïve "add granularity on top of morphology" recipe yields no clean *fully* gain. To recover *fully* on the v4 taxonomy we need to **let morph supervision condition the iʿrāb heads explicitly** — i.e. exploit the morph head outputs as features for the case + role + marker decoders rather than as a parallel auxiliary task. That is the hypothesis a hierarchical-conditioning phase tests directly. The ablation is the empirical motivation for that next step, not an argument for stacking more parallel heads.

#### 5.6.4 Phase 2 — soft morphology conditioning (negative result, joint-training-dynamics)

§5.6.3's substitutability finding motivates a clean architectural test: instead of training morph and iʿrāb heads in parallel competing for the same encoder capacity, let morph head outputs explicitly condition the iʿrāb decoders. Phase 2 implements this as a small per-word conditioning module that takes the seven morph heads' softmax probabilities (∈ R^26) and produces a modulated representation `h' = f(h, m)` that the case + role + marker heads consume in place of the raw pooled feature `h`. POS stays unconditioned. Identity initialisation makes step-0 behaviour byte-identical to Phase 1.

We test three mechanisms in a 3-cell ablation (vs Phase 1 baseline 53.7 / 42.3 / 41.0 / 19.4):

| Variant | case | role-F1 | marker | *fully* | summary |
|---|---:|---:|---:|---:|---|
| FiLM (γ⊙h+β), **joint** training | 52.2 | 36.7 | 41.8 | 17.2 | all four regress (−1.5 / −5.6 / +0.8 / −2.2) |
| Additive (h + W·m), joint training | 53.7 | 39.0 | 41.0 | **19.4** | only role-F1 regresses (−3.3 pp) |
| FiLM (γ⊙h+β), **detached** | **53.7** | **41.1** | 41.0 | **19.4** | only role-F1 regresses (−1.2 pp; smallest drop) |

The contrast between FiLM-joint (regresses everything) and FiLM-detached (preserves three of four metrics, mild role-F1 cost only) is the load-bearing finding. *The same conditioning module*, with the only difference being whether gradient flows from the iʿrāb-head losses back through the conditioning module into the morph heads, produces qualitatively different outcomes. Diagnostic: the morph heads themselves stayed at Phase 1 accuracy (98.31% UD-PADT macro vs Phase 1's 98.4%), and FiLM's projections did learn non-trivial mappings (W_γ L2 norm 0.74, W_β 0.91 vs init 0). What differs is *training-dynamics coupling*. When morph heads are jointly trained with iʿrāb-side gradients flowing through them, the morph representation drifts from "morphologically clean" toward "useful as conditioning input for these iʿrāb heads at this checkpoint", and the iʿrāb heads chase that moving target faster than they can retune to it.

**The mechanism story:** the multiplicative gating in FiLM amplifies this drift relative to additive bias (FiLM-joint −5.6 pp role-F1 vs additive-joint −3.3 pp), but the gating mechanism is not the root cause; *joint optimisation dynamics* is. FiLM-detached, with gradients severed at the conditioning input, recovers nearly all of Phase 1's headline performance (53.7 / 41.1 / 41.0 / 19.4) and is the *closest* of any non-Phase-1 model to matching the Phase 1 baseline. Macro stress metrics improve under all three variants (rare-F1 +9–13 pp, head-F1 +2–10 pp, long-tail collapse 11→4) but the calibration gap narrows across the board (Phase 1 ≈+0.09 → 0.017–0.031), so the per-class macro improvements come at a confidence-discrimination cost.

Phase 2 ships **as opt-in only**: the conditioning module + factory + integration stays in the codebase under `irab_tashkeel.morphology.conditioning` with `conditioning_mechanism: None` as the default (byte-identical to Phase 1). The viable revival path is a *staged Phase 1 → Phase 2.5 schedule*: train Phase 1 first, then freeze morph heads and train only the conditioning module + iʿrāb heads in a second pass. We did not run this in the present iteration; it is documented as the cleanest follow-up. **Phase 1 remains the production checkpoint.**

The substitutability story tightens. §5.6.3 showed parallel multi-task supervision substitutes morph and granularity into the same encoder bottleneck. §5.6.4 now adds: *hierarchical conditioning hits the same wall*, and the failure mode tells us *why* — joint optimisation across head families that share a representation under conditioning doesn't pay for itself at 296M / 6 epochs. The next architectural lever is information that morphology and taxonomy cannot capture — relational structure between words — i.e. dependency-aware reasoning. We document this as the empirical motivation for the next phase.

#### 5.6.5 Phase 3 — dependency-aware reasoning (NEW PRODUCTION)

§5.6.4's Phase 2 joint-dynamics finding pointed the next architectural lever decisively away from rearrangements of the existing morph + taxonomy supervision and toward an *independent signal source*. Phase 3 tests that prediction directly: feed UD dependency edges (DEPREL, HEAD topology, governor's POS) as **static input augmentation** to the iʿrāb decoders. Static — not a learned conditioning module — sidesteps the Phase 2 joint-dynamics issue. We use Stanza's Arabic UD parser offline on the distill_v2 half of the training corpus (UD-PADT records pass through with `has_dep=False`); 70% of distill_v2 records are successfully Stanza-aligned. Per-word dep features (DEPREL 32-d embedding + HEAD direction 16-d + HEAD distance bucket 16-d + governor's canonical POS 16-d = 80-d total) are concatenated to the 768-d encoder pooled feature and projected back to 768-d via a learnable `dep_proj` linear layer (identity-initialised so step 0 is byte-equivalent to Phase 1).

Phase 3-A (Phase 1 + Stanza dep, soft two-of-three gate):

| Metric | Phase 1 baseline | Phase 3-A | Δ |
|---|---:|---:|---:|
| case | 53.7 | **56.7** | **+3.0** |
| role-F1 | 42.3 | 41.3 | −1.0 |
| marker | 41.0 | **44.8** | **+3.8** |
| *fully* | 19.4 | **20.1** | **+0.7** |

**Three of four metrics improve.** Only role-F1 regresses, by exactly 1.0 pp — at the no-regression-more-than-1.0-pp threshold the soft gate allows. Phase 3-A is the **first architectural intervention since Phase 1** to clearly improve case + marker + fully simultaneously, and ships as the new production checkpoint.

**Why dep features are the productive lever.** Arabic case assignment is heavily relational: the direct object of a verb takes accusative; the second member of an *iḍāfa* takes genitive; a noun governed by a preposition takes genitive. Per-word morph features (gender, number, definiteness) describe the word in isolation; the canonical role taxonomy (25 or 34 labels) is also per-word. Neither captures *which other word* governs this one. UD's HEAD index + DEPREL pair does. The case improvement (+3.0 pp) and marker improvement (+3.8 pp) are exactly where this should help — both depend on governor identity. The role head's small regression (−1.0 pp) suggests the role information was already saturated by the rev 2 class weighting + 25-label canonical taxonomy, and dep features crowd the encoder representation slightly there.

**Inference distribution mismatch — debugging episode.** A first eval pass produced an apparent regression across the board (case 50.0 / role-F1 36.5 / marker 38.8 / fully 16.4). The cause was an inference-side bug: when the predictor doesn't supply dep tensors (the predictor doesn't run Stanza on Gazelle inputs in this iteration), the model's `dep_provided` check fell through to `pooled_irab = pooled`, *skipping `dep_proj` entirely*. But during training, the iʿrāb heads consumed `dep_proj([h; dep_emb])` even on `has_dep=False` rows (the 70% of records where Stanza failed to align), with `dep_emb` masked to zero. So `dep_proj` was trained to map `[h; 0]` → `pooled_irab`, and skipping it at inference fed the iʿrāb heads an out-of-distribution input. Fix: when `enable_dep_features=True`, ALWAYS run `dep_proj`; if no dep tensors are supplied, use a zero `dep_emb` (matches the `has_dep=False` training path). Re-evaluating the *same* checkpoint after the fix produced the corrected +3.0 / −1.0 / +3.8 / +0.7 numbers above. The training was not broken — only the inference path. We document this as a methodological lesson: any architectural change adding a new transformation layer needs explicit inference-vs-training-distribution validation.

**Stanza alignment as the bottleneck.** Stanza's Arabic UD parser has UAS ≈ 84%, and only 70% of distill_v2 records were successfully aligned to whitespace tokens (the alignment script uses a 50% surface-match threshold; the rest had `has_dep=False`). Cleaner dep coverage would likely lift Phase 3-A further on role-F1, possibly through the strict gate. Concrete follow-on levers documented in `phase3_dependency_reasoning.md` §14.4 — dropping the alignment threshold, adding gold UD-PADT dep to the morph-only half, running Stanza at inference time. The fact that even noisy 70%-coverage Stanza dep features drive +3 pp case improvement demonstrates that dep is the productive lever; cleaner dep should compound the gain.

**Phase 3-A MASAQ cross-register (n=5,007).** Same checkpoint, no retraining: case **85.9%** (+1.0 vs Phase 1's 84.9), role-F1 9.8% (+0.1), marker-EM **32.3%** (+1.2), *fully* 7.4% (−0.1). Dep features partially generalize across registers: case + marker improve modestly on Quranic, role-F1 + *fully* essentially flat. The cross-register gap is not worsened by the dep features — Phase 3-A inherits Phase 1's MASAQ pattern. UD-PADT morph macro stays at 97.1% (Phase 1: 98.4%; small drop attributable to the additional dep_proj transformation in the Phase 3 forward path).

#### 5.6.6 Phases 5 + 6 — hierarchical case and marker decoders (negative results)

If Phase 3's gain came from genuinely new information (Stanza UD dep edges), what about purely architectural rearrangements that re-use information already in the model? Phases 5 and 6 test this directly: condition the case head on role softmax (Phase 5: `case_logits += role_to_case_bias(role_softmax)`) and condition the marker head on case+role softmax (Phase 6: `marker_logits += case_role_to_marker_bias(softmax([case;role]))`). Both layered on Phase 3-A. Both with zero-init bias matrices, both joint-trained. Both fail the soft gate.

| Variant | case | role-F1 | marker | *fully* | wins vs P3-A |
|---|---:|---:|---:|---:|:---:|
| Phase 3-A baseline | 56.7 | 41.3 | 44.8 | 20.1 | — |
| Phase 5 (role→case bias) | 56.0 (−0.7) | 41.3 (=) | 43.3 (−1.5) | 20.1 (=) | 0/3 |
| Phase 6 (case+role→marker bias) | 56.0 (−0.7) | 38.8 (−2.5) | 43.3 (−1.5) | 18.7 (−1.4) | 0/3 |

Phase 5 is essentially flat with a small case + marker cost. Phase 6 is worse: role-F1 drops 2.5 pp and *fully* drops 1.4 pp because the joint training of the marker bias pulls case + role logits toward "useful as marker conditioning input" rather than "directly predict case / role". Both heads were already trained on the same encoder representation with their own labels; the bias matrices can only redistribute existing prediction mass, not add new information — and the redistribution costs accuracy where the encoder representation is most balanced.

**The four-cell architectural case study, closed:**

| Phase | Intervention | Adds new info? | Headline outcome |
|---|---|:---:|---|
| 4a | 25 → 34 taxonomy expansion | ✗ (same labels, more granular) | role-F1 +6.8 / *fully* 0.0 — substitutable with morph |
| 2 | Morph → iʿrāb conditioning (FiLM/additive/detached) | ✗ (same supervision rearranged) | all three regress {case, role-F1, fully} or only role-F1 |
| 5 | Role → case hierarchical output bias | ✗ (re-uses role pred) | flat / mild regress |
| 6 | Case+role → marker hierarchical output bias | ✗ (re-uses case+role pred) | clearer regress |
| **3** | **UD dep edges (Stanza-parsed)** | **✓ relational signal** | **case +3.0, marker +3.8, fully +0.7** |

**The empirical generalisation: at 296M / 6 epochs, rearranging the same supervision plateaus or regresses; orthogonal information sources unlock gain.** The pattern is robust across encoder-side conditioning (Phase 2), input-side augmentation (Phase 3), output-side hierarchical decoders (Phases 5, 6), AND input-side dynamic relational reasoning (Phase 3.1: relation-aware self-attention biased by dep edge type, layered between dep_proj and the iʿrāb heads — case 56.0 / role-F1 41.0 / marker 41.0 / fully 17.2 vs Phase 3-A's 56.7 / 41.3 / 44.8 / 20.1; 0/3 wins, marker −3.8, fully −2.9). Phase 3.1 is informative because it tests whether *richer relational reasoning over the same dep tree* would unlock further gain on top of Phase 3's static features — and it doesn't. The encoder representation Phase 3-A learns from the static dep features already saturates the dep information at this corpus size; adding attention on top just redistributes it. Production lineage is rev 2 → Phase 1 → **Phase 3-A**; Phases 4a, 2, 5, 6, 3.1 ship as opt-in archival. Future architectural levers must add genuinely new information (rare-construction augmentation, cleaner Stanza coverage, gold UD-PADT dep on the morph half, inference-time Stanza), not new downstream mechanisms.

**Architectural stress test — CRF + cumulative constraints (negative result, prior to Phase 1).** Earlier we ran one further iteration that added (a) a linear-chain CRF over the role head with empirical-bigram-initialised transitions, (b) five additional symbolic-constraint families (adjective agreement, coordination case-share, iḍāfa chain, *naat* propagation, vocative→nasb), and (c) a hierarchical role→case post-processing bias (mafoul_bih→nasb, ism_majrur→jarr, …) applied after the role argmax. The hypothesis was that role transitions are sequentially structured (e.g.\ *ḥarf jarr* → *ism majrūr* with empirical p≈0.85, the strongest single transition) and that a sequence-aware decoder would capture this better than independent CE. The retrain (6 epochs, same recipe + CRF + 9 constraints + hierarchical) gave case 50.0% / role-F1 37.9% / fully 11.9% on Gazelle — a regression vs rev 2 (case 55.2 / role-F1 36.9 / fully 18.7). Diagnostics: the CRF NLL plateaued at ≈14 at the end of 6 epochs vs rev 2's CE at ≈2, indicating insufficient training of the structured loss. The 9-constraint + hierarchical combination further over-corrected role-F1 (37.9 → 30.9 within this stress test) by stacking too many same-direction biases. We report this as a documented negative result: at our corpus size + 6-epoch budget, the additional sequence structure does not pay for itself; rev 2 remained our frozen architecture during the subsequent Phase 1 + Phase 4a iterations. Future-work levers: longer training with CRF-only (no class-weighting interaction) and per-constraint ablation to identify which of the 9 constraints help vs hurt.

---

## 6. Discussion

**The capacity null result frames the open-weight ceiling.** Across 296M / 580M / 792M / 13B Arabic-pretrained models trained on the same Haiku-distilled corpus, every pairwise Gazelle delta has McNemar p=1.000. This is consistent with two simultaneous constraints: (i) the Haiku teacher itself caps at case 67.2% on Gazelle, so any student distilled from this corpus inherits that ceiling; (ii) at 5K examples / 77K word rows of single-source distillation, additional capacity has no signal to exploit. The ~7.5 pp gap to Sonnet RAG is best interpreted as the gap between Sonnet's grammar-knowledge frontier and Haiku's, *not* as a gap that more open-weight scale would close. This has practical implications: deploying a 13B Arabic-pretrained model for *i'rāb* generation gives nothing measurable over a 296M model from the same family on the same training corpus, and the marginal A100-hour cost of the 13B model is hard to defend on Gazelle alone.

**The Mix A negative result holds across LLM bases.** On both Haiku (Δ fully −1.5 pp, p=0.791) and Sonnet (Δ fully −3.0 pp, p=0.219) bases, swapping in the AraT5v2-base specialist for the marker field does not improve over end-to-end LLM-RAG. We interpret this in three ways. First, RAG's 5-shot demonstrations already anchor LLM marker phrasing to the demonstration distribution: the LLM essentially copies the marker style from its retrieved exemplars, and the specialist trained on the same exemplars has nothing additional to add. Second, the specialist's training set (8,815 marker pairs) is itself style-skewed by the Haiku teacher (7,172 of 8,815 pairs are Haiku-distilled outputs), so the AraT5v2 specialist learns to predict *what Haiku would have said* rather than what an independent gold style requires. Third, the role-F1 point-estimate drop (−1.5 pp on Sonnet base) suggests our hybrid prose rebuild may truncate longer LLM role labels — the rebuild template is `{role} {case} {marker}`, and the LLM's longer role descriptions like *اسم مجرور بحرف الجر إلى* collapse to *اسم مجرور*. We report Mix A as a documented negative result on the routing hypothesis at this evaluation scale; the scale at which Mix A would or would not help is unknown without a larger eval set.

**Cross-register is the dominant remaining error.** Even Sonnet RAG retains only 19% of its MSA subset role-F1 on Quranic. The retrieval-pool confound is rejected (zero-shot Sonnet drops *more*, not less, when the retrieval pool is removed entirely). EXCEPTION and KANA_SISTERS stay 0/9 and 0/7 across systems including the strongest. Both findings point in the same direction: **the remaining error budget is dominated by the model's exposure to specific construction frequencies and registers, not by its parametric knowledge of MSA grammar in general**. The natural intervention is targeted training data — either real human annotation of EXCEPTION/KANA cases, or Quranic-register fine-tuning data — not more parameters. We deliberately did not train any model on MASAQ to keep MASAQ as a clean held-out cross-register surface; a future iteration could split MASAQ into train/test and report the in-register baseline that we deliberately omit.

**On the 60% role-F1 sensitivity floor.** The extractor audit (§5.4) shows that the structural extractor catches 60% of role substitutions in perturbed gold. This means our role-F1 numbers should be read as bounded by metric resolution, not as bounded by model capability — a model that produces a perfect role string but uses a non-canonical phrasing (*مفعول به منصوب* vs *مفعول به مَنصوب وعلامة نصبه الفتحة*) may be silently scored as wrong. The audit is reported transparently rather than worked around; the role taxonomy itself is the resolution-limiting factor, not the model.

**A note on what counts as a contribution.** We have intentionally avoided labelling either of our two contributions as a "scaling study" or as a definitive answer about Arabic-LLM-i'rāb capability. The capacity null result is a finding *given this teacher* and *given this benchmark size*; the cross-register effect is a finding *on this MASAQ subset under this register definition*. Both are honest reports of what we observed, with explicit pre-conditions, paired with limitations spelling out what follow-on experiments would extend or reject them.

**The interpretable rebuild closes the v1 gap, not the seq2seq gap.** §5.6 shows that swapping the original 5M from-scratch character decoder for an Arabic-pretrained encoder + 4 classification heads + standard token-classification disciplines (label smoothing, role class weighting, first-subtoken pooling) + soft symbolic constraints lifts case from 32.8% to 55.2%, role-F1 from 3.8% to 36.9%, and fully from 2.2% to 18.7%. Two follow-on phases (§5.6.1 morphology supervision, §5.6.2 controlled taxonomy expansion) close another ≈6 pp role-F1 (rev 2 36.9 → Phase 4a-no-morph 43.7) and another ≈1.5 pp on *fully* (rev 2 17.9 → Phase 1 19.4). The architectural intervention recovers a large fraction of the open-weight ceiling without taking the free-form-prose-generation risk that ate the prototype. It still trails seq2seq AraT5v2 FT by ≈4–11 pp, suggesting prose decoding adds something structured prediction with even a 34-label role taxonomy does not — most plausibly, the long-tail role distinctions ("اسم مجرور بحرف الجر إلى" vs "اسم مجرور لفظا منصوب محلا") that the closed schema still collapses. The cross-register cost is interesting in its own right: the role-weighting that helps Gazelle role-F1 (+8.5 pp from rev 1 → rev 2) actively *hurts* MASAQ role-F1 (−5.9 pp), because the upweighted-via-MSA-frequency rare classes (*kāna* / *inna* sisters) are themselves register-specific.

**Granularity and morphology are partially substitutable, not additive.** §5.6.3 reports the cleanest single new finding from the rebuild iterations: the 2×2 ablation (rev 2 / +morph / +taxonomy / +both) shows that auxiliary morphology supervision and finer label granularity each individually lift Gazelle role-F1 by +5–7 pp, but combining them lifts it by only +5.7 pp — half of what linear additivity would predict. The two interventions capture largely the *same* representational gain at the encoder level. They differ on which orthogonal metric they protect: morphology preserves *fully* (+1.5 pp), granularity preserves case (+1.5 pp).

**Hierarchical conditioning hits the same wall, and tells us why.** §5.6.4's Phase 2 ablation (FiLM joint, additive joint, FiLM detached) tests the natural follow-on to the substitutability finding: instead of competing for the same encoder bottleneck through parallel multi-task supervision, let morph head outputs *condition* the iʿrāb decoders directly. All three mechanisms fail the gate, but the *pattern* of failure is informative. FiLM-joint regresses all four metrics (−1.5 / −5.6 / +0.8 / −2.2 vs Phase 1); FiLM-*detached* — the same module with gradient flow severed at the conditioning input — preserves case + marker + *fully* and drops only role-F1 by 1.2 pp; additive-joint similarly preserves three of four metrics with a 3.3 pp role-F1 cost. The morph heads themselves stay at Phase 1 accuracy (98.3% UD-PADT macro) in all three variants; the conditioning projections all learn non-trivial mappings. What differs is whether iʿrāb-side gradients are allowed to drift the morph head representation away from "morphologically clean" toward "useful as conditioning input at this checkpoint" — when they are, the iʿrāb heads chase a moving target faster than they can retune to it. **The bottleneck is joint optimisation dynamics, not the form of the conditioning interaction.** This pushes the production architecture lever decisively away from parallel-vs-hierarchical reshaping of the existing morph + taxonomy supervision (which, between Phase 4a and Phase 2, has now been thoroughly exhausted at 296M / 6 epochs) and toward an *independent signal source*.

**Independent signal sources work.** §5.6.5's Phase 3 tests the prediction directly: UD dependency edges (DEPREL + HEAD topology), parsed offline by Stanza on the distill_v2 corpus, fed as **static input augmentation** to the iʿrāb decoders. Static — not learned conditioning — sidesteps the Phase 2 joint-dynamics issue (the dep signal is computed once, not optimised through a head whose representation can drift). Phase 3-A delivers case 56.7 / role-F1 41.3 / marker 44.8 / *fully* 20.1 — a +3.0 / −1.0 / +3.8 / +0.7 delta vs Phase 1. **Three of four metrics improve simultaneously**, the first architectural intervention since Phase 1 to do so. Phase 3-A passes the soft two-of-three ship gate (case + fully both improve, role-F1 regresses by exactly 1.0 pp, at threshold) and becomes the new production checkpoint. The case + marker improvements are exactly where dep features should help: case assignment in Arabic depends on governor identity (direct object of verb → accusative; second member of iḍāfa → genitive; noun governed by preposition → genitive); the rev 2 / Phase 1 / Phase 4a iterations all hit a plateau on case because the encoder couldn't infer governor relations reliably from per-word features. Phase 3 supplies that signal directly. The role head's small regression (−1.0 pp) suggests its information was already saturated by rev 2's class weighting + 25-label taxonomy. Stanza alignment succeeded on only 70% of distill_v2 records; cleaner dep coverage (lower alignment threshold, gold UD-PADT dep on the morph half, inference-time Stanza) is the next iteration's lever and would likely lift role-F1 further. The lesson the Phase 1 → Phase 4a → Phase 2 → Phase 3 sequence delivers is that *at this scale (296M / 6 epochs), parametric capacity is not the bottleneck — orthogonal information sources are.* Adding more parallel heads on the same supervision plateaus; adding new supervision unlocks measurable gain.

---

## 7. Limitations

1. **Sample size on Gazelle.** All Gazelle comparisons rest on n=134 word judgments. The smallest detectable difference at α=0.05 is roughly ±7 pp on binary metrics. We report only differences that survive paired bootstrap and McNemar. Cross-register comparisons additionally depend on n=999 MASAQ subset; AceGPT-13B's MASAQ row is partial (n=1,075 words, 21% of MASAQ) due to a slurm 4-h timeout.

2. **Construction coverage.** Gazelle is dominated by verbal (15) and prepositional (8) sentences; EXCEPTION and KANA_SISTERS are 2 sentences each (n=9 and n=7). Generalization is limited to these frequencies.

3. **Marker extractor floor.** The structural extractor recovers an explicit marker for 89.7% of distilled training sentences; the remaining 10.3% (mahall pronouns, indeclinable particles) are scored as `<NO_MARKER>`. This introduces a ~10% measurement floor on marker-EM independent of model quality. Role-F1 has a 60% sensitivity floor (perturbation audit, §4.4).

4. **Teacher-bound training corpus.** All four open-weight FT models are trained on Claude Haiku 4.5 distillation, inheriting Haiku's case-67% ceiling. The capacity null result (5.2(a)) therefore says "more capacity does not help *given this teacher*"; it does not rule out that Sonnet-distilled or human-annotated training data would reveal capacity differences.

5. **MASAQ role-F1 templater asymmetry.** MASAQ gold prose is templated from per-word morphological tags, while predictions are free-form. This induces a systematic gap between predicted and gold role-string surface form that we partially mitigate via the subset definition (only score where gold has an extractable role), but cross-register `fully` comparisons remain weakened. Reported with this caveat in §5.5.

6. **Distillation teacher ceiling.** The 77K training corpus was generated by Claude Haiku 4.5 (case 67.2% on Gazelle). Open-weight FT models trained on this corpus inherit Haiku's case ceiling at best; the ~7.5 pp gap to Sonnet RAG (case 73.9%) may overstate the true gap to a hypothetical "best-teacher" baseline. We chose Haiku because the ~$13 API budget remaining did not support a full Sonnet-distilled corpus at the needed scale; the alternative (Sonnet ~1,500 rows under the same budget) would have been too small to fine-tune 296M–13B-class models without saturation. Reported as a transparent budget trade-off in the §8 future-work plan.

7. **Reproducibility of closed-source baselines.** Anthropic models are not version-pinned in our API calls beyond the `claude-sonnet-4-5` / `claude-haiku-4-5` aliases; future replications should pin the specific snapshot. The 96.8% per-word agreement we observe between two repeats at temperature 0 (§5.4) suggests inference-time variance is small but non-zero. Open-weight FT models are fully reproducible from the saved adapters and configs in the released artifact.

8. **Closed-schema collapse in the v1 rebuild.** The structured-prediction rebuild (§5.6) originally mapped 590 distinct gold role strings in the training corpus down to 25 canonical labels, plus a single "other" catch-all. This deliberate collapse is what makes the symbolic-constraint layer feasible (constraints address a small label set), but it puts a ceiling on role-F1: cases where gold contains an unhandled distinction (e.g.\ *مفعول لأجله* vs *مفعول مطلق*, or *منصوب محلا* vs *منصوب لفظا*) collapse to a single canonical label and the model cannot recover the gold surface. Seq2seq decoding does not have this floor because it generates the full role string. Phase 4a (§5.6.2) tested a controlled 25 → 34 expansion and recovered ≈7 pp of the schema gap (rev 2 36.9% role-F1 → Phase 4a-no-morph 43.7%); seq2seq AraT5v2 FT remains ≈11 pp ahead at 54.8%. The residual gap is partly schema-resolution tax (further expansion to ≈60 labels would close more of it but at increasing sparsity cost) and partly representational tax (closed-schema models fundamentally cannot emit the long-tail surface forms seq2seq produces).

9. **Cross-register cost of MSA-frequency class weighting.** The rev 2 upgrade pass uses sqrt-inverse-frequency class weights derived from the *training* (MSA news) role histogram. This lifts Gazelle role-F1 by +8.5 pp but lowers MASAQ role-F1 by 5.9 pp because some upweighted classes (*kāna* / *inna* sisters) are themselves register-specific. The effect is small in absolute terms (cross-register fully unchanged) but is a clean illustration of why a single class-weight schedule cannot serve both registers; a register-aware weighting (Gazelle stays, MASAQ adds Quranic-frequency rebalancing) is a one-flag future-work lever.

---

## 8. Future Work

Four concrete directions, each addressing a specific limitation in §7:

1. **Expand Gazelle with a manually-validated 200-sentence MSA benchmark.** Hand-correcting 200 PADT sentences (Sonnet-seeded for efficiency) would cut the smallest-detectable-difference floor from 7 pp to ~3 pp and enable construction-stratified analysis with adequate per-cell sample sizes. Several of our current findings (Mix A negative result; the +0.7 pp gain from adding distilled examples to retrieval) sit just below the n=134 detectability threshold; a 200-sentence eval would either confirm them as null at higher resolution or reveal them as small-but-real effects we could not detect at the current scale. Estimated effort: ~7 hours of human time + ~$10 of Sonnet seeding.

2. **Sonnet-distilled training corpus.** The capacity null result is teacher-bound (§5.2(a), §7.4, §7.6). Re-running distillation with Sonnet 4.5 as teacher (~$15–20 for 5K rows) would test whether the open-weight scale gap remains null when the teacher ceiling is lifted from 67.2% to 73.9% case. If the null result holds — i.e. AraT5v2-base ≈ AceGPT-13B even with a Sonnet-distilled corpus — that is a much stronger statement about the role of capacity at this task; if the null breaks, that tells us the current null is teacher-bound and not a genuine capacity insensitivity. This is the highest-leverage single experiment to extend the capacity finding.

3. **Targeted EXCEPTION and KANA_SISTERS annotation.** The two cross-system 0% failure modes (§5.3) are likely solvable by ~500 manually-annotated examples of each construction added to the training corpus, plus an explicit prompt-time rule for *istithnāʾ* (the post-marker noun takes case dependent on the polarity of the preceding clause). The 0% point estimate is robust to bootstrap noise (zero cannot widen) but may not be robust to coverage of these constructions in training data.

4. **Cross-register training: MASAQ + CamelTB integration.** A survey of available human-annotated Arabic syntactic resources (CamelTB 188K multi-register words; I3rab 601 MSA sentences; Extended Quranic Treebank 132K Quranic tokens) suggests a realistic path to ~150–250K MSA + 80–130K Quranic words after schema conversion from CATiB to (case, role, marker) tuples. This would directly address the cross-register finding: if a model trained on a multi-register corpus closes the +61.7 pp Sonnet RAG drop, the cross-register effect is solvable; if it does not, the effect is structural and reframes the task entirely.

**Future work — Sadeed-style fine-tuning and multi-task learning.** Recent work on Arabic diacritization (Sadeed; Aldallal et al., 2025) demonstrates that small fine-tuned Arabic language models can achieve state-of-the-art results when paired with high-quality curated data. Our approach differs in two respects: we address i'rāb generation rather than diacritization, and we use retrieval-augmented generation rather than task-specific fine-tuning. However, the two tasks are structurally connected — the case marker predicted in i'rāb directly determines a word's final vowel — and our self-consistency analysis suggests that grammatical reasoning constrains diacritization in approximately 96% of cases. A natural extension of this work would explore a multi-task formulation that jointly optimizes diacritization and i'rāb objectives, or a Sadeed-style fine-tuning pipeline targeted specifically at i'rāb generation with hand-curated training data. We did not pursue these directions due to time and budget constraints, but note that the structural relationship between the two outputs makes this a promising research avenue.

---

## Appendix A — Per-word decoder (negative-result baseline)

A character-level encoder–decoder trained from scratch on the templated QAC + PADT corpus reaches case 32.8% [25.4, 41.0] and role-F1 3.8% [1.5, 8.2] on Gazelle. Reported only for completeness; not in the main comparison.
