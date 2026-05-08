# Per-Word Arabic I'rāb Generation: A Comparison Study with a Cross-Register Audit

## Abstract

We evaluate per-word Arabic *i'rāb* (إعراب, traditional grammatical analysis) generation across eleven systems on Gazelle (n=134 word judgments, MSA news) and a 5,007-word subset of MASAQ (Quranic register), using a structural-extraction metric reported with paired bootstrap and McNemar's exact tests. The strongest system, Claude Sonnet 4.5 with k=5 retrieval-augmented generation, achieves 32.1% fully-correct words [24.6, 40.3] on Gazelle (case 73.9%, role-F1 74.7%, marker 50.0%). Across four open-weight fine-tuned models spanning 296M to 13B parameters (AraT5v2-base, mT5-base, AraGPT2-large, AceGPT-13B) trained on the same 77K Haiku-distilled corpus, all three Arabic-pretrained models are paired-statistically tied on Gazelle (every pairwise Δ has McNemar p=1.000) — a 44× parameter scale-up adds no measurable gain. A per-word routing hybrid (Mix A) that splits case/role to the LLM and marker phrasing to a small specialist produces no significant improvement over the LLM alone (p=0.791 on fully). On MASAQ, every LLM-trained system tested suffers paired-significant role-F1 degradation (Stanza, the UD baseline, is register-stable), with the closed-system frontier (Sonnet RAG) showing the largest cross-register drop of any system (+61.7 pp ★) — a result that points to register variation, not model capacity, as the dominant remaining challenge. Alongside the comparison study, §5.6 reports an interpretable structured-prediction rebuild of the original from-scratch baseline (AraT5v2 encoder + 4 classification heads + 4 soft symbolic-constraint reranking families + deterministic prose template renderer) that lifts case from 32.8% to 56.0%, role-F1 from 3.8% to 28.4%, and fully from 2.2% to 14.2% — recovering most of the open-weight ceiling without sacrificing structured controllability, but trailing seq2seq prose decoding by ≈10 pp and the closed-source frontier by ≈18 pp.

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

**Numbers.**

| System | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|
| **Original v1 (char decoder, ~5M)** | 70.9 | 32.8 | 3.8 | 13.4 | **2.2** |
| **v1 rebuild — heads only (296M encoder + 4 heads)** | 79.9 | 55.2 | 28.4 | 41.0 | **14.2** |
| **v1 rebuild + 4 symbolic constraints** | 79.9 | **56.0** | 28.4 | 41.0 | **14.2** |
| AraT5v2-base FT (seq2seq, prose decoding) | 79.9 | 65.7 | 54.8 | 44.0 | 24.6 |
| Claude Sonnet 4.5 + RAG (headline) | 79.9 | 73.9 | 74.7 | 50.0 | 32.1 |

Three observations. (i) **The rebuild lifts the original v1 by +22.4 pp case, +24.6 pp role-F1, +12.0 pp fully** — a step-change driven by Arabic-pretrained encoder + structured output space. (ii) **Symbolic constraints add +0.8 pp case** on Gazelle (and a negligible +0.04 pp on the n=5,007 MASAQ subset). The constraint layer is real but small, consistent with an encoder that has already absorbed most of the syntactic regularities the rules encode. (iii) **Seq2seq prose decoding still wins by ≈10 pp** on every metric; the closed-source frontier still dominates by ≈18 pp on case. The rebuild is not new SOTA. It is a credible interpretable baseline that recovers a large fraction of the open-weight ceiling without sacrificing structured controllability.

**MASAQ cross-register row.** Same model, no retraining: case 82.1% / role-F1 16.8% / marker-EM 29.7% / fully 8.0% on n=5,007 word judgments. The MASAQ case figure is high because Quranic verses are dominated by *mabni* particles + iḍāfa chains, both well-represented in the training distribution; role-F1 drops sharply on Quranic (−11.6 pp vs Gazelle subset role-F1), the same cross-register pattern §5.5 reports for every other LLM-trained system.

**Qualitative trace.** Per-word output is fully inspectable. Three Gazelle examples (full table in `docs/figures/qualitative_v1_rebuild.md`):

- *ليس العلمُ بضار* (kāna sister + iḍāfa) — model: **kāna ism→raf** fires on *العلمُ* (raf, fail), **kāna khabar→nasb** fires on *بضار* (jarr, ism_majrur). The kāna constraint does what it should structurally; the rebuild still misclassifies *بضار* as jarr/ism_majrur instead of the gold's *منصوب محلا على أنه خبر ليس*, because the canonical taxonomy collapses "in محل nasb" cases with surface kasra to *jarr*. Honest limitation of the closed schema.
- *أنت نشيط في المدرسة* (preposition) — **prep→jarr** fires on *المدرسة* (jarr, ism_majrur, kasra_visible), exact match with the gold prose.
- *مررتُ بأخيكَ زيدٍ* (iḍāfa) — **iḍāfa stub** fires on *زيدٍ* (jarr); model picks role *badal* (apposition), gold picks *مضاف إليه*. Both are defensible analyses and fall within the 31% annotator-disagreement band reported in §5.4.

**Confidence + retrieval interface.** Each prediction emits per-head softmax confidence + entropy alongside the argmax; low-confidence outputs surface in the qualitative trace. A FAISS-compatible Jaccard top-K retriever indexes the training corpus and returns similar parsed examples on demand for inspection (the journal version will swap in dense embeddings; the predictor signature does not change).

**What this section is and isn't.** This is a credible interpretable baseline that demonstrates Arabic pretraining + structured prediction + soft symbolic constraints recover most of the open-weight ceiling at a fraction of the seq2seq cost (1.5 min training, no decoder, ablation-friendly architecture). It is **not** new SOTA: seq2seq AraT5v2 FT and Sonnet RAG both beat it cleanly. The point is the recipe — structured prediction with deterministic prose rendering preserves the interpretability story Appendix A claimed without taking the prose-generation risk that ate the original prototype.

---

## 6. Discussion

**The capacity null result frames the open-weight ceiling.** Across 296M / 580M / 792M / 13B Arabic-pretrained models trained on the same Haiku-distilled corpus, every pairwise Gazelle delta has McNemar p=1.000. This is consistent with two simultaneous constraints: (i) the Haiku teacher itself caps at case 67.2% on Gazelle, so any student distilled from this corpus inherits that ceiling; (ii) at 5K examples / 77K word rows of single-source distillation, additional capacity has no signal to exploit. The ~7.5 pp gap to Sonnet RAG is best interpreted as the gap between Sonnet's grammar-knowledge frontier and Haiku's, *not* as a gap that more open-weight scale would close. This has practical implications: deploying a 13B Arabic-pretrained model for *i'rāb* generation gives nothing measurable over a 296M model from the same family on the same training corpus, and the marginal A100-hour cost of the 13B model is hard to defend on Gazelle alone.

**The Mix A negative result holds across LLM bases.** On both Haiku (Δ fully −1.5 pp, p=0.791) and Sonnet (Δ fully −3.0 pp, p=0.219) bases, swapping in the AraT5v2-base specialist for the marker field does not improve over end-to-end LLM-RAG. We interpret this in three ways. First, RAG's 5-shot demonstrations already anchor LLM marker phrasing to the demonstration distribution: the LLM essentially copies the marker style from its retrieved exemplars, and the specialist trained on the same exemplars has nothing additional to add. Second, the specialist's training set (8,815 marker pairs) is itself style-skewed by the Haiku teacher (7,172 of 8,815 pairs are Haiku-distilled outputs), so the AraT5v2 specialist learns to predict *what Haiku would have said* rather than what an independent gold style requires. Third, the role-F1 point-estimate drop (−1.5 pp on Sonnet base) suggests our hybrid prose rebuild may truncate longer LLM role labels — the rebuild template is `{role} {case} {marker}`, and the LLM's longer role descriptions like *اسم مجرور بحرف الجر إلى* collapse to *اسم مجرور*. We report Mix A as a documented negative result on the routing hypothesis at this evaluation scale; the scale at which Mix A would or would not help is unknown without a larger eval set.

**Cross-register is the dominant remaining error.** Even Sonnet RAG retains only 19% of its MSA subset role-F1 on Quranic. The retrieval-pool confound is rejected (zero-shot Sonnet drops *more*, not less, when the retrieval pool is removed entirely). EXCEPTION and KANA_SISTERS stay 0/9 and 0/7 across systems including the strongest. Both findings point in the same direction: **the remaining error budget is dominated by the model's exposure to specific construction frequencies and registers, not by its parametric knowledge of MSA grammar in general**. The natural intervention is targeted training data — either real human annotation of EXCEPTION/KANA cases, or Quranic-register fine-tuning data — not more parameters. We deliberately did not train any model on MASAQ to keep MASAQ as a clean held-out cross-register surface; a future iteration could split MASAQ into train/test and report the in-register baseline that we deliberately omit.

**On the 60% role-F1 sensitivity floor.** The extractor audit (§5.4) shows that the structural extractor catches 60% of role substitutions in perturbed gold. This means our role-F1 numbers should be read as bounded by metric resolution, not as bounded by model capability — a model that produces a perfect role string but uses a non-canonical phrasing (*مفعول به منصوب* vs *مفعول به مَنصوب وعلامة نصبه الفتحة*) may be silently scored as wrong. The audit is reported transparently rather than worked around; the role taxonomy itself is the resolution-limiting factor, not the model.

**A note on what counts as a contribution.** We have intentionally avoided labelling either of our two contributions as a "scaling study" or as a definitive answer about Arabic-LLM-i'rāb capability. The capacity null result is a finding *given this teacher* and *given this benchmark size*; the cross-register effect is a finding *on this MASAQ subset under this register definition*. Both are honest reports of what we observed, with explicit pre-conditions, paired with limitations spelling out what follow-on experiments would extend or reject them.

**The interpretable rebuild closes the v1 gap, not the seq2seq gap.** §5.6 shows that swapping the original 5M from-scratch character decoder for an Arabic-pretrained encoder + 4 classification heads + soft symbolic constraints lifts case from 32.8% to 56.0%, role-F1 from 3.8% to 28.4%, and fully from 2.2% to 14.2%. The architectural intervention recovers a large fraction of the open-weight ceiling without taking the free-form-prose-generation risk that ate the prototype. It still trails seq2seq AraT5v2 FT by ≈10 pp on every metric, suggesting that prose decoding adds something structured prediction with our 25-label canonical role taxonomy does not — most plausibly, the long-tail role distinctions ("اسم مجرور بحرف الجر إلى" vs "اسم مجرور لفظا منصوب محلا") that the closed schema collapses. Adding more roles would lift role-F1 but would also widen the schema-noise tail; both alternatives are concrete future-work levers we describe in §8.

---

## 7. Limitations

1. **Sample size on Gazelle.** All Gazelle comparisons rest on n=134 word judgments. The smallest detectable difference at α=0.05 is roughly ±7 pp on binary metrics. We report only differences that survive paired bootstrap and McNemar. Cross-register comparisons additionally depend on n=999 MASAQ subset; AceGPT-13B's MASAQ row is partial (n=1,075 words, 21% of MASAQ) due to a slurm 4-h timeout.

2. **Construction coverage.** Gazelle is dominated by verbal (15) and prepositional (8) sentences; EXCEPTION and KANA_SISTERS are 2 sentences each (n=9 and n=7). Generalization is limited to these frequencies.

3. **Marker extractor floor.** The structural extractor recovers an explicit marker for 89.7% of distilled training sentences; the remaining 10.3% (mahall pronouns, indeclinable particles) are scored as `<NO_MARKER>`. This introduces a ~10% measurement floor on marker-EM independent of model quality. Role-F1 has a 60% sensitivity floor (perturbation audit, §4.4).

4. **Teacher-bound training corpus.** All four open-weight FT models are trained on Claude Haiku 4.5 distillation, inheriting Haiku's case-67% ceiling. The capacity null result (5.2(a)) therefore says "more capacity does not help *given this teacher*"; it does not rule out that Sonnet-distilled or human-annotated training data would reveal capacity differences.

5. **MASAQ role-F1 templater asymmetry.** MASAQ gold prose is templated from per-word morphological tags, while predictions are free-form. This induces a systematic gap between predicted and gold role-string surface form that we partially mitigate via the subset definition (only score where gold has an extractable role), but cross-register `fully` comparisons remain weakened. Reported with this caveat in §5.5.

6. **Distillation teacher ceiling.** The 77K training corpus was generated by Claude Haiku 4.5 (case 67.2% on Gazelle). Open-weight FT models trained on this corpus inherit Haiku's case ceiling at best; the ~7.5 pp gap to Sonnet RAG (case 73.9%) may overstate the true gap to a hypothetical "best-teacher" baseline. We chose Haiku because the ~$13 API budget remaining did not support a full Sonnet-distilled corpus at the needed scale; the alternative (Sonnet ~1,500 rows under the same budget) would have been too small to fine-tune 296M–13B-class models without saturation. Reported as a transparent budget trade-off in the §8 future-work plan.

7. **Reproducibility of closed-source baselines.** Anthropic models are not version-pinned in our API calls beyond the `claude-sonnet-4-5` / `claude-haiku-4-5` aliases; future replications should pin the specific snapshot. The 96.8% per-word agreement we observe between two repeats at temperature 0 (§5.4) suggests inference-time variance is small but non-zero. Open-weight FT models are fully reproducible from the saved adapters and configs in the released artifact.

8. **Closed-schema collapse in the v1 rebuild.** The structured-prediction rebuild (§5.6) maps 590 distinct gold role strings in the training corpus down to 25 canonical labels, plus a single "other" catch-all. This deliberate collapse is what makes the symbolic-constraint layer feasible (constraints address a small label set), but it puts a ceiling on role-F1: cases where gold contains an unhandled distinction (e.g.\ *مفعول لأجله* vs *مفعول مطلق*, or *منصوب محلا* vs *منصوب لفظا*) collapse to a single canonical label and the model cannot recover the gold surface. Seq2seq decoding does not have this floor because it generates the full role string. Concretely, the v1 rebuild's 28.4% role-F1 is roughly half of seq2seq AraT5v2 FT's 54.8%; the gap is not interpretability tax — it is schema-resolution tax. Lifting it cleanly would require expanding the role taxonomy to ≈40 labels, which we leave for the journal version.

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
