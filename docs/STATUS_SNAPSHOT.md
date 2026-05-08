# Project Status Snapshot

Last updated: 2026-05-08 (Phase 3 v1 rebuild rev 2 landed — see §13 below). Single source of truth across runs / sessions / machines. For cross-machine resumption see `docs/MACHINE_HANDOFF.md`.

---

## 1. Headline numbers (eval surface = Gazelle, n=134 word judgments)

| System | Params | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|---:|
| Stanza Arabic (UD pipeline) | n/a | 59.7 | 35.1 | 10.9 | 13.4 | 5.2 |
| Qwen2.5-7B-Instruct + RAG (k=5, 4-bit) | 7B | 66.4 | 43.3 | 20.8 | 19.4 | 3.0 |
| Qwen2.5-7B + augmented RAG pool (1060→6057) | 7B | 72.4 | 41.8 | 23.6 | 21.6 | 3.0 |
| Claude Haiku 4.5 zero-shot | — | 77.6 | 57.5 | 55.9 | 40.3 | 18.7 |
| Claude Haiku 4.5 + RAG (k=5) | — | 79.9 | 67.2 | 68.8 | 44.8 | 27.6 |
| Hybrid (Haiku RAG + AraT5v2 marker overlay) | — | 77.6 | 67.9 | 65.9 | 41.0 | 26.1 |
| **mT5-base (580M) FT on Haiku-5K** | 580M | 79.9 | 61.9 | 31.3 | 32.8 | **18.7** |
| AraT5v2-base (296M) FT on Haiku-5K | 296M | 79.9 | 65.7 | 54.2 | 44.0 | 24.6 |
| AraGPT2-large (792M) LoRA FT on Haiku-5K | 792M | 79.9 | 64.9 | 54.6 | 43.3 | 26.1 |
| AceGPT-13B QLoRA FT on Haiku-5K | 13B | 79.1 | 66.4 | 54.1 | 43.3 | 25.4 |
| Claude Sonnet 4.5 zero-shot | — | 78.4 | 72.4 | 76.0 | 44.0 | 27.6 |
| Sonnet RAG + AraT5v2 marker overlay (Hybrid v2) | — | 79.9 | 73.9 | 73.3 | 46.3 | 29.1 |
| **Claude Sonnet 4.5 + RAG (k=5)** *(headline)* | — | **79.9** | **73.9** | **74.6** | **50.0** | **32.1** |

Pending Gazelle eval: (none — AraGPT2-large evaluated 2026-05-03).

---

## 2. MASAQ eval surface (n=5,007 word judgments, 624 Quranic verses)

| System | Params | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|---:|
| Stanza Arabic | n/a | 59.0 | 44.6 | 14.9 | 14.8 | 5.2 |
| AraT5v2-base FT | 296M | 100.0 | 62.6 | 9.6 (subset 24.3) | 31.9 | 11.4 |
| mT5-base FT | 580M | 100.0 | 57.0 | 9.2 (subset 18.6) | 28.4 | 11.0 |
| AraGPT2-large FT | 792M | 99.9 | 61.1 | 8.0 (subset 20.2) | 31.1 | 10.0 |
| AceGPT-13B QLoRA FT (n=1075/5007 partial) | 13B | 99.9 | 62.0 | 9.8 (subset 22.2) | 32.9 | 10.7 |
| Claude Sonnet 4.5 + RAG | — | (see prior row, fully 6.6) | | | | |

**MASAQ paired comparison: AraGPT2-large vs AraT5v2-base** (n=5007, the 792M decoder vs 296M seq2seq):
- case Δ −1.4 pp ★ (p<0.001), fully Δ −1.4 pp ★ (p<0.001), marker Δ −0.8 (p=0.049)
- AraGPT2-large is paired-significantly *under* AraT5v2-base on MASAQ even though they tied on Gazelle (all p=1.000) → larger Arabic decoder doesn't transfer better to Quranic register than smaller Arabic seq2seq.

**Cross-register Δ (Gazelle role-F1 subset − MASAQ role-F1 subset):**
- stanza −7.2 pp [−11.8, +1.9] (ns)
- mt5_base +14.2 ★
- arat5_base +34.7 ★
- aragpt2_large +37.9 ★ (similar to arat5_base, CI overlaps)
- sonnet_rag +61.7 ★ (largest drop — Sonnet RAG suffers most from Quranic-register shift)
| Qwen2.5-7B + RAG | 7B | (paused at 17%; will resume) | | | | |

**MASAQ paired vs AraT5v2-base** (the only complete comparison so far):
- Stanza − AraT5v2 (n=5007): well −41.0, case −17.9, marker −17.0, fully −7.1 (all p<0.001 ★)

**Note on MASAQ vs Gazelle for AraT5v2-base:**
- Case: 62.6 (MASAQ) vs 65.7 (Gazelle) — close, **real cross-register comparable**
- Role-F1: 10.2 / 9.6 (MASAQ pre-fix / post-fix) vs 54.2 / 54.8 (Gazelle pre-fix / post-fix) — **measurement artifact, NOT a finding**. Two-stage diagnosis (`docs/MASAQ_role_audit.md`):
  - First, an extractor priority bug: longest-match-first wasn't enforced. Fixed in `structural.py`. Gazelle role-F1 stable to ≤0.6 pp on 7/8 systems (qwen_rag shifted 1.6 pp, within bootstrap CI). Effect on MASAQ AraT5: 10.2 → 9.6 (−0.6 pp). Necessary fix in principle but small empirical effect.
  - Second, the dominant 84% of disagreements come from the model producing verbose Quranic-commentary-style paraphrases that don't contain any canonical role term (e.g. `"الباء حرف جر... اسم مجرور وعلامة جره"`), so the extractor returns no role at all on the prediction side, while the MASAQ templater renders gold formulaically and the extractor finds the role there. This is a **templater-vs-model output-style mismatch**, not a cross-register effect.
  - Withdrawn as a finding. Reported as a methodological limitation: MASAQ role-F1 is not directly comparable to Gazelle role-F1 because the gold-text style differs.
- Marker: 31.9 (MASAQ) vs 44.0 (Gazelle) — **real cross-register comparable** (closed marker vocabulary, no template/output mismatch)
- fully: 12.3 (pre-fix) / 11.4 (post-fix) on MASAQ vs 24.6 / 25.4 on Gazelle — inherits the role artifact; cross-register fully comparison is weakened. Cite with caveat.

---

## 3. Cross-system findings (Gazelle, paired stats)

All ★ = McNemar p<0.05 + bootstrap CI excludes 0.

- **Sonnet RAG > Haiku RAG**: case Δ +6.7 ★ (p=0.035) — first paired-significant headline gain
- **Sonnet RAG > Stanza**: case Δ +38.8 ★, fully Δ +26.9 ★ (p<0.001)
- **Sonnet RAG > Qwen-7B+RAG**: case Δ +30.6 ★, fully Δ +29.1 ★ (p<0.001)
- **Sonnet RAG > AraT5v2-base FT**: case Δ +8.2 ★ (p=0.013), fully Δ +7.5 ★ (p=0.021)
- **AraT5v2-base FT > Qwen-7B+RAG**: case Δ +22.4 ★, fully Δ +21.6 ★ (p<0.001) — open-weight scaling: training matters more than parameter count
- **mT5-base FT < AraT5v2-base FT** (580M vs 296M): mT5 fully 18.7 vs AraT5v2 24.6 — Arabic-specific pretraining beats raw scale
- **Mix A negative result holds on both Haiku and Sonnet bases**: routing hypothesis rejected
- **MASAQ-augmented Sonnet pool (1060→2560)**: honest negative (fully −0.7, p=1.000)
- **Distill_v2-augmented Qwen pool (1060→6057)**: honest negative (fully unchanged 3.0)

---

## 4. Failure modes (per-construction analysis on Gazelle)

Cross-system 0% failures (all 5 systems including Sonnet RAG):
- **EXCEPTION (istithnāʾ)**: 0/9 across all systems
- **KANA_SISTERS**: 0/7 across 4 of 5 systems (haiku_rag scrapes 14.3%)

These hold for the trained AraT5v2-base too → failure is structural, not Claude-specific.

---

## 5. Methodology pieces shipped

- **Bootstrap CIs + paired bootstrap + McNemar's exact** for every comparison (`stats.py`)
- **Per-construction error analysis** with 11 tags (`error_analysis.py`)
- **Ambiguity analysis**: 31% of Gazelle words admit alternative analyses; permissive scoring lifts case +0.7 pp on Sonnet RAG, marker/fully unchanged
- **Extractor audit via perturbations**: specificity 100%, marker sens 92.7%, case 88.2%, role 60% (transparent metric limit)
- **System discrimination on perturbed gold**: Sonnet RAG case 96→15 on flip (Δ −81 pp), confirms discriminative behavior
- **Sensitivity ablations**: k ∈ {1,3,5,8,12}; pool 2x2 (Yarob/distilled/+MASAQ); prompt format alt; Sonnet repro variance (96.8% per-word agreement)
- **Reproducibility doc**: `reproducibility/REPRODUCIBILITY.md` + saved prompts + variance.md
- **MASAQ eval surface (NEW)**: 5,007 word judgments, ~6× tighter CIs than Gazelle

---

## 6. Currently running jobs

None as of 2026-05-06. SLURM queue is empty; no local background jobs. The chained AceGPT-13B MASAQ resume is staged in code (`scripts/slurm/48b_eval_acegpt13b_masaq_resume.sbatch`) but **not submitted** — user paused submission on 2026-05-06 to assess timing.

**Pending background work (parked, code ready):**
- AceGPT-13B MASAQ chained resume — `acegpt_irab.py` `max_new_tokens` lowered to 96; `evaluate_baseline(resume=True)` adds skip-if-already-scored logic. 4h sbatch to be resubmitted ~3× to clear remaining 495 verses (3,932 words). Target: replace partial-21% row with full $n{=}5{,}007$ row, drop the partial-MASAQ caveat from Limitations.

---

## 7. Models trained + checkpoints

| Model | Path | Status |
|---|---|---|
| AraT5v2-base FT (Phase 2.1) | `runs/irab_arat5v2_distill_v2_487235/final/` (deleted from HPC after rsync; local + git) | ✅ done, evaluated Gazelle + MASAQ |
| mT5-base FT (Phase 2.3) | `runs/irab_mt5_base_distill_v2_487432/final/` (deleted from HPC after rsync; local + git) | ✅ done, evaluated Gazelle + MASAQ |
| AraGPT2-large LoRA (Phase 2.4) | `runs/irab_aragpt2_distill_v2_487443/final/` (200 MB; local + git) | ✅ done, evaluated Gazelle + MASAQ (full) |
| AceGPT-13B QLoRA (Phase 2.5) | `runs/irab_acegpt13b_distill_v2_487888/final/` (200 MB; local + git; HPC base at `/home/3415496/acegpt13b/`) | ✅ trained 16h13m / 1 epoch / eval_loss 0.059. Gazelle eval ✓; MASAQ eval **partial 21%**. |

---

## 8. Substitutions made (with reasons)

| Asked for | Substituted with | Reason |
|---|---|---|
| AraT5v2-large (1.2B) | AraT5-base v1 (~296M, FAILED smoke) → mT5-base (580M) | UBC-NLP doesn't release AraT5 large; v1 base is same size as v2 base; mT5-base is closest non-Arabic-specific T5 intermediate |
| Jais-family-1p3b (1.3B) | AraGPT2-large (~792M) | Jais 1.3B gated on HF (401); AraGPT2 non-gated and same GPT-2 architecture so SFT script works unchanged |
| Jais-13B | AceGPT-13B (FreedomIntelligence/AceGPT-13B, Llama-2 base) | All 4 Jais-13B repos returned 403; AceGPT-13B is non-gated and Arabic-extended |

---

## 9. API spend tracker

Total budget: $50.

| Item | Spend |
|---|---:|
| Distillation v1 (Haiku 601 PADT, prior session) | $2.00 |
| Distillation v2 (Haiku 5K, this session) | $29.89 |
| Sonnet variance + ambiguity + k-sweep + prompt-alt + distilled-only + Qwen | ~$3.50 |
| **Total spent** | **~$35.39** |
| **Remaining** | **~$14.61** |

Allowed remaining (per spend-discipline memory): ambiguity rerun ~$3, Mix A re-test ~$4, eval reruns + buffer ~$5, hard reserve ~$1.

---

## 10. HPC operational notes

- Bocconi `stud` partition, MIG slice (3g.40gb = 40 GB)
- QoS: 1 concurrent job per user
- Need `--account=3415496` directive in sbatch (added retroactively after 4 failures)
- `save_total_limit=1, save_only_model=True` required to fit 50 GB user quota
- Prior failure modes documented: Phase 2.1 AraT5v2 final-save crash (disk quota), Jais gated-repo, AraT5v2-large doesn't exist
- 4/5 most recent HPC training jobs completed cleanly (mT5 487432 = first full-run-with-clean-save success)

---

## 11. Files for the writeup

- `docs/RESULTS.md` — main findings table + paired stats + per-construction + ambiguity + perturbation + variance + scaling
- `docs/REPORT.md` — narrative report (in progress)
- `docs/STATUS_SNAPSHOT.md` — this file
- `docs/future_work_drafts/sadeed_multitask.md` — pre-approved Future Work paragraph
- `data/distill_v2/STATS.md` — distillation corpus statistics
- `reproducibility/REPRODUCIBILITY.md` — reproducibility + ethics
- `reproducibility/variance.md` — Sonnet RAG inference-variance check
- `reproducibility/prompts/*.ar.txt` — verbatim prompts

---

## 12. Open questions / decisions waiting on user

1. **Submit AceGPT-13B chained MASAQ resume?** Code is committed (`ae50374`); awaits user go-ahead. Would clear the partial-21% caveat in the paper at ~12h compute split across 3–4 four-hour SLURM jobs.
2. **Reddit r/learn_arabic scrape.** WebFetch is blocked from this Claude Code instance for reddit.com. Try from new machine (browser, `curl`, or different agent).
3. **Presentation slides.** Not started; Hovy's deck rules in `~/Downloads/12_writing_presenting.pdf` (10 slides @ 1/min, 30 pt min, dark background, glass-shape structure).

---

## 13. Phase 3 — Interpretable structured-prediction baseline (v1 rebuild)

Landed 2026-05-08. Replaces the original char-decoder Appendix A baseline with a credible interpretable system. See REPORT.md §5.6 / REPORT.tex \subsection{Interpretable structured-prediction baseline}.

**Architecture:**
- Encoder: AraT5v2-base **encoder half** (296M, T5EncoderModel.from_pretrained — drops the cyclic-ref decoder)
- 4 classification heads: case (5) / role (25) / marker (18) / POS (6)
- 4 soft logit-bias symbolic constraint families: prep→jarr, inna sisters, kāna sisters, iḍāfa stub
- Deterministic Arabic-prose template renderer (canonical-tuple → prose)
- Quranic grammar-memory retriever (`retrieval/grammar_memory.py`): construction-aware Jaccard over MASAQ (624 verses; tag distribution PREP 345 / RELATIVE 301 / IDAFA 170 / INNA 141 / COORD 72 / EXCEPTION 71 / KANA 66 / VOCATIVE 41)
- Gradio demo (`app/structured_demo.py`): per-word table + prose + grammar-memory panel

**Training:** 6 epochs, label_smoothing=0.1, sqrt-inv-freq role class weights, first-subtoken pooling, best-checkpoint retention. ~4 min wall-clock on the stud MIG slice.

**Numbers (Gazelle, n=134):**
| System | well | case | role-F1 | marker | fully |
|---|---:|---:|---:|---:|---:|
| Original v1 (char dec) | 70.9 | 32.8 | 3.8 | 13.4 | 2.2 |
| Rebuild rev 1 (3 ep, mean pool) | 79.9 | 56.0 | 28.4 | 41.0 | 14.2 |
| **Rebuild rev 2** (ls + cw + first + 6 ep) | 79.9 | 55.2 | **36.9** | 41.0 | 17.9 |
| **Rebuild rev 2 + 4 constraints** | 79.9 | 55.2 | **36.9** | 41.0 | **18.7** |
| Reference: AraT5v2-base FT seq2seq | 79.9 | 65.7 | 54.8 | 44.0 | 24.6 |
| Reference: Sonnet RAG headline | 79.9 | 73.9 | 74.7 | 50.0 | 32.1 |

**Numbers (MASAQ, n=5,007):**
- Rebuild rev 2 + constraints: case 84.3% / role-F1 10.9% / marker-EM 30.6% / fully 7.9%
- Cross-register cost of MSA-frequency role weighting: rev 2 lifts MASAQ case (+2.2 pp) but lowers MASAQ role-F1 (−5.9 pp vs rev 1); reported transparently as a register-aware-weighting future-work lever.

**Files:**
- `src/irab_tashkeel/structured/{schema,word_irab,model,dataset,crf}.py` — schema + dataclass + multi-head model + dataset + linear-chain CRF (built but disabled in default config; see Phase 4 below)
- `src/irab_tashkeel/training/structured/train.py` — HF Trainer wrapper with role-weight + label-smoothing + first-pool + CRF init
- `src/irab_tashkeel/inference/{structured_predictor,symbolic_constraints,template_renderer,qualitative_trace}.py` — inference layer (9 constraint families implemented; rev 2 uses 4)
- `src/irab_tashkeel/retrieval/{jaccard_retriever,grammar_memory}.py` — Jaccard + Quranic grammar memory
- `app/structured_demo.py` — Gradio demo with attention heatmap + grammar-memory panel + reasoning trace
- `configs/structured_v1_rebuild.yaml` — single-source config (use_crf_role: false by default)
- `scripts/slurm/{49_smoke,50_train,51_eval}_structured_v1.sbatch` — HPC drivers (HF_HUB_OFFLINE=1 set for compute nodes)
- `scripts/structured/{build_structured_corpus,eval_structured_v1,generate_rare_construction_aug}.{py,sh}` — corpus + eval + augmentation script (needs ANTHROPIC_API_KEY to run)
- `data/structured_v1/{train,val}.jsonl` — 4747+250 sentences, 77K words (canonical labels)
- `runs/structured_v1_rebuild_490894/final/` — frozen rev 2 adapter (HPC + local)
- `runs/structured_v1_eval_490894/{gazelle,masaq}/` — frozen rev 2 eval predictions + summaries
- `runs/structured_v1_rebuild_490933/final/` — Phase 4 adapter (HPC; tested-and-rejected)
- `runs/structured_v1_eval_490933/{gazelle,masaq}/` — Phase 4 eval (regression vs rev 2)
- `docs/figures/qualitative_v1_rebuild.md` — 3-sentence qualitative trace from rev 2

---

## 14. Phase 4 — tested architectural upgrades (NEGATIVE RESULT)

Landed 2026-05-08, after rev 2. **Documented as an honest negative result; rev 2 stays as the frozen architecture.**

**What was added (all toggleable, all in code):**
1. Linear-chain CRF over the role head, empirical-bigram-initialised (`structured/crf.py`)
2. Five additional symbolic-constraint families (adjective agreement, coordination case-share, iḍāfa chain, naat propagation, vocative→nasb) — total now 9 (`inference/symbolic_constraints.py`)
3. Hierarchical role→case post-processing bias (`inference/symbolic_constraints.py::apply_hierarchical`)
4. Encoder attention extraction (`structured/model.py` with `output_attentions=True`)
5. Targeted rare-construction augmentation script (needs API key, not run)
6. Polished Gradio demo with attention heatmap + reasoning trace + grammar-memory panel
7. (One stable retrain, jobid 490933)

**Numbers (Gazelle, n=134):**

| Config | well | case | role-F1 | marker | fully |
|---|---:|---:|---:|---:|---:|
| Phase 4 heads only (CRF, hierarchical OFF, no extra constraints) | 79.1 | 50.0 | 37.9 | 31.3 | 11.9 |
| Phase 4 + 9 constraints + hierarchical | 79.9 | 49.3 | 30.9 | 32.1 | 11.9 |
| **Frozen rev 2 + 4 constraints (still best)** | 79.9 | **55.2** | **36.9** | **41.0** | **18.7** |

**Why it regressed:**
- CRF NLL plateaued at ~14 vs rev 2's CE at ~2 over the same 6 epochs → insufficient training of the structured loss at this scale
- The 9-constraint + hierarchical combination over-corrected role-F1 (37.9 → 30.9 within Phase 4) by stacking too many same-direction biases on the same per-word logit

**MASAQ (n=5,007):**
- Phase 4 + constraints: case 79.2 / role-F1 10.2 / fully 8.2  vs  rev 2 + 4 constraints: case 84.3 / role-F1 10.9 / fully 7.9
- Same regression pattern, slightly bigger gap on case (~5 pp).

**Default config restored to rev 2:** `configs/structured_v1_rebuild.yaml` has `use_crf_role: false`. The CRF + 5 extra constraints + hierarchical code remains in the repo (toggle to re-attempt with longer training); it's not on by default.

**Future-work levers from this iteration:**
- CRF-only retrain with more epochs (12+) and no class-weight interaction
- Per-constraint ablation to isolate net-helpful vs net-hurtful members of the larger constraint set
- Run the rare-construction augmentation (`scripts/structured/generate_rare_construction_aug.py`, needs API key)
