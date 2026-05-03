# Project Status Snapshot

Last updated: 2026-05-03 (auto-mode session). This is the single source of truth across runs while user is asleep / between sessions.

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
| Claude Sonnet 4.5 zero-shot | — | 78.4 | 72.4 | 76.0 | 44.0 | 27.6 |
| Sonnet RAG + AraT5v2 marker overlay (Hybrid v2) | — | 79.9 | 73.9 | 73.3 | 46.3 | 29.1 |
| **Claude Sonnet 4.5 + RAG (k=5)** *(headline)* | — | **79.9** | **73.9** | **74.6** | **50.0** | **32.1** |

Pending Gazelle eval: AraGPT2-large (HPC smoke RUNNING).

---

## 2. MASAQ eval surface (n=5,007 word judgments, 624 Quranic verses)

| System | Params | well | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|---:|
| Stanza Arabic | n/a | 59.0 | 44.6 | 14.9 | 14.8 | 5.2 |
| AraT5v2-base FT | 296M | 100.0 | 62.6 | 10.2 | 31.9 | 12.3 |
| mT5-base FT | 580M | (running) | | | | |
| AraGPT2-large FT | 792M | (pending) | | | | |
| Qwen2.5-7B + RAG | 7B | (paused at 17%; will resume) | | | | |

**MASAQ paired vs AraT5v2-base** (the only complete comparison so far):
- Stanza − AraT5v2 (n=5007): well −41.0, case −17.9, marker −17.0, fully −7.1 (all p<0.001 ★)

**Note on MASAQ vs Gazelle for AraT5v2-base:**
- Case: 62.6 (MASAQ) vs 65.7 (Gazelle) — close
- Role-F1: 10.2 (MASAQ) vs 54.2 (Gazelle) — collapse, register mismatch
- fully: 12.3 (MASAQ) vs 24.6 (Gazelle) — about half

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

| Job | Where | Status | ETA |
|---|---|---|---|
| AraGPT2-large smoke (487441) | HPC | RUNNING since ~04:55 | ~3-5 min |
| mT5-base on MASAQ | local 4060 | RUNNING (~3/624 verses) | ~50 min |

**On deck after AraGPT2 smoke passes:**
- Submit AraGPT2-large full sbatch (use `46_train_irab_jais_full.sbatch` — points to same config)
- After AraGPT2 full finishes (~5-7h): rsync + Gazelle + MASAQ eval

**Resumed after mT5 MASAQ finishes:**
- Restart Qwen on MASAQ (was at 111/624 = 18% before being paused)

---

## 7. Models trained + checkpoints

| Model | Path | Status |
|---|---|---|
| AraT5v2-base FT (Phase 2.1) | `runs/irab_arat5v2_distill_v2_487235/final/` (1.4 GB local) | ✅ done, evaluated |
| mT5-base FT (Phase 2.3) | `runs/irab_mt5_base_distill_v2_487432/final/` (2.2 GB local) | ✅ done, Gazelle ✓ MASAQ pending |
| AraGPT2-large FT (Phase 2.4) | (pending — smoke 487441 first) | smoke RUNNING |

---

## 8. Substitutions made (with reasons)

| Asked for | Substituted with | Reason |
|---|---|---|
| AraT5v2-large (1.2B) | AraT5-base v1 (~296M, FAILED smoke) → mT5-base (580M) | UBC-NLP doesn't release AraT5 large; v1 base is same size as v2 base; mT5-base is closest non-Arabic-specific T5 intermediate |
| Jais-family-1p3b (1.3B) | AraGPT2-large (~792M) | Jais is gated on HF (401 unauthorized); AraGPT2 is non-gated and same GPT-2 architecture so SFT script works unchanged |
| 5K Sonnet RAG eval on MASAQ ($3.56) | (deferred) | Spend-discipline guard blocked auto-spend; needs explicit user re-confirmation on wake |

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

1. **Sonnet RAG on MASAQ ($3.56)**: blocked by spend-discipline guard. Need explicit user re-confirmation to spend. Without it, MASAQ table has no Sonnet number — can still claim Sonnet RAG headline on Gazelle, but no cross-register Sonnet check.
2. **Jais auth**: if user wants Jais 1.3B specifically, they need to log into HF on HPC. Otherwise AraGPT2-large is the substitute.
3. **Qwen MASAQ resume**: was at 17% when paused. Restart will lose progress. Worth it if user wants Qwen MASAQ row populated.
