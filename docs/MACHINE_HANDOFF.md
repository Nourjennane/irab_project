# Machine handoff — moving the Arabic i'rāb project to a new laptop

Date: 2026-05-06.
Latest commit: `ae50374` on `main`.
Repo: `https://github.com/HatemSaadallah/irab_project`.

This document captures the live state and the steps to bring it up on a new machine. It is the cross-machine resumption brief; for the project narrative, read `docs/HANDOFF.md`. For day-to-day status of phases / numbers / paired stats, read `docs/STATUS_SNAPSHOT.md`.

---

## 1. What is in git (everything you need is here)

```
git clone https://github.com/HatemSaadallah/irab_project
cd irab_project
git checkout main          # latest = ae50374
```

The repo carries:
- All source: `src/irab_tashkeel/{data, models, training, inference, evaluation}/`
- All configs: `configs/*.yaml`
- All sbatch scripts: `scripts/slurm/*.sbatch`
- The paper: `docs/paper/REPORT.tex` + `REPORT.pdf` + `figures/*.pdf` + `generate_figures.py`
- All scoring scripts: `scripts/role_subset_scoring.py`, `scripts/cross_register_bootstrap.py`, `scripts/sonnet_masaq_batch.py`, etc.
- All docs: `docs/RESULTS.md`, `docs/STATUS_SNAPSHOT.md`, `docs/REPORT.md`, `docs/HANDOFF.md`
- Smoke / final adapter for AraGPT2-large: `runs/irab_aragpt2_distill_v2_487443/final/` (200 MB, in git)
- Smoke / final adapter for AceGPT-13B: `runs/irab_acegpt13b_distill_v2_487888/final/` (200 MB, in git)
- Prediction JSONLs for every system except Sonnet RAG/zero-shot MASAQ live in `runs/baseline_eval_*/`. Sonnet API outputs are gitignored (regeneratable from cache).

What is NOT in git (gitignored, sized):
- `data/distill_v2/` — 65 MB Haiku-distilled training corpus (5K Anthropic batch outputs)
- `data/masaq/` — extracted MASAQ source files
- `data/tashkeela/` — old tashkeel corpus (1.1 GB, can probably skip)
- `data/distilled_irab.jsonl` — 601-row Phase-1 distill
- `data/marker_pairs.jsonl` — 8,815 marker pairs (Mix A training)
- `data/gold_seed.jsonl` — 82-row PADT gold seed
- `data/perturbed_eval.jsonl` — 402-row perturbation audit
- `data/ambiguity_annotations.jsonl` — annotator-disagreement audit
- `data/cache/`, `data/hf_cache/`, `data/ud_padt/`, `data/ud_nyuad/`, `data/yarob_src/`
- `runs/` other than the two adapter dirs (most are prediction JSONLs already in git)
- `.venv/`, `__pycache__/`, `logs/`
- `*.pt` model checkpoints

---

## 2. State on this machine that is NOT in git (rsync these)

From `/home/hatem/Desktop/irab_project/`:

```
data/distill_v2/                       # 65 MB; Haiku-5K training corpus
data/distilled_irab.jsonl              # ~3 MB; Phase-1 Haiku-601 distill (RAG pool)
data/marker_pairs.jsonl                # ~3 MB; Mix A specialist training set
data/marker_pairs_yarob_only.jsonl     # ~600 KB; Yarob-only ablation
data/gold_seed.jsonl                   # ~50 KB; 82-row PADT gold seed
data/perturbed_eval.jsonl              # ~200 KB; extractor audit set
data/ambiguity_annotations.jsonl       # ~10 KB; second-annotator pass
data/masaq_eval.jsonl                  # 524 KB; the 624-verse held-out eval (in git? check)
runs/irab_aragpt2_distill_v2_487443/   # 1.4 GB local cache after rsync from HPC
runs/irab_acegpt13b_distill_v2_487888/ # adapter only is small; full has the partial MASAQ preds
```

Suggest `rsync -avz hatem@old-machine:Desktop/irab_project/data/  new-machine:Desktop/irab_project/data/` for these subtrees specifically.

Or use a cloud-storage sync (Google Drive / Dropbox) for the data/ subtree; it is small.

---

## 3. State on Bocconi HPC

Host: `3415496@10.35.5.3` (requires Bocconi VPN — `globalprotect connect --portal vpnstudents.unibocconi.it`).

```
~/irab_project/                                               # repo clone
~/acegpt13b/                                                  # 25 GB Llama-2 base + tokenizer
~/.conda/envs/irab/                                           # 5.5 GB conda env (pre-built)
~/irab_project/runs/irab_acegpt13b_distill_v2_487888/         # 408 MB; trained adapter + smoke
~/irab_project/runs/baseline_eval_acegpt13b_gazelle/          # full Gazelle predictions (134 words)
~/irab_project/runs/baseline_eval_masaq_acegpt13b/            # PARTIAL MASAQ: 129/624 verses (1,075 words)
~/.cache/huggingface/hub/datasets--UBC-NLP--gazelle_benchmark # Gazelle cached for offline use
```

BeeGFS quota: 91.32 / 93.13 GiB used (1.81 GiB free). Tight. Do not download anything new before clearing space.

Slurm queue currently empty. No active jobs. Last AceGPT-13B MASAQ partial run was job 488454 (TIMEOUT at 4h, 21% complete).

---

## 4. Secrets / external dependencies (must be set on new machine)

```
ANTHROPIC_API_KEY=...        # for Sonnet/Haiku API calls; not in repo. ~$13 of budget remains.
~/.ssh/id_ed25519            # Bocconi HPC public key authorised on 10.35.5.3
GlobalProtect VPN            # vpnstudents.unibocconi.it (required for HPC access)
git remote                   # already https-based, no creds needed if using gh
```

Plus a Python env. Install on new machine:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install bitsandbytes peft  # not in pyproject.toml; needed for AceGPT inference
```

`pyproject.toml` lists the rest (transformers 4.46.x, accelerate, anthropic, datasets, sentence-transformers, etc.).

---

## 5. Phases / current state (matches STATUS_SNAPSHOT)

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — Haiku distillation 5K (77K rows) | ✅ done | corpus at `data/distill_v2/` |
| Phase 2.1 — AraT5v2-base full FT | ✅ done | `runs/irab_arat5v2_distill_v2_*/` (deleted from HPC; local + git) |
| Phase 2.3 — mT5-base FT | ✅ done | local + git |
| Phase 2.4 — AraGPT2-large LoRA FT | ✅ done | adapter in git |
| Phase 2.5 — AceGPT-13B QLoRA FT | ✅ done | adapter in git, 16h train, 1 epoch |
| **Phase 2.5b — AceGPT-13B Gazelle eval** | ✅ done | `runs/baseline_eval_acegpt13b_gazelle/` |
| **Phase 2.5c — AceGPT-13B MASAQ eval** | 🟡 partial 21% | 129 of 624 verses scored (4h SLURM timeout). Reported as partial in paper. |
| Phase 2.6 — AceGPT-13B full MASAQ via chained 4h jobs | ⏸ paused per user | Code ready (resume support + max_new_tokens 96 + chained sbatch) but not yet submitted. Awaiting user go-ahead. |
| Sonnet RAG MASAQ (n=5,007) | ✅ done | full coverage |
| Sonnet zero-shot MASAQ (n=400 verses, 657 words) | ✅ done | confound rejection |
| Cross-register paired stats | ✅ done | results in REPORT and RESULTS.md |
| Per-construction error analysis | ✅ done | EXCEPTION + KANA_SISTERS 0% across all systems |
| Paper LaTeX | ✅ done | `docs/paper/REPORT.{tex,pdf}`, 6 pages, all figures + tables |
| Final writeup task #46 | 🟡 paper draft 1 done | a reviewer-style revision pass landed in commit `d76b9e6`. Polish + presentation still open. |

---

## 6. Open backlog

1. **Phase 2.6 chained MASAQ run.** Resume code is committed (`ae50374`); submit when ready. Expected ~12h compute split across 3–4 four-hour SLURM jobs. After it lands: replace partial 21% row in REPORT and RESULTS, drop the partial-MASAQ caveat from Limitations.
2. **Reddit r/learn_arabic scrape.** Couldn't fetch from this machine (Claude Code blocked). Try from new machine with browser or `curl` directly. Looking specifically for: i'rāb worked-out parses, learner-error patterns on case/role/marker, MSA-vs-Quranic register notes.
3. **Presentation slides.** Not started. Hovy's deck in `~/Downloads/12_writing_presenting.pdf` lists rules: 10 slides (1/min), 30 pt min font, dark background, one slide one thought, glass-shape structure (motivation → details → outlook).
4. **Submit final report on the exam date.** Per the deck, "Due on the exam date (please register)". Deadline ~2026-05-10.

---

## 7. Quick verification on the new machine

After clone + venv + data copy + secrets:

```bash
# 1. project tree compiles
.venv/bin/python -c "import irab_tashkeel; print('ok')"

# 2. paper compiles (xelatex required; on Debian: apt install texlive-xetex texlive-fonts-extra)
cd docs/paper && xelatex REPORT.tex

# 3. figures regenerate from JSONLs
.venv/bin/python docs/paper/generate_figures.py

# 4. Stanza baseline reproduces (cheapest sanity check)
.venv/bin/python -m irab_tashkeel.evaluation.run_baselines --eval gazelle --baselines stanza --out /tmp/sanity

# 5. paired stats reproduce (numerical-equality check)
PYTHONPATH=. .venv/bin/python scripts/role_subset_scoring.py
PYTHONPATH=. .venv/bin/python scripts/cross_register_bootstrap.py

# 6. HPC reachable
ssh 3415496@10.35.5.3 'echo HPC OK; squeue -u 3415496'
```

If all six pass, the move is clean.

---

## 8. Memory carryover (Claude Code only)

The `~/.claude/projects/-home-hatem-Desktop-irab-project/memory/` directory holds session-persistent notes. Key files:
- `MEMORY.md` — index
- `feedback_writeup_framing_rules.md` — never "scaling study", cap contributions at 2, paired stats always
- `feedback_no_claude_coauthor.md` — no Co-Authored-By trailer
- `feedback_post_distillation_spend_discipline.md` — tight API-spend rules
- `project_coauthor.md` — coauthor is Nour Jennane
- `project_deadline.md` — ~2026-05-10
- `project_api_budget.md` — ~$13 remaining
- `project_irab_pivot.md` — design log
- `project_future_work_drafts.md` — pre-approved Sadeed paragraph
- `compute_bocconi_hpc.md` — HPC operational notes

If continuing in Claude Code on the new machine, copy this directory across so the agent retains the rules.
