# Handoff Brief — Push Arabic i'rāb Project from B+/A- to 30/30

**You are picking up a near-complete NLP-class project on Arabic i'rāb (إعراب) generation.**
The student has ~4-5 days. The infrastructure is built; the missing pieces are a fine-tuning experiment, a real benchmark, and a writeup. Below is everything you need to take it home.

---

## 1. The objective

Given an undiacritized Arabic sentence, produce **per-word traditional i'rāb** (full grammatical analysis as Arabic prose) PLUS the diacritized form. Example:

```
INPUT:   ذهب الطالب إلى المدرسة
OUTPUT:
  ذهب     | ذَهَبَ          | فعل ماضٍ مبني على الفتح
  الطالب  | الطَّالِبُ      | فاعل مرفوع وعلامة رفعه الضمة الظاهرة
  إلى     | إِلَىٰ          | حرف جر مبني لا محل له من الإعراب
  المدرسة | الْمَدْرَسَةِ    | اسم مجرور بحرف الجر وعلامة جره الكسرة الظاهرة
```

---

## 2. Repo state

```
/home/hatem/Desktop/irab_project/    (git: HatemSaadallah/irab_project, branch main)
├── src/irab_tashkeel/
│   ├── data/                        # all data loaders, distillation, schema
│   │   ├── qac.py, ud_arabic.py, yarob.py, gazelle.py, distill.py, distilled_loader.py
│   │   └── build_dataset.py         # combines all into MTLExample list
│   ├── models/                      # the existing per-word decoder (used as failed baseline)
│   ├── training/llm/
│   │   ├── format.py                # SFT chat formatter
│   │   ├── qlora_sft.py             # vanilla HF QLoRA (Stack A vanilla)
│   │   ├── qlora_unsloth.py         # Unsloth + Liger + packing (Stack A fast)
│   │   ├── arat5_sft.py             # full FT AraT5v2 on full i'rāb (Stack B)
│   │   └── marker_arat5_sft.py      # Mix A Phase 2: marker-only FT  ← MAIN BET
│   ├── inference/
│   │   ├── predictor.py             # the per-word decoder predictor
│   │   ├── llm_baselines.py         # Claude zero-shot + RAG (current best)
│   │   ├── constrained.py           # taxonomy-snap postprocessor
│   │   └── cli.py                   # one-shot CLI
│   └── evaluation/
│       ├── structural.py            # regex-FSM extractor → case/role/marker/fully metrics
│       ├── run_baselines.py         # 3-baseline evaluation harness
│       ├── marker_extract.py        # builds marker training data
│       └── prepare_gold_seed.py     # 200-sentence gold-seeder (not yet run)
├── app/app.py                       # Streamlit demo with Claude RAG backend +
│                                    #   self-consistency repair + correction panel
├── configs/                         # YAML configs for each training path
├── scripts/slurm/                   # Bocconi sbatch scripts (00-40)
├── data/
│   ├── distilled_irab.jsonl         # 601 Claude-distilled MSA pairs (~7,200 word judgments)
│   ├── marker_pairs.jsonl           # 8,815 (sentence, word, case, role) → marker pairs
│   ├── ud_padt/, yarob_src/, hf_cache/, quran-morphology.txt
│   └── irab_spm.model               # SP tokenizer for the per-word decoder
├── docs/
│   ├── PLAN_10DAY.md                # the day-by-day plan
│   ├── RESULTS.md                   # current eval table
│   ├── PIPELINE.md                  # architecture diagram
│   └── HANDOFF.md                   # ← THIS FILE
└── runs/
    ├── model_small/best.pt          # the per-word decoder (failed baseline, 32.8% case)
    ├── baseline_eval_v2/            # the current best eval results
    └── marker_smoke/final/          # smoke-trained AraT5v2 (proves pipeline; not useful yet)
```

**Conda env:** `irab` (already created). `pip install -e ".[dev,llm,distill,app]"` to ensure deps.

**Bocconi HPC is available** (account `3415496`, partition `stud` only — MIG-split A100 4g.40gb ≈ 22GB).

---

## 3. Current numbers (Gazelle eval, 30 sentences = 134 word judgments)

| System | well-formed | case | role-F1 | marker | **fully** |
|---|---:|---:|---:|---:|---:|
| Per-word decoder | 70.9% | 32.8% | 3.8% | 13.4% | TBD |
| Claude Haiku 4.5 zero-shot | 77.6% | 57.5% | 55.9% | 34.3% | TBD |
| **Claude Haiku 4.5 + RAG (Yarob+distilled, k=5)** | **79.9%** | **67.2%** | **68.8%** | **44.8%** | **27.6%** |

`fully` = case ∧ role ∧ marker all simultaneously correct. **27.6% is the headline aggregate** — only 1 in 4 words gets the complete i'rāb right. The Mix A bet is to lift this above 40%.

---

## 4. The bet — Mix A (per-word routing)

**Hypothesis:** Claude RAG is strong on case (67%) and role (69%) but weak on marker (45%). A small fine-tuned AraT5v2 specialized on **marker prediction only** *might* lift marker EM into the 50-65% range. Lower-end (50%) is more honest given:
- 8K training examples is thin for style-fitting
- Words that are hard for case/role are likely also hard for marker (correlated failure modes), so marker improvement on the easy cases doesn't translate cleanly to fully-correct lift
- The 30-step smoke test (loss 7.50 → 5.77) tells us the pipeline works, **not** that final accuracy will be high

**Realistic outcome range for `fully`:** 28-40%. Plan for the median (32-35%), not the optimistic end. If the hybrid lands at 30-34% you frame it as a modest documented improvement; if it lands ≥35% you have a clean win; if it lands ≤28% you write a negative result honestly.

**Architecture:**
```
sentence → Claude RAG → [{word, irab, case, role, marker}, ...]
                              │
                              ▼  for each word:
                AraT5v2-marker(case, role, sentence, word) → corrected_marker
                              │
                              ▼
                final per-word i'rāb with the marker overlaid
```

**Phase 2 (training) — code is already written**, see `src/irab_tashkeel/training/llm/marker_arat5_sft.py` and `configs/marker_arat5v2.yaml`. Smoke-tested locally (loss descended cleanly 7.50 → 5.77 over 30 steps on 100 examples).

**Phase 3 (inference) — NOT YET WRITTEN**: needs `inference/hybrid.py` that wires Claude RAG + the trained marker model.

---

## 5. Concrete tasks ranked by points-per-effort

### Tier 1 — must-do (in order)

#### Task 1.1: Run the marker fine-tune (HPC ONLY)
**Bocconi:** `ssh 3415496@10.35.5.3 'cd ~/irab_project && git pull && sbatch scripts/slurm/33_train_marker_arat5v2.sbatch'` (~3-4h on stud MIG slice).

Do **not** train locally — laptop must stay free for the writeup.

Output: a fine-tuned AraT5v2 at `runs/marker_arat5v2_<JOBID>/final/`. Once done, scp/git-pull it back so inference can use it.

**Whatever number comes back (30%, 38%, or 45% on `fully`), within 24h of starting the job, you have your headline.** Plan around it.

#### Task 1.2: Build the Hybrid inference pipeline
Create `src/irab_tashkeel/inference/hybrid.py`:

```python
# pseudocode — implement in real code
class HybridPredictor:
    def __init__(self, marker_model_path, rag_pool):
        self.tokenizer = AutoTokenizer.from_pretrained(marker_model_path)
        self.marker_model = AutoModelForSeq2SeqLM.from_pretrained(marker_model_path)
        self.rag_pool = rag_pool

    def predict(self, sentence: str) -> List[WordIrab]:
        # 1) Get Claude RAG analysis
        rag_items = claude_fewshot_rag(sentence, self.rag_pool, k=5)
        # 2) For each word, override marker with the AraT5v2 prediction
        for it in rag_items:
            prompt = f"أعرب علامة: {it.word} | في: {sentence} | الحالة: {it.case} | المحل: {it.role}"
            new_marker = self.marker_model.generate(...)  # decoder
            if new_marker != "<NO_MARKER>":
                it.marker = new_marker
                # Re-render the irab string with the new marker
                it.irab = rebuild_irab_with_marker(it.role, it.case, new_marker)
        return rag_items
```

Add `--baselines hybrid` to `evaluation/run_baselines.py` so it scores the hybrid system the same way as the others.

#### Task 1.3: Re-run all evals with `fully_correct_word` for every system
Currently `fully` is only computed for the latest combined-pool RAG run. Re-run on:
- Per-word decoder (`runs/baseline_eval/decoder.predictions.jsonl` exists)
- Claude zero-shot (`runs/baseline_eval/claude_zero.predictions.jsonl` exists)
- Claude RAG Yarob-only (re-run if needed)
- Claude RAG combined pool (`runs/baseline_eval_v2/claude_rag.predictions.jsonl` exists)
- **Hybrid** (new)

Update `docs/RESULTS.md` with the complete table.

#### Task 1.4: Build manual gold benchmark (200 sentences)
1. Run `python -m irab_tashkeel.evaluation.prepare_gold_seed --n 200 --model claude-sonnet-4-5 --budget_usd 12 --out data/gold_seed.jsonl` (~$10, ~30 min). Reads PADT, seeds Sonnet RAG outputs, writes JSONL.
2. **Hand-correct each row** by setting `verified=true` and editing the `irab` field where wrong. Realistically 30 sentences/h × 200 = ~7h of human work. (USER does this; we cannot.)
3. Wire `--eval gold` into `evaluation/run_baselines.py` to score against this benchmark.

#### Task 1.5: Final writeup (3-4 pages)
Sections:
- **Problem & contributions** — i'rāb as structured prediction; comparison framework; Mix A; self-consistency-repair finding
- **Related work** — Gazelle (Hijjawi et al. 2024), CamelParser2.0, AraT5v2, Sadeed
- **Data** — sources + preprocessing + the templated-vs-real distinction
- **Methodology** — structural-extraction metric (justify why chrF/BLEU are wrong); fully_correct_word as headline
- **Experiments** — full table (decoder vs Claude vs Hybrid)
- **Discussion** — when does each component carry weight; Claude self-disagreement rate; failure modes
- **Future work** — full QLoRA on 9B Arabic LLM; DPO on the marker head; live constrained decoding

### Tier 2 — only if Tier 1 lands clean and time remains

These are nice-to-haves for the writeup, **not contributions that move the grade from B+ to 30**. Don't sink a day into any of them. Skip entirely if Tier 1 + writeup is at risk.

- **Constrained decoding integration.** Wire `inference/constrained.py` into the Hybrid output. <2 hours. Adds a sentence to the writeup; mention in future work otherwise.
- **Quantify Claude's self-consistency rate.** Run the case-vs-diac postprocessor over the 601 distilled samples; report the disagreement %. ONE sentence in the writeup. Skip the rabbit hole unless the number is striking (>15%).
- **LLM-as-judge eval.** Skip unless the hybrid result is unclear and you need a tiebreaker.
- **Per-source ablation.** Skip — adds runs without changing the central claim.

### Tier 3 — skip

- Stack A 9B QLoRA. Different project.
- Stack B AraT5v2 standalone full FT. Less interesting given Mix A.

---

## 6. Constraints & budget

- **Time**: ~4-5 days remaining (2026-04-30 → ~2026-05-10).
- **GPU**: Bocconi `stud` partition only (MIG-split A100, 4g.40gb ≈ 22GB, max 1 day per job, 1 concurrent job per QoS limit). Local RTX 4060 (8GB) is feasible for AraT5v2 with `adamw_bnb_8bit` + grad checkpointing.
- **Anthropic API**: ~$25 remaining of original $30 budget (already spent ~$5 on distillation + eval). Hard cap any single run at $15.
- **OpenAI API**: not yet used; available for LLM-as-judge if needed.

---

## 7. What NOT to do

- **Don't retrain the per-word decoder.** It's structurally limited (single-vector cross-attention memory); more data won't fix it. Use it only as the documented failed-baseline.
- **Don't add more templated training data.** QAC-templated and PADT-templated are already saturating; the model just memorizes the rule. Real data (Yarob + distilled) > more template volume.
- **Don't replace the structural metric with chrF/BLEU.** Those reward almost-right-but-wrong outputs and would undermine your methodology.
- **Don't use the full distilled+templated set for Mix A FT.** `marker_pairs.jsonl` is already filtered; train on it directly.
- **Don't skip the manual gold benchmark.** 30-sentence Gazelle is too small for a serious headline number.
- **Don't paste API keys into committed files.** Use `os.environ["ANTHROPIC_API_KEY"]` only.

---

## 8. Quick-start commands

```bash
cd /home/hatem/Desktop/irab_project
source .venv/bin/activate || conda activate irab

# Sanity (should print row counts)
wc -l data/distilled_irab.jsonl data/marker_pairs.jsonl

# Train the marker model — HPC ONLY (do NOT run on the laptop)
ssh 3415496@10.35.5.3 'cd ~/irab_project && git pull && sbatch scripts/slurm/33_train_marker_arat5v2.sbatch'

# Re-run the current best eval (~$0.20)
ANTHROPIC_API_KEY=sk-... python -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle --baselines claude_rag --model claude-haiku-4-5 \
    --out runs/baseline_eval_v3

# Streamlit demo
ANTHROPIC_API_KEY=sk-... MODEL_CKPT=runs/model_small/best.pt \
    streamlit run app/app.py
```

---

## 9. Where to read for context

In order, if you want to ramp up fast:

1. `docs/PIPELINE.md` — architecture overview
2. `docs/RESULTS.md` — current numbers + interpretation
3. `docs/PLAN_10DAY.md` — the original 10-day plan (now ~Day 1)
4. `src/irab_tashkeel/inference/llm_baselines.py` — Claude RAG + retrieval (the current best system)
5. `src/irab_tashkeel/evaluation/structural.py` — the metric definition
6. `src/irab_tashkeel/training/llm/marker_arat5_sft.py` — the Mix A training script
7. `app/app.py` — the demo, including self-consistency repair logic

---

## 10. The honest grade picture

The grade is decided by **whether the writeup is sharp**, not by whether the hybrid hits 30% or 40% on `fully`.

- **Floor (panic + spread thin):** spreading effort across 6 sub-experiments with 4 days remaining lands at 25-28. Each Tier 2 item that gets half-done makes the writeup messier, not stronger.
- **Realistic ceiling (focused execution):** working hybrid (any result) + ≥50-sentence hand-corrected benchmark + clean 3-page writeup framing the contribution as a comparison study + methodology = **28-30 range**.
- **30L is not on the table** in 4 days. That requires a result that surprises the grader; you don't have time to manufacture one.

**The contribution that lands:**

> "We compare three approaches to per-word Arabic i'rāb generation: a from-scratch character decoder (32.8% case), zero-shot Claude (57.5%), and retrieval-augmented Claude (67.2%). We propose a hybrid that combines RAG for case/role assignment with a fine-tuned AraT5v2 for marker phrasing, hypothesizing that case/role are knowledge-bound while marker is style-bound. We report results, including a [N]-sentence manually-validated benchmark."

That structure is defensible at 28-30 regardless of whether the hybrid wins by 10 points or 2.

**The bet that actually matters:** can you write a paper-shaped writeup in 2-3 days? If yes, you're at 28-30 regardless of hybrid result. If no — if you're going to spend 4 days on more experiments and 6 hours hastily writing — you're at 25-28 regardless of how good the experiments were.

**Pick the writeup. Run only the experiments that strengthen its central claim.**

---

*Brief written 2026-04-30 by Claude Opus 4.7 working on this codebase. Last commit: `415c2e4`.*
