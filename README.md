# Per-Word Arabic Iʿrāb Generation — Research Repository

**📍 Current state (2026-05-09):** experimentation is complete. The
production system is **Phase 3-A** (AraT5v2-base + morphology + UD dep
features); see [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md) for
production paths, archival experiments, evaluation utilities, final
tables, and the frozen architecture policy.

The full research report is in [`docs/REPORT.md`](docs/REPORT.md) and
the paper LaTeX source is in [`docs/paper/REPORT.tex`](docs/paper/REPORT.tex).

---

## Project history (legacy README, kept for context)

The original project framing was an explainable Arabic NLP system that jointly predicted:
1. **Tashkīl** (diacritics) — `ذهب` → `ذَهَبَ`
2. **I'rāb** (grammatical role) — `ذهب` → *verb, past tense, indeclinable*
3. **Errors** (orthographic + grammatical) — flagging `الى` → `إلى`

The current scope narrowed to per-word iʿrāb generation only — see
`docs/REPORT.md` for the empirical case study and the path that led
to Phase 3-A.

Unlike pure neural diacritizers (CATT, Sadeed, CAMeL-BERT), this system attaches a **grammatical justification** to every diacritic it predicts. The i'rāb head acts both as an auxiliary training signal (regularizing the encoder) and as an interpretability layer (exposing the model's syntactic reasoning).

## Quick start

```bash
# Clone + install
git clone <repo-url> irab-tashkeel
cd irab-tashkeel
pip install -e ".[dev]"

# Download corpora
python scripts/download_data.py --all

# Train a small dev model (~5M params, CPU-friendly)
python -m irab_tashkeel.training.cli --config configs/model_small.yaml

# Train the production model (60M params, requires GPU)
python -m irab_tashkeel.training.cli --config configs/model_medium.yaml

# Evaluate
python -m irab_tashkeel.evaluation.eval_cli --checkpoint runs/medium/best.pt

# Run the Streamlit demo
export MODEL_CKPT=runs/medium/best.pt
streamlit run app/app.py
```

## What's in this repo

| Directory | Purpose |
|---|---|
| `src/irab_tashkeel/` | Python package — data, models, training, inference, rules |
| `configs/` | YAML model/training configs (small / medium / large) |
| `scripts/` | Command-line utilities (download data, upload to HF Hub) |
| `tests/` | Unit tests — run with `pytest` |
| `docs/` | Architecture, data, training, evaluation, deployment docs |
| `app/` | Streamlit demo (deploys to HF Spaces) |
| `notebooks/` | Jupyter notebooks for exploration and error analysis |

## Architecture (one-paragraph)

A character-level transformer encoder shared across three task heads:
- **Diac head** — per-character 15-way classifier (one class per possible diacritic or combination)
- **I'rāb head** — per-word 11-way classifier (fāʿil, mafʿūl bih, mubtadaʾ, khabar, etc.)
- **Error head** — BIO-tagger for hamza/tāʾ-marbūṭa/case errors

Trained jointly on **Tashkeela** (~75M words, diacritics only) + **QAC** (78K words, diacritics + i'rāb) + **I3rab Treebank** (600 MSA sentences, i'rāb only) + **synthetic error injections**. Weighted multi-task loss with per-sample masking so each example only contributes to the heads for which it has labels.

For the full story see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## For Claude Code users

If you're using Claude Code to develop this project, start by reading:
1. [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — which modules do what
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the model design
3. [`docs/DATA.md`](docs/DATA.md) — the corpora and schemas

Common tasks:
- **Add a new error type**: edit `src/irab_tashkeel/data/synthetic.py` + update `ERR_LABELS` in `src/irab_tashkeel/models/labels.py`
- **Tweak the grammar engine**: `src/irab_tashkeel/rules/grammar.py` — each rule is a method
- **Add an evaluation metric**: `src/irab_tashkeel/evaluation/metrics.py`
- **Change UI**: `app/app.py`

## Citations

Built on research contributions from:
- Alasmary et al. 2024 — CATT (character-based tashkīl transformer)
- Aldallal et al. 2025 — Sadeed (LLM-based diacritization)
- Dukes & Habash 2010 — Quranic Arabic Corpus morphology
- Halabi, Fayyoumi & Awajan 2021 — I3rab Dependency Treebank
- Obeid et al. 2020 — CAMeL Tools
- Zerrouki & Balla 2017 — Tashkeela corpus

## License

MIT. See `LICENSE`.
