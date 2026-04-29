# SLURM scripts for Bocconi HPC

Run these in order from the repo root on the HPC login node.

## 0. One-time setup
```bash
sbatch scripts/slurm/00_setup_env.sbatch
```
Creates the `irab` conda env, installs deps (transformers, peft, bitsandbytes,
flash-attn, openai, anthropic), and builds the unified dataset cache to
`data/cache/combined.pkl`. ~30 min.

## 1. Smoke test (15 min)
```bash
sbatch scripts/slurm/10_smoke_test.sbatch
```
Verifies CUDA + Lightning + the per-word decoder all work on debug GPU.

## 2a. AraT5v2 fine-tune (Stack B — lower risk, ~6-10h)
```bash
sbatch scripts/slurm/20_train_arat5v2.sbatch
```
Full FT of UBC-NLP/AraT5v2-base-1024. Use this first if you want any
working seq2seq baseline ASAP.

## 2b. QLoRA on Fanar-1-9B-Instruct (Stack A primary — ~14h on H100)
```bash
sbatch scripts/slurm/30_train_qlora_fanar.sbatch
```
Higher ceiling, harder to debug. Submit only after Stack B is working.

## 2c. QLoRA on ALLaM-7B-Instruct (Stack A alternative — ~10h on A100)
```bash
sbatch scripts/slurm/31_train_qlora_allam.sbatch
```

## 3. (Optional) GPT-4o distillation for synthetic MSA pairs
```bash
export OPENAI_API_KEY=sk-...
sbatch scripts/slurm/40_distill.sbatch
```
Generates 5,000 distilled samples (~$25 with gpt-4o-mini). Edit `--n` and
`--budget_usd` in the script to scale. After this finishes, rerun the
dataset cache build so the distilled set gets included.

## Tips
- All logs go to `logs/`. Check with `tail -f logs/<job>_<jobid>.out`.
- `squeue -u $USER` to check status; `scancel <jobid>` to cancel.
- HF model + dataset caches live at `$HOME/.hf_cache` (BeeGFS, persistent).
- Adapters/checkpoints land in `runs/<job>_<jobid>/`.
