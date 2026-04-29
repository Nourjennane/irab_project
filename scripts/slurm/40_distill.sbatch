#!/bin/bash
#SBATCH --job-name=irab-distill
#SBATCH --partition=cpu_large_N_mix
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/distill_%j.out

# Generate ~10K MSA i'rāb samples by querying GPT-4o-mini (cheap).
# Set OPENAI_API_KEY in your shell before sbatch (e.g. via ~/.bashrc).
# Cost cap is enforced inside the script.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

module load miniconda3
source activate irab

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY not set; aborting"
    exit 1
fi

# Edit --n and --budget_usd to taste.
python -m irab_tashkeel.data.distill \
    --provider openai \
    --model gpt-4o-mini \
    --n 5000 \
    --budget_usd 25 \
    --out data/distilled_irab.jsonl

# After this completes, rebuild the dataset cache so the new samples
# get pulled in on the next training job:
#   python -m irab_tashkeel.training.cli --config configs/model_small.yaml --force-rebuild
