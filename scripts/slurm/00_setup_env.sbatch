#!/bin/bash
#SBATCH --job-name=irab-setup
#SBATCH --partition=debug_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/setup_%j.out

# One-time setup: create/activate the conda env, install heavy deps,
# build the dataset cache. Run this once before any training job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniconda3
module load cuda/12.4

source activate irab 2>/dev/null || { conda create -y -n irab python=3.11 && source activate irab; }

pip install -e ".[dev,app]"
pip install \
    "transformers>=4.46" \
    "accelerate>=1.0" \
    "peft>=0.13" \
    "bitsandbytes>=0.45" \
    "datasets>=3.1" \
    "trl>=0.12" \
    "huggingface_hub>=0.26" \
    "openai>=1.55" \
    "anthropic>=0.40"

# FlashAttention-2 (compile from wheel — may take a few minutes)
pip install flash-attn==2.7.* --no-build-isolation || \
    echo "⚠ flash-attn install failed; QLoRA script will fall back to vanilla SDPA"

# Build the unified dataset cache (QAC + Tashkeela + I3rab + synthetic + Yarob + PADT).
# Force rebuild so the new schema with irab_targets is materialized.
python -m irab_tashkeel.training.cli \
    --config configs/model_small.yaml \
    --force-rebuild \
    --dataset-cache data/cache/combined.pkl 2>&1 | tee logs/build_dataset.log &
BUILD_PID=$!
# Let it run only long enough to build + cache + train SP tokenizer; then kill.
# Watch for "Cached dataset" + "Saved tokenizer" to know when both are done.
sleep 600
kill -INT $BUILD_PID 2>/dev/null || true
echo "✓ env + dataset prep done"
