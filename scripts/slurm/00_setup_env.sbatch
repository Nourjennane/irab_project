#!/bin/bash
#SBATCH --job-name=irab-setup
#SBATCH --partition=medium_cpu
#SBATCH --time=02:00:00
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

# Build the unified dataset cache (QAC + Tashkeela + I3rab + synthetic + Yarob + PADT
# + distilled). Pure data work — no model training.
python - <<'PYEOF'
from pathlib import Path
from irab_tashkeel.data.build_dataset import build_combined_dataset, save_examples, report
ex = build_combined_dataset(
    tashkeela_n=50000,
    qac_max_verses=None,
    i3rab_path=None,
    synthetic_per_type=2000,
    seed=42,
    data_dir=Path("data"),
)
report(ex)
save_examples(ex, Path("data/cache/combined.pkl"))
print(f"✓ cached {len(ex)} examples")
PYEOF

# Train the i'rāb SentencePiece tokenizer from the cache.
python -m irab_tashkeel.models.irab_tokenizer \
    --dataset-cache data/cache/combined.pkl \
    --out data/irab_spm.model \
    --vocab-size 5000

echo "✓ env + dataset prep done"
