#!/bin/bash
#SBATCH --job-name=irab-arat5v2
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/arat5v2_%j.out

# Stack B — full FT of AraT5v2-base-1024 (296M). ~6-10h on A100 40GB.
# Lower-risk fallback. Use first if you want a working baseline ASAP.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

# HF cache on BeeGFS so it survives node reboots
export HF_HOME=$HOME/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

OUT=runs/arat5v2_${SLURM_JOB_ID}
python -m irab_tashkeel.training.llm.arat5_sft \
    --config configs/llm_arat5v2.yaml \
    --output "$OUT"

echo "✓ done. checkpoints in $OUT"
